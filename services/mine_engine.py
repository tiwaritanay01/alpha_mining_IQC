"""Autonomous multi-generational Alpha Mining Engine."""

import time
from typing import Optional

from rich.console import Console

from services.experiment_service import (
    get_experiment,
    create_child_experiment,
    get_top_score,
)
from services.worldquant_client import WorldQuantClient
from services.generator import FieldAwareMutationEngine
from services.operator_service import get_all_operators
from services.field_service import get_all_fields
from services.pattern_memory import is_duplicate_structure
from services.pruning_service import prune_experiments

console = Console(force_terminal=True)

class MineEngine:
    """Orchestrates the multi-generational mining loop."""

    def __init__(self):
        self.client = WorldQuantClient()
        self.client.authenticate()
        
        operators = get_all_operators()
        fields = get_all_fields()
        self.generator = FieldAwareMutationEngine(
            operators=[op.name for op in operators] if operators else None,
            fields=[f.name for f in fields] if fields else None
        )

    def run(self, parent_id: Optional[int], generations: int = 5, top_k: int = 5, children_per_alpha: int = 10):
        """
        Execute the autonomous mining loop.
        """
        console.print(f"\n[bold cyan]🚀 STARTING AUTONOMOUS MINE ENGINE[/]")
        console.print(f"Generations: {generations} | Top K: {top_k} | Children/Alpha: {children_per_alpha}\n")
        
        current_generation = 1
        
        # If a specific parent is provided, we seed the first generation with it
        if parent_id is not None:
            seed_exp = get_experiment(parent_id)
            if not seed_exp:
                console.print(f"[red]Seed experiment #{parent_id} not found.[/]")
                return
            parents = [seed_exp]
            console.print(f"[dim]Seeded with Experiment #{parent_id}[/]")
        else:
            parents = get_top_score(limit=top_k)
            console.print(f"[dim]Seeded with top {len(parents)} experiments from DB[/]")

        while current_generation <= generations:
            console.print(f"\n[bold yellow]── GENERATION {current_generation} ──[/]")
            
            if not parents:
                console.print("[red]No valid parents found to continue mining.[/]")
                break

            generated_children = []
            
            # 1. Generate variants
            console.print(f"[dim]Generating mutations for {len(parents)} parents...[/]")
            for parent in parents:
                variants = self.generator.generate(parent.expression, count=children_per_alpha)
                
                for variant in variants:
                    # Memory deduplication
                    if is_duplicate_structure(variant):
                        continue
                        
                    child = create_child_experiment(
                        parent_id=parent.id,
                        theme=parent.theme,
                        expression=variant,
                    )
                    if child:
                        generated_children.append(child)

            if not generated_children:
                console.print("[yellow]No unique valid variants generated in this generation.[/]")
                break
                
            console.print(f"[green]Generated {len(generated_children)} unique valid children.[/]")

            # 2. Batch Simulate in chunks of 3 (Brain concurrency limit)
            console.print(f"[dim]Simulating {len(generated_children)} variants in chunks of 3 to respect API limits...[/]")
            success_count = 0
            
            chunk_size = 3
            for i in range(0, len(generated_children), chunk_size):
                chunk = generated_children[i:i + chunk_size]
                progress_urls = {}
                
                # Submit chunk
                for child in chunk:
                    sim_url = self.client.submit_simulation(child.expression)
                    if sim_url:
                        progress_urls[child.id] = sim_url
                    time.sleep(1.0) # Pace submissions
                
                # Poll chunk
                if progress_urls:
                    console.print(f"[dim]  Polling chunk {i//chunk_size + 1}/{(len(generated_children) + chunk_size - 1)//chunk_size} ({len(progress_urls)} sims)...[/]")
                    results = self.client.poll_simulations_batch(progress_urls, max_workers=chunk_size)
                    
                    from services.experiment_service import import_from_api_response
                    for exp_id, result in results.items():
                        if result:
                            updated = import_from_api_response(exp_id, result)
                            if updated and updated.score and updated.score > 0:
                                success_count += 1
                                
            console.print(f"[green]Simulation complete. {success_count}/{len(generated_children)} yielded positive scores.[/]")

            # 3. Prune weak alphas
            archived = prune_experiments(keep_top=50)
            console.print(f"[dim]Pruning complete. Archived {archived} weak alphas.[/]")

            # 4. Correlation-Aware Selection of Survivors
            # Instead of purely taking top-K, we want the most diverse top-K.
            try:
                from services.embedding_service import EmbeddingService
                embedding_svc = EmbeddingService()
                
                # Let's get the absolute top-K*2 first to have a pool
                pool = get_top_score(limit=top_k * 2)
                parents = []
                
                for exp in pool:
                    if len(parents) >= top_k:
                        break
                        
                    # Embed this experiment
                    text = f"{exp.theme} {exp.expression}"
                    emb = embedding_svc.embed_text(text)
                    embedding_svc.store_embedding(exp.id, emb)
                    
                    # Check similarity against already selected parents
                    is_correlated = False
                    for p in parents:
                        sim_scores = embedding_svc.find_similar(p.id, top_k=5)
                        # sim_scores is a list of (exp_id, score)
                        for s_id, s_score in sim_scores:
                            if s_id == exp.id and s_score > 0.90:  # 90% correlation proxy threshold
                                is_correlated = True
                                break
                        if is_correlated:
                            break
                    
                    if not is_correlated:
                        parents.append(exp)
                
                # If we filtered out too many, backfill with top scorers
                if len(parents) < top_k:
                    for exp in pool:
                        if len(parents) >= top_k:
                            break
                        if exp not in parents:
                            parents.append(exp)
                            
                console.print(f"[dim]Selected {len(parents)} diverse parents for next generation.[/]")
            except Exception as e:
                # Fallback to simple top-K if embeddings fail
                parents = get_top_score(limit=top_k)
            
            current_generation += 1

        console.print(f"\n[bold green]✅ MINING LOOP COMPLETE[/]")
