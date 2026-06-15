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

            # 2. Batch Simulate
            console.print("[dim]Queuing batch simulation...[/]")
            success_count = 0
            
            for child in generated_children:
                sim_url = self.client.submit_simulation(child.expression)
                if not sim_url:
                    continue
                    
                # In a real async loop we'd poll them all together, but for CLI we do sequentially or wait
                result = self.client.poll_simulation(sim_url)
                if result:
                    # Metrics are automatically updated by import_from_api_response inside poll if we used that
                    # Let's import it manually here since our client doesn't know the ID
                    from services.experiment_service import import_from_api_response
                    updated = import_from_api_response(child.id, result)
                    if updated and updated.score and updated.score > 0:
                        success_count += 1
                        
            console.print(f"[green]Simulation complete. {success_count}/{len(generated_children)} yielded positive scores.[/]")

            # 3. Prune weak alphas
            archived = prune_experiments(keep_top=50)
            console.print(f"[dim]Pruning complete. Archived {archived} weak alphas.[/]")

            # 4. Select next generation survivors
            parents = get_top_score(limit=top_k)
            
            current_generation += 1

        console.print(f"\n[bold green]✅ MINING LOOP COMPLETE[/]")
