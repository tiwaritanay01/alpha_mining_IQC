"""CLI commands for the Alpha Mining Machine."""

import json
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

# Force UTF-8 output on Windows to support Rich formatting
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from services.experiment_service import (
    create_experiment,
    create_child_experiment,
    get_all_experiments,
    get_experiment,
    get_top_sharpe,
    get_top_fitness,
    get_top_returns,
    get_recent_winners,
    get_best_generated,
    update_metrics,
    get_children,
    get_tree,
    search_experiments,
    get_theme_stats,
    get_best_themes,
    import_from_api_response,
)

app = typer.Typer(
    name="alpha",
    help="IQC Alpha Mining Machine -- Terminal Research Platform",
    rich_markup_mode="rich",
    no_args_is_help=True,
)
console = Console(force_terminal=True)


# ── EXPRESSION HELPERS ────────────────────────────────────────────────────────


def _truncate(text: str, max_len: int = 50) -> str:
    """Truncate a string with ellipsis if too long."""
    if not text:
        return "—"
    return text[:max_len - 1] + "…" if len(text) > max_len else text


def _fmt_float(value: Optional[float], decimals: int = 4) -> str:
    """Format a float or return dash if None."""
    if value is None:
        return "—"
    return f"{value:.{decimals}f}"


# ── PHASE 1: CORE COMMANDS (RICH) ────────────────────────────────────────────


@app.command()
def add():
    """Create a new root experiment."""
    theme = typer.prompt("Theme")
    expression = typer.prompt("Expression")
    notes = typer.prompt("Notes", default="")

    exp = create_experiment(theme, expression, notes)

    console.print(
        Panel(
            f"[bold green]Created Experiment #{exp.id}[/]\n\n"
            f"[dim]Theme:[/]      {exp.theme}\n"
            f"[dim]Expression:[/] {_truncate(exp.expression, 70)}\n"
            f"[dim]Notes:[/]      {exp.notes or '—'}",
            title="✅ New Experiment",
            border_style="green",
        )
    )


@app.command("list")
def list_experiments():
    """List all experiments in a rich table."""
    experiments = get_all_experiments()

    if not experiments:
        console.print("[yellow]No experiments found.[/]")
        return

    table = Table(
        title="📋 All Experiments",
        show_lines=False,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("ID", style="bold", justify="right", width=5)
    table.add_column("Theme", style="magenta", width=15)
    table.add_column("Expression", width=50)
    table.add_column("Gen", justify="center", width=4)
    table.add_column("Status", width=10)
    table.add_column("Sharpe", justify="right", width=8)

    for exp in experiments:
        status_color = "green" if exp.status == "tested" else "dim"
        table.add_row(
            str(exp.id),
            exp.theme,
            _truncate(exp.expression, 48),
            str(exp.generation),
            f"[{status_color}]{exp.status}[/]",
            _fmt_float(exp.sharpe),
        )

    console.print()
    console.print(table)
    console.print()


@app.command()
def show(exp_id: int = typer.Argument(..., help="Experiment ID")):
    """Show full details of an experiment."""
    exp = get_experiment(exp_id)

    if not exp:
        console.print(f"[red]Experiment #{exp_id} not found.[/]")
        raise typer.Exit(1)

    children = get_children(exp_id)

    detail = (
        f"[bold]ID:[/]          {exp.id}\n"
        f"[bold]Theme:[/]       {exp.theme}\n"
        f"[bold]Expression:[/]  {exp.expression}\n"
        f"[bold]Parent ID:[/]   {exp.parent_id or '—'}\n"
        f"[bold]Generation:[/]  {exp.generation}\n"
        f"[bold]Status:[/]      {exp.status}\n"
        f"[bold]Notes:[/]       {exp.notes or '—'}\n"
        f"[bold]Created:[/]     {exp.created_at}\n"
        f"\n"
        f"[bold cyan]── Metrics ──[/]\n"
        f"[bold]Sharpe:[/]      {_fmt_float(exp.sharpe)}\n"
        f"[bold]Fitness:[/]     {_fmt_float(exp.fitness)}\n"
        f"[bold]Turnover:[/]    {_fmt_float(exp.turnover)}\n"
        f"[bold]Returns:[/]     {_fmt_float(exp.returns)}\n"
        f"\n"
        f"[bold cyan]── Lineage ──[/]\n"
        f"[bold]Children:[/]    {len(children)}"
    )

    console.print()
    console.print(Panel(detail, title=f"🔬 Experiment #{exp.id}", border_style="cyan"))
    console.print()


@app.command()
def score(exp_id: int = typer.Argument(..., help="Experiment ID")):
    """Manually update performance metrics for an experiment."""
    sharpe = float(typer.prompt("Sharpe"))
    fitness = float(typer.prompt("Fitness"))
    turnover = float(typer.prompt("Turnover"))
    returns = float(typer.prompt("Returns"))

    exp = update_metrics(exp_id, sharpe, fitness, turnover, returns)

    if exp:
        console.print(f"[green]✅ Metrics updated for Experiment #{exp_id}[/]")
    else:
        console.print(f"[red]Experiment #{exp_id} not found.[/]")


@app.command()
def leaderboard():
    """Display top experiments ranked by various metrics."""
    
    def _print_table(title: str, experiments: list, highlight_col: str):
        if not experiments:
            return

        table = Table(
            title=title,
            header_style="bold yellow",
            border_style="dim",
        )
        table.add_column("Rank", style="bold", justify="right", width=5)
        table.add_column("ID", justify="right", width=5)
        table.add_column("Theme", style="magenta", width=15)
        table.add_column("Expression", width=45)
        
        table.add_column("Sharpe", justify="right", style="green" if highlight_col == "sharpe" else "", width=8)
        table.add_column("Fitness", justify="right", style="green" if highlight_col == "fitness" else "", width=8)
        table.add_column("Returns", justify="right", style="green" if highlight_col == "returns" else "", width=8)
        table.add_column("Gen", justify="right", width=4)

        for rank, exp in enumerate(experiments, 1):
            table.add_row(
                str(rank),
                str(exp.id),
                exp.theme,
                _truncate(exp.expression, 43),
                _fmt_float(exp.sharpe),
                _fmt_float(exp.fitness),
                _fmt_float(exp.returns),
                str(exp.generation),
            )

        console.print()
        console.print(table)

    console.print("\n[bold cyan]🏆 ALPHA MINING LEADERBOARD 🏆[/]\n")

    top_sharpe = get_top_sharpe(limit=5)
    _print_table("🥇 Top by Sharpe", top_sharpe, "sharpe")

    top_fitness = get_top_fitness(limit=5)
    _print_table("🏋️ Top by Fitness", top_fitness, "fitness")

    top_returns = get_top_returns(limit=5)
    _print_table("💸 Top by Returns", top_returns, "returns")

    recent_winners = get_recent_winners(limit=5)
    _print_table("🔥 Recent Winners (Sharpe >= 1.0)", recent_winners, "sharpe")

    best_generated = get_best_generated(limit=5)
    _print_table("🤖 Best Generated Variants", best_generated, "sharpe")
    
    console.print()

# ── PHASE 1: LINEAGE TREE ────────────────────────────────────────────────────


@app.command()
def tree(exp_id: int = typer.Argument(..., help="Root experiment ID")):
    """Display the experiment lineage tree."""
    tree_data = get_tree(exp_id)

    if not tree_data:
        console.print(f"[red]Experiment #{exp_id} not found.[/]")
        raise typer.Exit(1)

    rich_tree = _build_rich_tree(tree_data)

    console.print()
    console.print(rich_tree)
    console.print()


def _build_rich_tree(node: dict, is_root: bool = True) -> Tree:
    """Recursively build a Rich Tree from a tree_data dict."""
    exp = node["experiment"]
    label = _tree_label(exp, is_root)

    if is_root:
        rich_tree = Tree(label)
    else:
        rich_tree = Tree(label)

    for child_node in node.get("children", []):
        child_tree = _build_rich_tree(child_node, is_root=False)
        rich_tree.add(child_tree)

    return rich_tree


def _tree_label(exp, is_root: bool = False) -> str:
    """Format a tree node label."""
    status_icon = "✅" if exp.status == "tested" else "⬜"
    sharpe_str = f" [green]Sharpe={_fmt_float(exp.sharpe)}[/]" if exp.sharpe else ""

    if is_root:
        return (
            f"[bold cyan]Alpha #{exp.id}[/] {status_icon} "
            f"[dim]{_truncate(exp.expression, 60)}[/]{sharpe_str}"
        )
    return (
        f"[bold]Alpha #{exp.id}[/] {status_icon} "
        f"[dim]{_truncate(exp.expression, 55)}[/]{sharpe_str}"
    )


# ── PHASE 1: SEARCH ──────────────────────────────────────────────────────────


@app.command()
def search(query: str = typer.Argument(..., help="Search term")):
    """Search experiments by theme, expression, or notes."""
    results = search_experiments(query)

    if not results:
        console.print(f"[yellow]No experiments matching '{query}'.[/]")
        return

    table = Table(
        title=f"🔍 Search Results for '{query}'",
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("ID", style="bold", justify="right", width=5)
    table.add_column("Theme", style="magenta", width=15)
    table.add_column("Expression", width=50)
    table.add_column("Status", width=10)
    table.add_column("Sharpe", justify="right", width=8)

    for exp in results:
        table.add_row(
            str(exp.id),
            exp.theme,
            _truncate(exp.expression, 48),
            exp.status,
            _fmt_float(exp.sharpe),
        )

    console.print()
    console.print(table)
    console.print(f"\n[dim]{len(results)} result(s) found.[/]\n")


# ── PHASE 2: OPERATOR COMMANDS ────────────────────────────────────────────────


@app.command("sync-operators")
def sync_operators_cmd():
    """Pull operators from the WorldQuant Brain API and save to database."""
    from services.worldquant_client import WorldQuantClient
    from services.operator_service import sync_operators

    client = WorldQuantClient()

    console.print("[dim]Authenticating with WorldQuant Brain...[/]")
    if not client.authenticate():
        console.print("[red]❌ Authentication failed. Check your .env credentials.[/]")
        raise typer.Exit(1)

    console.print("[dim]Fetching operators...[/]")
    api_data = client.fetch_operators()

    if not api_data:
        console.print("[red]❌ No operators returned from API.[/]")
        raise typer.Exit(1)

    created, updated = sync_operators(api_data)

    console.print(
        Panel(
            f"[green]Created:[/] {created}\n"
            f"[yellow]Updated:[/] {updated}\n"
            f"[dim]Total from API:[/] {len(api_data)}",
            title="✅ Operators Synced",
            border_style="green",
        )
    )


@app.command("operators")
def list_operators(
    category: Optional[str] = typer.Argument(None, help="Filter by category"),
):
    """List all operators, optionally filtered by category."""
    from services.operator_service import get_all_operators, get_operator_categories

    operators = get_all_operators(category)

    if not operators:
        if category:
            console.print(f"[yellow]No operators in category '{category}'.[/]")
            cats = get_operator_categories()
            if cats:
                console.print(f"[dim]Available categories: {', '.join(cats)}[/]")
        else:
            console.print("[yellow]No operators found. Run 'sync-operators' first.[/]")
        return

    table = Table(
        title=f"📦 Operators{' — ' + category if category else ''}",
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Name", style="bold", width=25)
    table.add_column("Category", style="magenta", width=15)
    table.add_column("Description", width=50)
    table.add_column("Scope", width=10)

    for op in operators:
        table.add_row(
            op.name,
            op.category or "—",
            _truncate(op.description or "", 48),
            op.scope or "—",
        )

    console.print()
    console.print(table)
    console.print(f"\n[dim]{len(operators)} operator(s).[/]\n")


# ── DATA FIELD COMMANDS ───────────────────────────────────────────────────────

@app.command("sync-fields")
def sync_fields_cmd():
    """Pull data fields from the WorldQuant Brain API and save to database."""
    from services.worldquant_client import WorldQuantClient
    from services.field_service import sync_fields

    client = WorldQuantClient()

    console.print("[dim]Authenticating with WorldQuant Brain...[/]")
    if not client.authenticate():
        console.print("[red]❌ Authentication failed. Check your .env credentials.[/]")
        raise typer.Exit(1)

    console.print("[dim]Fetching data fields...[/]")
    api_data = client.fetch_data_fields()

    if not api_data:
        console.print("[red]❌ No data fields returned from API.[/]")
        raise typer.Exit(1)

    created, updated = sync_fields(api_data)

    console.print(
        Panel(
            f"[green]Created:[/] {created}\n"
            f"[yellow]Updated:[/] {updated}\n"
            f"[dim]Total from API:[/] {len(api_data)}",
            title="✅ Data Fields Synced",
            border_style="green",
        )
    )

@app.command("fields")
def list_fields(
    category: Optional[str] = typer.Argument(None, help="Filter by category"),
):
    """List all data fields, optionally filtered by category."""
    from services.field_service import get_all_fields

    # Can't easily import categories here without duplicating code, just get all and filter
    fields = get_all_fields()
    
    if category:
        fields = [f for f in fields if f.category and category.lower() in f.category.lower()]

    if not fields:
        console.print("[yellow]No data fields found. Run 'sync-fields' first.[/]")
        return

    table = Table(
        title=f"📊 Data Fields{' — ' + category if category else ''}",
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Name", style="bold", width=30)
    table.add_column("Category", style="magenta", width=15)
    table.add_column("Dataset", style="blue", width=15)
    table.add_column("Description", width=50)

    for f in fields:
        table.add_row(
            f.name,
            f.category or "—",
            f.dataset or "—",
            _truncate(f.description or "", 48),
        )

    console.print()
    console.print(table)
    console.print(f"\n[dim]{len(fields)} field(s).[/]\n")

@app.command("field")
def show_field(name: str = typer.Argument(..., help="Name of the data field")):
    """Show details for a specific data field."""
    from services.field_service import get_field

    f = get_field(name)
    if not f:
        console.print(f"[red]Data field '{name}' not found.[/]")
        return

    console.print(
        Panel(
            f"[bold cyan]Name:[/]        {f.name}\n"
            f"[bold magenta]Category:[/]    {f.category or '—'}\n"
            f"[bold blue]Dataset:[/]     {f.dataset or '—'}\n"
            f"[bold]Description:[/] {f.description or '—'}\n",
            title=f"📊 Field: {f.name}",
            border_style="cyan",
        )
    )

@app.command("search-fields")
def search_fields_cmd(query: str = typer.Argument(..., help="Search query")):
    """Search data fields by name, description, or dataset."""
    from services.field_service import search_fields

    fields = search_fields(query)

    if not fields:
        console.print(f"[yellow]No data fields matching '{query}'.[/]")
        return

    table = Table(
        title=f"🔍 Search Results for '{query}'",
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Name", style="bold", width=30)
    table.add_column("Dataset", style="blue", width=15)
    table.add_column("Description", width=60)

    for f in fields:
        table.add_row(
            f.name,
            f.dataset or "—",
            _truncate(f.description or "", 58),
        )

    console.print()
    console.print(table)
    console.print(f"\n[dim]{len(fields)} matching field(s).[/]\n")


# ── PHASE 3: SIMULATE ────────────────────────────────────────────────────────


@app.command()
def simulate(exp_id: int = typer.Argument(..., help="Experiment ID to simulate")):
    """Submit an experiment for simulation via the WorldQuant API."""
    from services.worldquant_client import WorldQuantClient

    exp = get_experiment(exp_id)
    if not exp:
        console.print(f"[red]Experiment #{exp_id} not found.[/]")
        raise typer.Exit(1)

    from services.generator import FieldAwareMutationEngine
    from services.operator_service import get_all_operators
    from services.field_service import get_all_fields

    operators = get_all_operators()
    fields = get_all_fields()
    engine = FieldAwareMutationEngine(
        operators=[op.name for op in operators] if operators else None,
        fields=[f.name for f in fields] if fields else None
    )

    if not engine.is_valid(exp.expression):
        console.print(f"[red]Validation failed. Expression contains unknown fields or operators.[/]")
        raise typer.Exit(1)

    console.print(
        Panel(
            f"[bold]Expression:[/] {exp.expression}\n"
            f"[bold]Theme:[/]      {exp.theme}",
            title=f"🚀 Simulating Experiment #{exp_id}",
            border_style="yellow",
        )
    )

    if not typer.confirm("Submit this simulation?"):
        console.print("[dim]Cancelled.[/]")
        return

    client = WorldQuantClient()

    console.print("[dim]Authenticating...[/]")
    if not client.authenticate():
        console.print("[red]❌ Authentication failed.[/]")
        raise typer.Exit(1)

    console.print("[dim]Submitting simulation...[/]")
    sim_progress_url = client.submit_simulation(exp.expression)

    if not sim_progress_url:
        console.print("[red]❌ Simulation submission failed.[/]")
        raise typer.Exit(1)

    console.print("[dim]Polling for results...[/]")
    result = client.poll_simulation(sim_progress_url)

    if not result:
        console.print("[red]❌ Simulation failed or timed out.[/]")
        raise typer.Exit(1)

    # Auto-import results
    updated_exp = import_from_api_response(exp_id, result)

    if updated_exp:
        console.print(
            Panel(
                f"[bold]Sharpe:[/]   {_fmt_float(updated_exp.sharpe)}\n"
                f"[bold]Fitness:[/]  {_fmt_float(updated_exp.fitness)}\n"
                f"[bold]Turnover:[/] {_fmt_float(updated_exp.turnover)}\n"
                f"[bold]Returns:[/]  {_fmt_float(updated_exp.returns)}",
                title="✅ Simulation Complete",
                border_style="green",
            )
        )
    else:
        console.print("[yellow]Simulation complete but could not parse metrics.[/]")
        console.print(f"[dim]Raw result: {json.dumps(result, indent=2)[:500]}[/]")


@app.command("simulate-batch")
def simulate_batch(
    exp_ids: list[int] = typer.Argument(..., help="List of Experiment IDs to simulate"),
):
    """Simulate a batch of experiments sequentially."""
    from services.worldquant_client import WorldQuantClient
    client = WorldQuantClient()

    console.print("[dim]Authenticating...[/]")
    if not client.authenticate():
        console.print("[red]❌ Authentication failed.[/]")
        raise typer.Exit(1)

    success_count = 0
    for exp_id in exp_ids:
        exp = get_experiment(exp_id)
        if not exp:
            console.print(f"[yellow]Experiment #{exp_id} not found. Skipping.[/]")
            continue
        
        console.print(f"\n[cyan]🚀 Simulating #{exp_id}...[/]")
        sim_url = client.submit_simulation(exp.expression)
        if not sim_url:
            console.print(f"[red]Failed to submit #{exp_id}.[/]")
            continue
            
        result = client.poll_simulation(sim_url)
        if not result:
            console.print(f"[red]Failed polling #{exp_id}.[/]")
            continue
            
        updated = import_from_api_response(exp_id, result)
        if updated:
            console.print(f"[green]✅ #{exp_id} -> Sharpe: {_fmt_float(updated.sharpe)}[/]")
            success_count += 1
        else:
            console.print(f"[yellow]⚠️ #{exp_id} completed but metrics not parsed.[/]")
            
    console.print(f"\n[bold green]Batch complete. Successfully simulated {success_count}/{len(exp_ids)} experiments.[/]")


@app.command("simulate-generated")
def simulate_generated(
    parent_id: int = typer.Argument(..., help="Parent experiment ID"),
):
    """Simulate all unscored children of a parent experiment."""
    from services.experiment_service import get_children
    
    children = get_children(parent_id)
    # Filter for unscored (generated) children
    to_simulate = [c.id for c in children if c.sharpe is None]
    
    if not to_simulate:
        console.print(f"[yellow]No unscored children found for parent #{parent_id}.[/]")
        return
        
    console.print(f"[cyan]Found {len(to_simulate)} unscored children. Starting batch simulation...[/]")
    simulate_batch(to_simulate)


# ── PHASE 4: IMPORT RESULT ───────────────────────────────────────────────────


@app.command("import-result")
def import_result(
    exp_id: int = typer.Argument(..., help="Experiment ID"),
    paste: bool = typer.Option(False, "--paste", help="Paste raw API JSON response"),
):
    """Import simulation results for an experiment."""
    exp = get_experiment(exp_id)
    if not exp:
        console.print(f"[red]Experiment #{exp_id} not found.[/]")
        raise typer.Exit(1)

    if paste:
        console.print("[dim]Paste the raw API JSON response (then press Enter twice):[/]")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)

        raw_json = "\n".join(lines)
        try:
            raw_response = json.loads(raw_json)
        except json.JSONDecodeError:
            console.print("[red]❌ Invalid JSON.[/]")
            raise typer.Exit(1)

        updated = import_from_api_response(exp_id, raw_response)
        if updated:
            console.print(
                Panel(
                    f"[bold]Sharpe:[/]   {_fmt_float(updated.sharpe)}\n"
                    f"[bold]Fitness:[/]  {_fmt_float(updated.fitness)}\n"
                    f"[bold]Turnover:[/] {_fmt_float(updated.turnover)}\n"
                    f"[bold]Returns:[/]  {_fmt_float(updated.returns)}",
                    title=f"✅ Imported Results for #{exp_id}",
                    border_style="green",
                )
            )
        else:
            console.print("[red]❌ Could not parse metrics from response.[/]")
    else:
        # Manual mode
        sharpe = float(typer.prompt("Sharpe"))
        fitness = float(typer.prompt("Fitness"))
        turnover = float(typer.prompt("Turnover"))
        returns = float(typer.prompt("Returns"))

        updated = update_metrics(exp_id, sharpe, fitness, turnover, returns)
        if updated:
            console.print(f"[green]✅ Metrics imported for Experiment #{exp_id}[/]")
        else:
            console.print(f"[red]Failed to update Experiment #{exp_id}.[/]")


# ── PHASE 5: GENERATE VARIANTS ───────────────────────────────────────────────


@app.command()
def generate(
    exp_id: int = typer.Argument(..., help="Parent experiment ID"),
    count: int = typer.Option(30, "--count", "-n", help="Max variants to generate"),
):
    """Generate smart variants of an experiment using the mutation engine."""
    from services.generator import FieldAwareMutationEngine
    from services.operator_service import get_all_operators
    from services.field_service import get_all_fields

    parent = get_experiment(exp_id)
    if not parent:
        console.print(f"[red]Experiment #{exp_id} not found.[/]")
        raise typer.Exit(1)

    # Load operators and fields from DB for mutation
    operators = get_all_operators()
    operator_names = [op.name for op in operators] if operators else None

    fields = get_all_fields()
    field_names = [f.name for f in fields] if fields else None

    engine = FieldAwareMutationEngine(operators=operator_names, fields=field_names)
    variants = engine.generate(parent.expression, count=count)

    if not variants:
        console.print("[yellow]No variants could be generated.[/]")
        return

    created_count = 0
    for variant in variants:
        child = create_child_experiment(exp_id, parent.theme, variant)
        if child:
            created_count += 1

    console.print(
        Panel(
            f"[bold]Parent:[/]    #{exp_id}\n"
            f"[bold]Expression:[/] {_truncate(parent.expression, 60)}\n"
            f"[bold]Generated:[/] {created_count} variants\n"
            f"[bold]Unique:[/]    {len(variants)} mutations produced",
            title="🧬 Variant Generation Complete",
            border_style="green",
        )
    )

@app.command()
def mine(
    exp_id: Optional[int] = typer.Argument(None, help="Parent experiment ID (optional, will use top if omitted)"),
    generations: int = typer.Option(5, "--generations", "-g", help="Number of generations to evolve"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of top parents to keep per generation"),
    children: int = typer.Option(10, "--children", "-c", help="Number of children to generate per parent"),
):
    """End-to-End Autonomous Alpha Mining Loop."""
    from services.mine_engine import MineEngine
    engine = MineEngine()
    engine.run(parent_id=exp_id, generations=generations, top_k=top_k, children_per_alpha=children)


# ── PHASE 6: THEME ANALYTICS ─────────────────────────────────────────────────


@app.command()
def stats():
    """Display aggregate statistics per theme."""
    theme_stats = get_theme_stats()

    if not theme_stats:
        console.print("[yellow]No experiments found.[/]")
        return

    table = Table(
        title="📊 Theme Statistics",
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Theme", style="bold magenta", width=20)
    table.add_column("Experiments", justify="right", width=12)
    table.add_column("Avg Sharpe", justify="right", width=12)
    table.add_column("Best Sharpe", justify="right", style="green", width=12)
    table.add_column("Avg Fitness", justify="right", width=12)
    table.add_column("Best Fitness", justify="right", style="green", width=12)

    for s in theme_stats:
        table.add_row(
            s["theme"],
            str(s["count"]),
            _fmt_float(s["avg_sharpe"]),
            _fmt_float(s["best_sharpe"]),
            _fmt_float(s["avg_fitness"]),
            _fmt_float(s["best_fitness"]),
        )

    console.print()
    console.print(table)
    console.print()


@app.command("best-themes")
def best_themes(
    metric: str = typer.Option("sharpe", "--metric", "-m", help="Rank by: sharpe or fitness"),
):
    """Rank themes by average Sharpe or Fitness."""
    results = get_best_themes(metric=metric)

    if not results:
        console.print("[yellow]No scored experiments yet.[/]")
        return

    table = Table(
        title=f"🥇 Best Themes by Avg {metric.capitalize()}",
        header_style="bold yellow",
        border_style="dim",
    )
    table.add_column("Rank", justify="right", width=5)
    table.add_column("Theme", style="bold magenta", width=20)
    table.add_column("Scored", justify="right", width=8)
    table.add_column(f"Avg {metric.capitalize()}", justify="right", style="green", width=12)
    table.add_column(f"Best {metric.capitalize()}", justify="right", width=12)

    for rank, row in enumerate(results, 1):
        table.add_row(
            str(rank),
            row["theme"],
            str(row["count"]),
            _fmt_float(row["avg_metric"]),
            _fmt_float(row["best_metric"]),
        )

    console.print()
    console.print(table)
    console.print()


# ── PHASE 7: INSIGHTS ────────────────────────────────────────────────────────


@app.command("field-stats")
def field_stats_cmd():
    """Display performance statistics for each data field."""
    from services.insights_service import get_field_stats

    all_experiments = get_all_experiments()
    scored = [e for e in all_experiments if e.sharpe is not None]

    if not scored:
        console.print("[yellow]No scored experiments to analyze.[/]")
        return

    stats = get_field_stats(scored)

    if not stats:
        console.print("[yellow]No field data found in scored experiments.[/]")
        return

    table = Table(
        title="📊 Field Statistics",
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Field", style="bold blue", width=25)
    table.add_column("Used In", justify="right", width=12)
    table.add_column("Avg Sharpe", justify="right", style="green", width=12)
    table.add_column("Avg Fitness", justify="right", width=12)

    for s in stats[:50]:  # Show top 50
        table.add_row(
            s["field"],
            str(s["count"]),
            _fmt_float(s["avg_sharpe"]),
            _fmt_float(s["avg_fitness"]),
        )

    console.print()
    console.print(table)
    console.print()


@app.command("operator-stats")
def operator_stats_cmd():
    """Display performance statistics for each operator."""
    from services.insights_service import get_operator_stats

    all_experiments = get_all_experiments()
    scored = [e for e in all_experiments if e.sharpe is not None]

    if not scored:
        console.print("[yellow]No scored experiments to analyze.[/]")
        return

    stats = get_operator_stats(scored)

    if not stats:
        console.print("[yellow]No operator data found in scored experiments.[/]")
        return

    table = Table(
        title="⚙️ Operator Statistics",
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Operator", style="bold magenta", width=25)
    table.add_column("Used In", justify="right", width=12)
    table.add_column("Avg Sharpe", justify="right", style="green", width=12)
    table.add_column("Avg Fitness", justify="right", width=12)

    for s in stats[:50]:  # Show top 50
        table.add_row(
            s["operator"],
            str(s["count"]),
            _fmt_float(s["avg_sharpe"]),
            _fmt_float(s["avg_fitness"]),
        )

    console.print()
    console.print(table)
    console.print()


@app.command()
def insights():
    """Analyze patterns across all experiments (statistical, no LLM)."""
    from services.insights_service import generate_insights

    all_experiments = get_all_experiments()
    scored = [e for e in all_experiments if e.sharpe is not None]

    if not scored:
        console.print("[yellow]No scored experiments to analyze.[/]")
        return

    result = generate_insights(scored)

    # Top themes
    console.print()
    console.print(
        Panel(
            "\n".join(
                f"  [green]•[/] [bold]{t['theme']}[/] — Avg Sharpe: {_fmt_float(t['avg_sharpe'])}, "
                f"Count: {t['count']}"
                for t in result.get("top_themes", [])
            ) or "[dim]No data[/]",
            title="🏆 Top Performing Themes",
            border_style="green",
        )
    )

    # Worst themes
    console.print(
        Panel(
            "\n".join(
                f"  [red]•[/] [bold]{t['theme']}[/] — Avg Sharpe: {_fmt_float(t['avg_sharpe'])}, "
                f"Count: {t['count']}"
                for t in result.get("worst_themes", [])
            ) or "[dim]No data[/]",
            title="⚠️  Worst Performing Themes",
            border_style="red",
        )
    )

    # Winner operators
    console.print(
        Panel(
            "\n".join(
                f"  [green]•[/] [bold]{op}[/] — used {freq}x"
                for op, freq in result.get("winner_operators", [])
            ) or "[dim]No data[/]",
            title="🎯 Most Common Operators in Winners",
            border_style="cyan",
        )
    )

    # Loser operators
    console.print(
        Panel(
            "\n".join(
                f"  [red]•[/] [bold]{op}[/] — used {freq}x"
                for op, freq in result.get("loser_operators", [])
            ) or "[dim]No data[/]",
            title="❌ Most Common Operators in Losers",
            border_style="yellow",
        )
    )

    # Observations
    if result.get("observations"):
        console.print(
            Panel(
                "\n".join(f"  → {obs}" for obs in result["observations"]),
                title="💡 Observations",
                border_style="bright_blue",
            )
        )

    console.print()


# ── PHASE 8: EMBEDDING SEARCH ────────────────────────────────────────────────


@app.command("embed-all")
def embed_all():
    """Compute and store embeddings for all experiments."""
    from services.embedding_service import EmbeddingService

    service = EmbeddingService()
    all_experiments = get_all_experiments()

    if not all_experiments:
        console.print("[yellow]No experiments to embed.[/]")
        return

    count = 0
    with console.status("[bold cyan]Computing embeddings..."):
        for exp in all_experiments:
            text = f"{exp.theme} {exp.expression} {exp.notes or ''}"
            embedding = service.embed_text(text)
            service.store_embedding(exp.id, embedding)
            count += 1

    console.print(f"[green]✅ Embedded {count} experiments.[/]")


@app.command()
def similar(
    exp_id: int = typer.Argument(..., help="Experiment ID to find similar"),
    top_k: int = typer.Option(10, "--top", "-k", help="Number of similar results"),
):
    """Find experiments similar to the given one."""
    from services.embedding_service import EmbeddingService

    exp = get_experiment(exp_id)
    if not exp:
        console.print(f"[red]Experiment #{exp_id} not found.[/]")
        raise typer.Exit(1)

    service = EmbeddingService()
    results = service.find_similar(exp_id, top_k=top_k)

    if not results:
        console.print("[yellow]No similar experiments found. Run 'embed-all' first.[/]")
        return

    table = Table(
        title=f"🔗 Experiments Similar to #{exp_id}",
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("ID", justify="right", width=5)
    table.add_column("Similarity", justify="right", width=10)
    table.add_column("Theme", style="magenta", width=15)
    table.add_column("Expression", width=45)
    table.add_column("Sharpe", justify="right", width=8)

    for sim_id, score_val in results:
        sim_exp = get_experiment(sim_id)
        if sim_exp:
            table.add_row(
                str(sim_exp.id),
                f"{score_val:.4f}",
                sim_exp.theme,
                _truncate(sim_exp.expression, 43),
                _fmt_float(sim_exp.sharpe),
            )

    console.print()
    console.print(table)
    console.print()


# ── PHASE 9: RAG RESEARCH AGENT ──────────────────────────────────────────────


@app.command()
def ask(question: str = typer.Argument(..., help="Research question")):
    """Ask a research question using the RAG pipeline."""
    from services.embedding_service import EmbeddingService
    from services.research_agent import ResearchAgent

    service = EmbeddingService()
    agent = ResearchAgent(embedding_service=service)

    console.print(
        Panel(
            f"[bold]{question}[/]",
            title="🤔 Research Question",
            border_style="cyan",
        )
    )

    with console.status("[bold cyan]Researching..."):
        answer = agent.ask(question)

    console.print(
        Panel(
            answer,
            title="📝 Research Summary",
            border_style="green",
        )
    )
    console.print()


# ── LLM CONNECTIVITY TEST ────────────────────────────────────────────────────


@app.command("llm-test")
def llm_test():
    """Test connectivity to the configured LLM provider."""
    from services.research_agent import ResearchAgent

    # Create a minimal agent just for config/connectivity testing
    class _DummyEmbedding:
        pass

    agent = ResearchAgent(embedding_service=_DummyEmbedding())

    # Step 1: Show current config
    console.print()
    console.print(
        Panel(
            f"[bold]Provider:[/]  {agent.provider}\n"
            f"[bold]Model:[/]    {agent.model}\n"
            f"[bold]Base URL:[/] "
            f"{agent.ollama_base_url if agent.provider == 'ollama' else 'https://api.openai.com/v1'}\n"
            f"[bold]API Key:[/]  {'***' + agent.api_key[-4:] if len(agent.api_key) > 4 else ('(set)' if agent.api_key else '[red](not set)[/]')}",
            title="LLM Configuration",
            border_style="cyan",
        )
    )

    # Step 2: Validate config
    valid, msg = agent.validate_config()
    if not valid:
        console.print(
            Panel(
                f"[red]{msg}[/]",
                title="[red]Configuration Error[/]",
                border_style="red",
            )
        )
        raise typer.Exit(1)

    console.print("[dim]Configuration valid. Testing connectivity...[/]")

    # Step 3: Test connectivity
    with console.status("[bold cyan]Connecting..."):
        success, result_msg = agent.test_connectivity()

    if success:
        console.print(
            Panel(
                f"[green]{result_msg}[/]",
                title="[green]Connection Successful[/]",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel(
                f"[red]{result_msg}[/]",
                title="[red]Connection Failed[/]",
                border_style="red",
            )
        )
        raise typer.Exit(1)

    console.print()