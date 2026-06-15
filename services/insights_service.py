"""Statistical insights engine for alpha experiments.

Analyzes patterns across scored experiments without using LLMs.
Produces actionable observations about themes, operators, and performance.
"""

import re
from collections import Counter
from typing import Any


def generate_insights(scored_experiments: list) -> dict[str, Any]:
    """Analyze scored experiments and return structured insights.

    Args:
        scored_experiments: List of Experiment objects with non-null sharpe.

    Returns:
        Dict with keys:
            - top_themes: Best performing themes by avg sharpe
            - worst_themes: Worst performing themes
            - winner_operators: Most common operators in top-quartile alphas
            - loser_operators: Most common operators in bottom-quartile alphas
            - observations: List of natural-language observations
    """
    if not scored_experiments:
        return {
            "top_themes": [],
            "worst_themes": [],
            "winner_operators": [],
            "loser_operators": [],
            "observations": [],
        }

    # ── Theme analysis ────────────────────────────────────────────────────

    theme_sharpes: dict[str, list[float]] = {}
    for exp in scored_experiments:
        theme = exp.theme or "Unknown"
        if theme not in theme_sharpes:
            theme_sharpes[theme] = []
        theme_sharpes[theme].append(exp.sharpe)

    theme_stats = []
    for theme, sharpes in theme_sharpes.items():
        avg = sum(sharpes) / len(sharpes) if sharpes else 0
        theme_stats.append({
            "theme": theme,
            "avg_sharpe": round(avg, 4),
            "count": len(sharpes),
            "best": round(max(sharpes), 4),
            "worst": round(min(sharpes), 4),
        })

    theme_stats.sort(key=lambda x: x["avg_sharpe"], reverse=True)
    top_themes = theme_stats[:5]
    worst_themes = list(reversed(theme_stats[-5:])) if len(theme_stats) > 1 else []

    # ── Operator analysis ─────────────────────────────────────────────────

    # Sort experiments by sharpe
    sorted_exps = sorted(scored_experiments, key=lambda e: e.sharpe, reverse=True)

    # Quartile split
    n = len(sorted_exps)
    q1_cutoff = max(n // 4, 1)

    winners = sorted_exps[:q1_cutoff]
    losers = sorted_exps[-q1_cutoff:] if n > 1 else []

    winner_ops = _count_operators([e.expression for e in winners])
    loser_ops = _count_operators([e.expression for e in losers])

    winner_operators = winner_ops.most_common(10)
    loser_operators = loser_ops.most_common(10)

    # ── Observations ──────────────────────────────────────────────────────

    observations = _generate_observations(
        scored_experiments,
        top_themes,
        worst_themes,
        winner_operators,
        loser_operators,
    )

    return {
        "top_themes": top_themes,
        "worst_themes": worst_themes,
        "winner_operators": winner_operators,
        "loser_operators": loser_operators,
        "observations": observations,
    }


def _count_operators(expressions: list[str]) -> Counter:
    """Extract and count operator function calls from expressions."""
    counter: Counter = Counter()

    # Match function-call-style operators: name(...)
    pattern = r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\("

    for expr in expressions:
        operators = re.findall(pattern, expr)
        for op in operators:
            # Skip single-letter variables and common math
            if len(op) > 1 and op not in ("max", "min", "if"):
                counter[op] += 1

    return counter


def _generate_observations(
    experiments: list,
    top_themes: list[dict],
    worst_themes: list[dict],
    winner_ops: list[tuple],
    loser_ops: list[tuple],
) -> list[str]:
    """Generate natural-language observations from the analysis."""
    observations = []

    # Overall stats
    sharpes = [e.sharpe for e in experiments]
    avg_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0
    observations.append(
        f"Average Sharpe across {len(experiments)} scored experiments: "
        f"{avg_sharpe:.4f}"
    )

    # Top theme insight
    if top_themes:
        best = top_themes[0]
        observations.append(
            f"Best theme: '{best['theme']}' with avg Sharpe {best['avg_sharpe']:.4f} "
            f"across {best['count']} experiments"
        )

    # Worst theme insight
    if worst_themes and worst_themes[0]["avg_sharpe"] < avg_sharpe:
        worst = worst_themes[0]
        observations.append(
            f"Underperforming theme: '{worst['theme']}' with avg Sharpe "
            f"{worst['avg_sharpe']:.4f} ({worst['count']} experiments)"
        )

    # Winner operator patterns
    if winner_ops:
        top_winner_ops = [op for op, _ in winner_ops[:3]]
        observations.append(
            f"Winning alphas favor: {', '.join(top_winner_ops)}"
        )

    # Loser operator patterns
    if loser_ops:
        top_loser_ops = [op for op, _ in loser_ops[:3]]
        observations.append(
            f"Losing alphas commonly use: {', '.join(top_loser_ops)}"
        )

    # Operator overlap
    if winner_ops and loser_ops:
        winner_set = {op for op, _ in winner_ops[:5]}
        loser_set = {op for op, _ in loser_ops[:5]}
        unique_to_winners = winner_set - loser_set
        if unique_to_winners:
            observations.append(
                f"Operators unique to winners: {', '.join(unique_to_winners)}"
            )
        unique_to_losers = loser_set - winner_set
        if unique_to_losers:
            observations.append(
                f"Operators unique to losers: {', '.join(unique_to_losers)}"
            )

    # Generation depth analysis
    generations = [e.generation for e in experiments]
    if generations and max(generations) > 0:
        gen_sharpes: dict[int, list[float]] = {}
        for e in experiments:
            gen = e.generation
            if gen not in gen_sharpes:
                gen_sharpes[gen] = []
            gen_sharpes[gen].append(e.sharpe)

        for gen in sorted(gen_sharpes.keys()):
            avg = sum(gen_sharpes[gen]) / len(gen_sharpes[gen])
            if gen == 0:
                observations.append(
                    f"Root alphas (gen 0): avg Sharpe {avg:.4f} "
                    f"({len(gen_sharpes[gen])} experiments)"
                )
            else:
                observations.append(
                    f"Generation {gen} variants: avg Sharpe {avg:.4f} "
                    f"({len(gen_sharpes[gen])} experiments)"
                )

    return observations


def get_operator_stats(scored_experiments: list) -> list[dict]:
    """Calculate average performance metrics per operator."""
    from collections import defaultdict

    op_stats = defaultdict(lambda: {"count": 0, "sharpes": [], "fitnesses": []})

    for exp in scored_experiments:
        if not exp.sharpe:
            continue
            
        pattern = r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
        ops_in_exp = set(re.findall(pattern, exp.expression))
        
        for op in ops_in_exp:
            if len(op) > 1 and op not in ("max", "min", "if"):
                op_stats[op]["count"] += 1
                op_stats[op]["sharpes"].append(exp.sharpe)
                op_stats[op]["fitnesses"].append(exp.fitness or 0.0)

    results = []
    for op, stats in op_stats.items():
        count = stats["count"]
        if count > 0:
            avg_sharpe = sum(stats["sharpes"]) / count
            avg_fit = sum(stats["fitnesses"]) / count
            results.append({
                "operator": op,
                "count": count,
                "avg_sharpe": avg_sharpe,
                "avg_fitness": avg_fit
            })

    results.sort(key=lambda x: x["count"], reverse=True)
    return results


def get_field_stats(scored_experiments: list) -> list[dict]:
    """Calculate average performance metrics per data field."""
    from collections import defaultdict
    from services.field_service import get_all_fields

    fields = {f.name.lower() for f in get_all_fields()}
    field_stats = defaultdict(lambda: {"count": 0, "sharpes": [], "fitnesses": []})

    for exp in scored_experiments:
        if not exp.sharpe:
            continue
            
        # Find words not followed by '(' to roughly identify fields
        tokens = set(re.findall(r"\b([a-zA-Z_]\w*)\b(?!\s*\()", exp.expression))
        
        for token in tokens:
            t_lower = token.lower()
            if t_lower in fields:
                field_stats[t_lower]["count"] += 1
                field_stats[t_lower]["sharpes"].append(exp.sharpe)
                field_stats[t_lower]["fitnesses"].append(exp.fitness or 0.0)

    results = []
    for field, stats in field_stats.items():
        count = stats["count"]
        if count > 0:
            avg_sharpe = sum(stats["sharpes"]) / count
            avg_fit = sum(stats["fitnesses"]) / count
            results.append({
                "field": field,
                "count": count,
                "avg_sharpe": avg_sharpe,
                "avg_fitness": avg_fit
            })

    results.sort(key=lambda x: x["count"], reverse=True)
    return results
