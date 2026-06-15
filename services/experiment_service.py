"""Service layer for Experiment CRUD, lineage, search, and analytics."""

from typing import Optional

from sqlalchemy import func, or_

from database.database import get_db
from database.models import Experiment


from services.scoring_service import score_alpha
from services.strategy_service import classify_expression

# ── CRUD ──────────────────────────────────────────────────────────────────────


def create_experiment(
    theme: str,
    expression: str,
    notes: str = "",
) -> Experiment:
    """Create a new root experiment."""
    with get_db() as db:
        exp = Experiment(
            theme=theme,
            expression=expression,
            notes=notes,
            classification=classify_expression(expression),
        )
        db.add(exp)
        db.flush()
        db.refresh(exp)
        return exp


def get_experiment(exp_id: int) -> Optional[Experiment]:
    """Get a single experiment by ID."""
    with get_db() as db:
        return db.query(Experiment).filter(Experiment.id == exp_id).first()


def get_all_experiments() -> list[Experiment]:
    """Get all experiments ordered by ID descending, ignoring archived."""
    with get_db() as db:
        return (
            db.query(Experiment)
            .filter(Experiment.is_archived == 0)
            .order_by(Experiment.id.desc())
            .all()
        )


def update_metrics(
    exp_id: int,
    sharpe: float,
    fitness: float,
    turnover: float,
    returns: float,
) -> Optional[Experiment]:
    """Update performance metrics for an experiment."""
    with get_db() as db:
        exp = db.query(Experiment).filter(Experiment.id == exp_id).first()
        if not exp:
            return None

        exp.sharpe = sharpe
        exp.fitness = fitness
        exp.turnover = turnover
        exp.returns = returns
        exp.status = "tested"
        exp.score = score_alpha(sharpe, fitness, turnover)
        if not exp.classification:
            exp.classification = classify_expression(exp.expression)
            
        db.flush()
        db.refresh(exp)
        return exp


def get_top_score(limit: int = 10) -> list[Experiment]:
    """Get top experiments ranked by composite score."""
    with get_db() as db:
        return (
            db.query(Experiment)
            .filter(Experiment.score.isnot(None))
            .filter(Experiment.is_archived == 0)
            .order_by(Experiment.score.desc())
            .limit(limit)
            .all()
        )


def get_top_sharpe(limit: int = 10) -> list[Experiment]:
    """Get top experiments ranked by Sharpe ratio."""
    with get_db() as db:
        return (
            db.query(Experiment)
            .filter(Experiment.sharpe.isnot(None))
            .filter(Experiment.is_archived == 0)
            .order_by(Experiment.sharpe.desc())
            .limit(limit)
            .all()
        )


def get_top_fitness(limit: int = 10) -> list[Experiment]:
    """Get top experiments ranked by Fitness."""
    with get_db() as db:
        return (
            db.query(Experiment)
            .filter(Experiment.fitness.isnot(None))
            .filter(Experiment.is_archived == 0)
            .order_by(Experiment.fitness.desc())
            .limit(limit)
            .all()
        )


def get_top_returns(limit: int = 10) -> list[Experiment]:
    """Get top experiments ranked by Returns."""
    with get_db() as db:
        return (
            db.query(Experiment)
            .filter(Experiment.returns.isnot(None))
            .filter(Experiment.is_archived == 0)
            .order_by(Experiment.returns.desc())
            .limit(limit)
            .all()
        )


def get_recent_winners(limit: int = 10, min_sharpe: float = 1.0) -> list[Experiment]:
    """Get recently created experiments with a good Sharpe ratio."""
    with get_db() as db:
        return (
            db.query(Experiment)
            .filter(Experiment.sharpe.isnot(None))
            .filter(Experiment.sharpe >= min_sharpe)
            .filter(Experiment.is_archived == 0)
            .order_by(Experiment.id.desc())
            .limit(limit)
            .all()
        )


def get_best_generated(limit: int = 10) -> list[Experiment]:
    """Get top experiments that are children (generated variants)."""
    with get_db() as db:
        return (
            db.query(Experiment)
            .filter(Experiment.parent_id.isnot(None))
            .filter(Experiment.sharpe.isnot(None))
            .filter(Experiment.is_archived == 0)
            .order_by(Experiment.sharpe.desc())
            .limit(limit)
            .all()
        )


def create_child_experiment(
    parent_id: int,
    theme: str,
    expression: str,
) -> Optional[Experiment]:
    """Create a child experiment linked to a parent."""
    with get_db() as db:
        parent = (
            db.query(Experiment)
            .filter(Experiment.id == parent_id)
            .first()
        )
        if not parent:
            return None

        child = Experiment(
            theme=theme,
            expression=expression,
            parent_id=parent_id,
            generation=parent.generation + 1,
            classification=classify_expression(expression),
        )
        db.add(child)
        db.flush()
        db.refresh(child)
        return child


# ── LINEAGE ───────────────────────────────────────────────────────────────────


def get_children(parent_id: int) -> list[Experiment]:
    """Get direct children of an experiment."""
    with get_db() as db:
        return (
            db.query(Experiment)
            .filter(Experiment.parent_id == parent_id)
            .order_by(Experiment.id)
            .all()
        )


def get_tree(exp_id: int) -> Optional[dict]:
    """
    Recursively build an experiment family tree.

    Returns a dict:
        {
            "experiment": Experiment,
            "children": [{ "experiment": ..., "children": [...] }, ...]
        }
    """
    exp = get_experiment(exp_id)
    if not exp:
        return None

    children = get_children(exp_id)
    child_trees = [get_tree(child.id) for child in children]

    return {
        "experiment": exp,
        "children": [t for t in child_trees if t is not None],
    }


# ── SEARCH ────────────────────────────────────────────────────────────────────


def search_experiments(query: str) -> list[Experiment]:
    """Search experiments by theme, expression, and notes (case-insensitive)."""
    with get_db() as db:
        pattern = f"%{query}%"
        return (
            db.query(Experiment)
            .filter(
                or_(
                    Experiment.theme.ilike(pattern),
                    Experiment.expression.ilike(pattern),
                    Experiment.notes.ilike(pattern),
                )
            )
            .order_by(Experiment.id.desc())
            .all()
        )


# ── ANALYTICS ─────────────────────────────────────────────────────────────────


def get_theme_stats() -> list[dict]:
    """
    Aggregate stats per theme.

    Returns list of dicts with:
        theme, count, avg_sharpe, avg_fitness, best_sharpe, best_fitness
    """
    with get_db() as db:
        results = (
            db.query(
                Experiment.theme,
                func.count(Experiment.id).label("count"),
                func.avg(Experiment.sharpe).label("avg_sharpe"),
                func.avg(Experiment.fitness).label("avg_fitness"),
                func.max(Experiment.sharpe).label("best_sharpe"),
                func.max(Experiment.fitness).label("best_fitness"),
            )
            .group_by(Experiment.theme)
            .order_by(func.count(Experiment.id).desc())
            .all()
        )

        return [
            {
                "theme": row.theme,
                "count": row.count,
                "avg_sharpe": round(row.avg_sharpe, 4) if row.avg_sharpe else None,
                "avg_fitness": round(row.avg_fitness, 4) if row.avg_fitness else None,
                "best_sharpe": round(row.best_sharpe, 4) if row.best_sharpe else None,
                "best_fitness": round(row.best_fitness, 4) if row.best_fitness else None,
            }
            for row in results
        ]


def get_best_themes(metric: str = "sharpe", limit: int = 10) -> list[dict]:
    """Rank themes by average metric (sharpe or fitness)."""
    metric_col = Experiment.sharpe if metric == "sharpe" else Experiment.fitness

    with get_db() as db:
        results = (
            db.query(
                Experiment.theme,
                func.count(Experiment.id).label("count"),
                func.avg(metric_col).label("avg_metric"),
                func.max(metric_col).label("best_metric"),
            )
            .filter(metric_col.isnot(None))
            .group_by(Experiment.theme)
            .order_by(func.avg(metric_col).desc())
            .limit(limit)
            .all()
        )

        return [
            {
                "theme": row.theme,
                "count": row.count,
                "avg_metric": round(row.avg_metric, 4) if row.avg_metric else None,
                "best_metric": round(row.best_metric, 4) if row.best_metric else None,
            }
            for row in results
        ]


def import_from_api_response(exp_id: int, raw_response: dict) -> Optional[Experiment]:
    """
    Parse a WorldQuant API simulation response and update experiment metrics.

    Expected keys in raw_response:
        sharpe, fitness, turnover, returns (or nested under 'is' / 'result')
    """
    # Handle both flat and nested WQ response formats
    data = raw_response

    # WQ Brain nests results under 'is' key in some responses
    if "is" in data:
        data = data["is"]

    sharpe = _extract_float(data, ["sharpe", "sharpeRatio", "Sharpe"])
    fitness = _extract_float(data, ["fitness", "Fitness"])
    turnover = _extract_float(data, ["turnover", "Turnover"])
    returns_ = _extract_float(data, ["returns", "Returns", "ret"])

    if sharpe is None:
        return None

    return update_metrics(
        exp_id,
        sharpe=sharpe,
        fitness=fitness or 0.0,
        turnover=turnover or 0.0,
        returns=returns_ or 0.0,
    )


# ── HELPERS ───────────────────────────────────────────────────────────────────


def _extract_float(data: dict, keys: list[str]) -> Optional[float]:
    """Try multiple key names to extract a float value."""
    for key in keys:
        if key in data and data[key] is not None:
            try:
                return float(data[key])
            except (ValueError, TypeError):
                continue
    return None