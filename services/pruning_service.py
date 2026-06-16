"""Service for archiving and pruning underperforming experiments."""

from database.database import get_db
from database.models import Experiment


def prune_experiments(keep_top: int = 50) -> int:
    """
    Archive underperforming experiments, keeping the top K scored ones.

    Safety rules:
    - NEVER archive root experiments (parent_id IS NULL).
    - NEVER archive un-scored experiments (score IS NULL) — they haven't been tested yet.
    - Only archive scored children that fall outside the top-K.

    Returns:
        Number of experiments archived.
    """
    archived_count = 0
    with get_db() as db:
        # Get the IDs of the top 'keep_top' scored experiments
        top_exps = (
            db.query(Experiment.id)
            .filter(Experiment.score.isnot(None))
            .filter(Experiment.is_archived == 0)
            .order_by(Experiment.score.desc())
            .limit(keep_top)
            .all()
        )
        top_ids = {r[0] for r in top_exps}

        # Only prune experiments that:
        # 1. Have been scored (score IS NOT NULL)
        # 2. Are NOT root experiments (parent_id IS NOT NULL)
        # 3. Are NOT in the top-K
        # 4. Are not already archived
        to_prune = (
            db.query(Experiment)
            .filter(Experiment.is_archived == 0)
            .filter(Experiment.score.isnot(None))        # Only prune scored
            .filter(Experiment.parent_id.isnot(None))     # Never prune roots
            .filter(Experiment.id.notin_(top_ids))
            .all()
        )

        for exp in to_prune:
            exp.is_archived = 1
            archived_count += 1

    return archived_count

