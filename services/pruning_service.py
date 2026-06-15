"""Service for archiving and pruning underperforming experiments."""

from database.database import get_db
from database.models import Experiment

def prune_experiments(keep_top: int = 50) -> int:
    """
    Archive underperforming or un-scored experiments, keeping the top K.
    
    Returns:
        Number of experiments archived.
    """
    archived_count = 0
    with get_db() as db:
        # Get the IDs of the top 'keep_top' experiments by score
        top_exps = (
            db.query(Experiment.id)
            .filter(Experiment.score.isnot(None))
            .filter(Experiment.is_archived == 0)
            .order_by(Experiment.score.desc())
            .limit(keep_top)
            .all()
        )
        top_ids = {r[0] for r in top_exps}

        # Find all unarchived experiments that are NOT in the top list
        to_prune = (
            db.query(Experiment)
            .filter(Experiment.is_archived == 0)
            .filter(Experiment.id.notin_(top_ids))
            .all()
        )

        for exp in to_prune:
            exp.is_archived = 1
            archived_count += 1
            
        db.commit()

    return archived_count
