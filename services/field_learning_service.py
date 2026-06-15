"""Service for tracking field usage and performance."""

import re
from typing import Counter
from collections import Counter as PythonCounter

from database.database import get_db
from database.models import Experiment

def get_best_fields(limit: int = 20) -> list[str]:
    """
    Get the most frequently used fields in high-scoring alphas.
    """
    field_counts: Counter[str] = PythonCounter()
    
    with get_db() as db:
        # Get top scoring experiments
        top_exps = (
            db.query(Experiment.expression)
            .filter(Experiment.score.isnot(None))
            .filter(Experiment.score > 0)
            .order_by(Experiment.score.desc())
            .limit(100)
            .all()
        )
        
        for exp in top_exps:
            # Extract fields
            tokens = set(re.findall(r"[a-zA-Z_]\w*", exp.expression))
            # Rough filter for typical field names
            for token in tokens:
                if token.islower() and not token.startswith("ts_") and token not in ("rank", "delay", "decay_linear", "group", "if", "then", "else", "true", "false", "and", "or", "not"):
                    field_counts[token] += 1
                    
    if not field_counts:
        return []
        
    return [f[0] for f in field_counts.most_common(limit)]
