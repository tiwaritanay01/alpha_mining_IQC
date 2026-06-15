"""Service for scoring alphas based on multi-objective criteria."""

def score_alpha(sharpe: float, fitness: float, turnover: float) -> float:
    """
    Calculate a composite score for an alpha.
    
    Weights:
    - 60% Sharpe
    - 30% Fitness
    - 10% penalty for Turnover
    """
    if sharpe is None:
        return 0.0
    
    f = fitness if fitness is not None else 0.0
    t = turnover if turnover is not None else 0.0
    
    return (0.6 * sharpe) + (0.3 * f) - (0.1 * abs(t))
