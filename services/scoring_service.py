"""Service for scoring alphas based on multi-objective criteria."""


def score_alpha(sharpe: float, fitness: float, turnover: float) -> float:
    """
    Calculate a composite score for an alpha.

    Weights:
    - 60% Sharpe
    - 30% Fitness
    - 10% turnover penalty (only above threshold, capped)

    Turnover normalization:
    - Turnover < 0.5 → no penalty
    - Turnover 0.5–2.0 → linear penalty scaled to [0, 0.3]
    - Turnover > 2.0 → max penalty of 0.3
    This prevents turnover from dominating when its raw value is 10-100x
    larger than Sharpe/fitness.
    """
    if sharpe is None:
        return 0.0

    s = sharpe
    f = fitness if fitness is not None else 0.0
    t = turnover if turnover is not None else 0.0

    # Normalize turnover penalty to a 0–0.3 range
    if t <= 0.5:
        turnover_penalty = 0.0
    elif t <= 2.0:
        turnover_penalty = ((t - 0.5) / 1.5) * 0.3  # linear scale to max 0.3
    else:
        turnover_penalty = 0.3  # hard cap

    return (0.6 * s) + (0.3 * f) - turnover_penalty

