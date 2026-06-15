"""Service for classifying expressions into strategy regimes."""

def classify_expression(expression: str) -> str:
    """
    Classify an expression into a strategy regime based on keywords.
    """
    expr_lower = expression.lower()
    
    # Value strategies typically look at fundamental ratios and earnings
    value_keywords = ["operating_income", "enterprise_value", "revenue", "ebitda", "assets", "liabilities", "cashflow"]
    
    # Momentum strategies typically look at returns and moving averages
    momentum_keywords = ["returns", "close", "vwap", "ts_mean", "ts_decay"]
    
    # Volatility strategies look at standard deviations and variance
    volatility_keywords = ["ts_std_dev", "ts_variance", "volatility"]
    
    # Mean reversion looks at differences from mean or negative returns
    mean_reversion_keywords = ["-returns", "ts_zscore", "rank(-"]
    
    scores = {
        "value": sum(1 for kw in value_keywords if kw in expr_lower),
        "momentum": sum(1 for kw in momentum_keywords if kw in expr_lower),
        "volatility": sum(1 for kw in volatility_keywords if kw in expr_lower),
        "mean reversion": sum(1 for kw in mean_reversion_keywords if kw in expr_lower),
    }
    
    best_strategy = max(scores.items(), key=lambda x: x[1])
    
    if best_strategy[1] > 0:
        return best_strategy[0]
    
    return "unknown"
