"""Service for detecting structural similarities and preventing duplicate simulations."""

import hashlib
import re

from database.database import get_db
from database.models import Experiment

def hash_expression_structure(expression: str) -> str:
    """
    Hash the structure of an expression ignoring specific field names and constants.
    """
    # Replace all specific numbers with N
    expr = re.sub(r"\b\d+(\.\d+)?\b", "N", expression)
    
    # Replace all probable field names with F
    tokens = set(re.findall(r"[a-zA-Z_]\w*", expr))
    
    allowed = {"ts_decay_linear", "ts_mean", "rank", "zscore", "ts_scale", "delay", "group", "if", "then", "else", "and", "or"}
    
    for token in tokens:
        if token.lower() not in allowed and not token.startswith("ts_") and token != "N" and token != "F":
            expr = re.sub(r"\b" + re.escape(token) + r"\b", "F", expr)
            
    # Return MD5 of structure
    return hashlib.md5(expr.encode()).hexdigest()

def is_duplicate_structure(expression: str) -> bool:
    """
    Check if the structure of this expression already exists in the database.
    """
    # 1. Compute the structural hash
    expr_hash = hash_expression_structure(expression)
    
    with get_db() as db:
        # 2. Check for exact string match first (fastest)
        existing_exact = db.query(Experiment.id).filter(Experiment.expression == expression).first()
        if existing_exact:
            return True
            
        # 3. Check for structural duplicate using the hash
        existing_struct = db.query(Experiment.id).filter(Experiment.structure_hash == expr_hash).first()
        if existing_struct:
            return True
            
        return False
