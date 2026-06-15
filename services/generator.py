"""Modular alpha expression mutation engine.

Generates diverse variants of alpha expressions through 7 mutation types:
1. Window Mutation — change numeric window parameters
2. Operator Mutation — swap compatible operators
3. Neutralization Mutation — swap neutralization methods
4. Decay Mutation — vary decay parameters
5. Weight Mutation — add/modify term weights
6. Field Substitution — swap valid datasets/fields
7. Wrapper Mutation — wrap expressions with cross-sectional/time-series operators
"""

import random
import re
from typing import Optional


# ── FALLBACK SETTINGS ─────────────────────────────────────────────────────────

DEFAULT_SWAP_GROUPS: list[list[str]] = [
    ["ts_mean", "ts_sum", "ts_std_dev", "ts_median"],
    ["ts_rank", "rank", "zscore"],
    ["ts_min", "ts_max", "ts_argmin", "ts_argmax"],
    ["ts_decay_linear", "ts_decay_exp_window"],
    ["ts_delta", "ts_backfill"],
    ["abs", "log", "sign"],
    ["correlation", "ts_covariance"],
]

DEFAULT_FIELDS = ["close", "volume", "returns", "vwap", "open", "high", "low", "adv20"]

WRAPPER_OPERATORS = ["rank", "zscore", "normalize", "scale"]
TS_WRAPPER_OPERATORS = ["ts_rank", "ts_zscore", "ts_scale"]

NEUTRALIZATIONS = ["SUBINDUSTRY", "INDUSTRY", "SECTOR", "MARKET"]
DECAY_VALUES = [0, 2, 4, 6, 8, 10, 15, 20]
WINDOW_VALUES = [3, 5, 10, 15, 20, 30, 40, 60]
WEIGHT_PAIRS = [
    (0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2),
    (0.3, 0.7), (0.4, 0.6), (0.2, 0.8),
]


class FieldAwareMutationEngine:
    """Modular alpha expression mutation engine aware of fields and operators."""

    def __init__(
        self,
        operators: Optional[list[str]] = None,
        fields: Optional[list[str]] = None,
    ) -> None:
        self.swap_groups = self._build_swap_groups(operators)
        self.valid_operators = set(operators) if operators else set(sum(DEFAULT_SWAP_GROUPS, []))
        self.fields = fields if fields else DEFAULT_FIELDS
        self.valid_fields = set(self.fields)

        try:
            from services.field_learning_service import get_best_fields
            self.best_fields = get_best_fields(limit=30)
        except Exception:
            self.best_fields = []

    def generate(self, expression: str, count: int = 30) -> list[str]:
        """Apply all mutation types and return up to `count` unique valid variants."""
        variants: set[str] = set()

        # First, ensure the base expression is valid by healing any unknown fields
        base_expr = self._clean_invalid_fields(expression)

        # Apply each mutation type on the healed expression
        variants.update(self._window_mutation(base_expr))
        variants.update(self._operator_mutation(base_expr))
        variants.update(self._neutralization_mutation(base_expr))
        variants.update(self._decay_mutation(base_expr))
        variants.update(self._weight_mutation(base_expr))
        variants.update(self._field_mutation(base_expr))
        variants.update(self._wrapper_mutation(base_expr))

        # Filter out invalid and identical expressions
        variants.discard(expression)
        variants.discard(base_expr)
        
        valid_variants = {v.strip() for v in variants if self.is_valid(v.strip())}

        # If we didn't generate enough through pure mutation, just generate random field swaps on the healed base
        attempts = 0
        while len(valid_variants) < count and attempts < 100:
            rand_healed = self._clean_invalid_fields(base_expr, force_randomize_fields=True)
            if rand_healed != expression and self.is_valid(rand_healed):
                valid_variants.add(rand_healed)
            attempts += 1

        # Sample if we have more than requested
        result = list(valid_variants)
        if len(result) > count:
            result = random.sample(result, count)

        # If still empty, return at least the healed base if it differs from parent
        if not result and base_expr != expression and self.is_valid(base_expr):
            return [base_expr]

        return result

    def _clean_invalid_fields(self, expression: str, force_randomize_fields: bool = False) -> str:
        """Replace all invalid fields (and optionally all fields) with valid ones."""
        tokens = set(re.findall(r"[a-zA-Z_]\w*", expression))
        allowed_non_fields = self.valid_operators | set(NEUTRALIZATIONS) | {"and", "or", "not", "if", "then", "else", "true", "false"}
        
        cleaned = expression
        for token in tokens:
            if token.upper() in NEUTRALIZATIONS:
                continue
                
            is_valid_field = token.lower() in self.valid_fields or token in self.valid_fields
            is_valid_op = token.lower() in allowed_non_fields or token in allowed_non_fields
            
            # Replace if it's an invalid field, or if we force randomizing valid fields
            if not is_valid_op:
                if not is_valid_field or force_randomize_fields:
                    replacement = random.choice(self.fields) if self.fields else token
                    pattern = r"\b" + re.escape(token) + r"\b"
                    cleaned = re.sub(pattern, replacement, cleaned)
                    
        return cleaned

    # ── VALIDATION ────────────────────────────────────────────────────────

    def is_valid(self, expression: str) -> bool:
        """Validate if an expression conforms to basic constraints."""
        if not expression or not str(expression).strip():
            return False

        if len(expression) > 60000:
            return False

        if expression.count("(") != expression.count(")"):
            return False

        # Extract tokens that look like variables or operators (alphanumeric + underscore)
        # Exclude pure numbers
        tokens = set(re.findall(r"[a-zA-Z_]\w*", expression))
        
        # All tokens must be either a known operator, known field, or standard keyword/neutralization
        allowed = self.valid_operators | self.valid_fields | set(NEUTRALIZATIONS)
        # Also allow boolean operators and common keywords
        allowed.update(["and", "or", "not", "if", "then", "else", "true", "false"])

        for token in tokens:
            if token.upper() in NEUTRALIZATIONS:
                continue
            if token.lower() not in allowed and token not in allowed:
                return False

        return True

    # ── MUTATION 1: WINDOW ────────────────────────────────────────────────

    def _window_mutation(self, expression: str) -> list[str]:
        variants = []
        pattern = r"(?<=[\(,\s])(\d+)(?=[\),\s])"
        matches = list(re.finditer(pattern, expression))

        for match in matches:
            current = int(match.group())
            if current > 200:
                continue

            for window in WINDOW_VALUES:
                if window != current:
                    variant = expression[:match.start()] + str(window) + expression[match.end():]
                    variants.append(variant)
        return variants

    # ── MUTATION 2: OPERATOR ──────────────────────────────────────────────

    def _operator_mutation(self, expression: str) -> list[str]:
        variants = []
        for group in self.swap_groups:
            for op in group:
                if op in expression:
                    for replacement in group:
                        if replacement != op:
                            variant = expression.replace(op, replacement, 1)
                            if variant != expression:
                                variants.append(variant)
        return variants

    # ── MUTATION 3: NEUTRALIZATION ────────────────────────────────────────

    def _neutralization_mutation(self, expression: str) -> list[str]:
        variants = []
        for neut in NEUTRALIZATIONS:
            if neut in expression.upper():
                for replacement in NEUTRALIZATIONS:
                    if replacement != neut:
                        variant = re.sub(
                            re.escape(neut), replacement, expression, flags=re.IGNORECASE, count=1
                        )
                        if variant != expression:
                            variants.append(variant)
        return variants

    # ── MUTATION 4: DECAY ─────────────────────────────────────────────────

    def _decay_mutation(self, expression: str) -> list[str]:
        variants = []
        decay_pattern = r"(decay[_\w]*\([^)]*?,\s*)(\d+)(\s*\))"
        matches = list(re.finditer(decay_pattern, expression, re.IGNORECASE))

        if matches:
            for match in matches:
                current = int(match.group(2))
                for decay in DECAY_VALUES:
                    if decay != current:
                        variant = expression[:match.start(2)] + str(decay) + expression[match.end(2):]
                        variants.append(variant)
        else:
            ts_decay_pattern = r"(ts_decay_\w+\([^)]+,\s*)(\d+)(\s*\))"
            matches = list(re.finditer(ts_decay_pattern, expression))
            for match in matches:
                current = int(match.group(2))
                for decay in DECAY_VALUES:
                    if decay != current and decay > 0:
                        variant = expression[:match.start(2)] + str(decay) + expression[match.end(2):]
                        variants.append(variant)
        return variants

    # ── MUTATION 5: WEIGHT ────────────────────────────────────────────────

    def _weight_mutation(self, expression: str) -> list[str]:
        variants = []
        parts = self._split_additive(expression)

        if len(parts) < 2:
            return variants

        for w1, w2 in WEIGHT_PAIRS:
            cleaned_parts = [self._strip_coefficient(p.strip()) for p in parts]
            if len(cleaned_parts) == 2:
                variant = f"{w1}*({cleaned_parts[0]}) + {w2}*({cleaned_parts[1]})"
                variants.append(variant)
            elif len(cleaned_parts) >= 3:
                weights = self._distribute_weights(len(cleaned_parts), bias_idx=0)
                weighted = [f"{w:.2f}*({p})" for w, p in zip(weights, cleaned_parts)]
                variants.append(" + ".join(weighted))
        return variants

    # ── MUTATION 6: FIELD SUBSTITUTION ────────────────────────────────────

    def _field_mutation(self, expression: str) -> list[str]:
        """Swap valid or invalid fields/datasets with random valid fields."""
        variants = []
        tokens = list(set(re.findall(r"[a-zA-Z_]\w*", expression)))
        
        allowed_non_fields = self.valid_operators | set(NEUTRALIZATIONS) | {"and", "or", "not", "if", "then", "else", "true", "false"}

        for token in tokens:
            if token.lower() in self.valid_fields or token in self.valid_fields or (token.lower() not in allowed_non_fields and token.upper() not in NEUTRALIZATIONS):
                sample_size = min(5, len(self.fields))
                
                pool = []
                if hasattr(self, 'best_fields') and self.best_fields:
                    pool.extend(random.sample(self.best_fields, min(3, len(self.best_fields))))
                pool.extend(random.sample(self.fields, sample_size))
                
                for replacement in set(pool):
                    if replacement != token:
                        # Use word boundary regex to replace the field
                        pattern = r"\b" + re.escape(token) + r"\b"
                        variant = re.sub(pattern, replacement, expression)
                        if variant != expression:
                            variants.append(variant)
        return variants

    # ── MUTATION 7: WRAPPER ───────────────────────────────────────────────

    def _wrapper_mutation(self, expression: str) -> list[str]:
        """Wrap the entire expression or top-level terms with operators."""
        variants = []

        # 1. Wrap the entire expression
        for wrapper in WRAPPER_OPERATORS:
            if wrapper in self.valid_operators:
                variants.append(f"{wrapper}({expression})")

        for wrapper in TS_WRAPPER_OPERATORS:
            if wrapper in self.valid_operators:
                for window in [5, 10, 20]:
                    variants.append(f"{wrapper}({expression}, {window})")

        return variants

    # ── HELPERS ───────────────────────────────────────────────────────────

    def _build_swap_groups(self, operators: Optional[list[str]]) -> list[list[str]]:
        if not operators:
            return DEFAULT_SWAP_GROUPS

        groups = [list(g) for g in DEFAULT_SWAP_GROUPS]
        op_to_group: dict[str, int] = {}
        
        for idx, group in enumerate(groups):
            for op in group:
                op_to_group[op] = idx

        for op_name in operators:
            if op_name not in op_to_group:
                matched = False
                for idx, group in enumerate(groups):
                    for existing in group:
                        if (
                            op_name.startswith("ts_") and existing.startswith("ts_")
                            and _similar_prefix(op_name, existing)
                        ):
                            groups[idx].append(op_name)
                            op_to_group[op_name] = idx
                            matched = True
                            break
                    if matched:
                        break
        return groups

    def _split_additive(self, expression: str) -> list[str]:
        parts = []
        depth = 0
        current = []
        for char in expression:
            if char == '(':
                depth += 1
                current.append(char)
            elif char == ')':
                depth -= 1
                current.append(char)
            elif char == '+' and depth == 0:
                parts.append(''.join(current).strip())
                current = []
            else:
                current.append(char)
        remaining = ''.join(current).strip()
        if remaining:
            parts.append(remaining)
        return parts

    def _strip_coefficient(self, term: str) -> str:
        match = re.match(r"^\d+\.?\d*\s*\*\s*\(?(.+?)\)?\s*$", term)
        if match:
            inner = match.group(1)
            if inner.count('(') != inner.count(')'):
                return term
            return inner
        return term

    def _distribute_weights(self, n: int, bias_idx: int = 0) -> list[float]:
        base = 1.0 / n
        weights = [base] * n
        shift = base * 0.2
        weights[bias_idx] += shift
        weights[(bias_idx + 1) % n] -= shift
        return [round(w, 2) for w in weights]


def _similar_prefix(a: str, b: str) -> bool:
    prefix_a = a.split("_")[:2]
    prefix_b = b.split("_")[:2]
    return prefix_a == prefix_b