"""Service layer for Operator CRUD and sync from WorldQuant API."""

from typing import Optional

from database.database import get_db
from database.models import Operator


def sync_operators(api_data: list[dict]) -> tuple[int, int]:
    """
    Upsert operators from API response data.

    Returns:
        Tuple of (created_count, updated_count).
    """
    created = 0
    updated = 0

    with get_db() as db:
        for item in api_data:
            name = item.get("name", "").strip()
            if not name:
                continue

            # Convert list values to string to avoid SQLite errors
            scope_val = item.get("scope")
            if isinstance(scope_val, list):
                scope_val = ",".join(str(x) for x in scope_val)

            level_val = item.get("level")
            if isinstance(level_val, list):
                level_val = ",".join(str(x) for x in level_val)

            existing = (
                db.query(Operator)
                .filter(Operator.name == name)
                .first()
            )

            if existing:
                # Update fields if changed
                existing.category = item.get("category", existing.category)
                existing.definition = item.get("definition", existing.definition)
                existing.description = item.get("description", existing.description)
                existing.scope = scope_val if scope_val else existing.scope
                existing.level = level_val if level_val else existing.level
                updated += 1
            else:
                op = Operator(
                    name=name,
                    category=item.get("category"),
                    definition=item.get("definition"),
                    description=item.get("description"),
                    scope=scope_val,
                    level=level_val,
                )
                db.add(op)
                created += 1

        db.flush()

    return created, updated


def get_all_operators(category: Optional[str] = None) -> list[Operator]:
    """Get operators, optionally filtered by category."""
    with get_db() as db:
        query = db.query(Operator).order_by(Operator.name)

        if category:
            query = query.filter(Operator.category.ilike(f"%{category}%"))

        operators = query.all()

        # Eagerly load for detached use
        for op in operators:
            _ = op.id, op.name, op.category, op.description, op.scope
        return operators


def get_operator_categories() -> list[str]:
    """Get distinct operator categories."""
    with get_db() as db:
        results = (
            db.query(Operator.category)
            .distinct()
            .order_by(Operator.category)
            .all()
        )
        return [r[0] for r in results if r[0]]


def get_operators_by_category(category: str) -> list[str]:
    """Get operator names for a specific category."""
    with get_db() as db:
        results = (
            db.query(Operator.name)
            .filter(Operator.category.ilike(f"%{category}%"))
            .all()
        )
        return [r[0] for r in results]
