"""Service layer for Data Field CRUD and sync from WorldQuant API."""

import random
from typing import Optional

from sqlalchemy import or_

from database.database import get_db
from database.models import DataField


def sync_fields(api_data: list[dict]) -> tuple[int, int]:
    """
    Upsert data fields from API response data.

    Returns:
        Tuple of (created_count, updated_count).
    """
    created = 0
    updated = 0

    with get_db() as db:
        for item in api_data:
            name = item.get("id", item.get("name", "")).strip()  # Brain uses "id" for field names in the JSON response
            if not name:
                continue

            existing = (
                db.query(DataField)
                .filter(DataField.name == name)
                .first()
            )

            if existing:
                # Update fields if changed
                new_cat = item.get("category", {})
                existing.category = new_cat.get("name", new_cat.get("id", existing.category)) if isinstance(new_cat, dict) else (new_cat or existing.category)
                existing.description = item.get("description", existing.description)
                existing.dataset = item.get("dataset", {}).get("id", existing.dataset) if isinstance(item.get("dataset"), dict) else item.get("dataset", existing.dataset)
                existing.field_type = item.get("type", existing.field_type)
                updated += 1
            else:
                dataset = item.get("dataset", {}).get("id") if isinstance(item.get("dataset"), dict) else item.get("dataset")
                
                new_cat = item.get("category", {})
                category_str = new_cat.get("name", new_cat.get("id", "")) if isinstance(new_cat, dict) else new_cat
                
                df = DataField(
                    name=name,
                    category=category_str,
                    description=item.get("description"),
                    dataset=dataset,
                    field_type=item.get("type"),
                    instrument_type="EQUITY",
                    region="USA",
                    universe="TOP3000",
                    delay=1,
                )
                db.add(df)
                created += 1

        db.flush()

    return created, updated


def get_all_fields() -> list[DataField]:
    """Get all data fields."""
    with get_db() as db:
        fields = db.query(DataField).order_by(DataField.name).all()
        # Eager load for detached use
        for f in fields:
            _ = f.id, f.name, f.category, f.dataset, f.description
        return fields


def get_field(name: str) -> Optional[DataField]:
    """Get a single data field by exact name."""
    with get_db() as db:
        f = db.query(DataField).filter(DataField.name == name).first()
        if f:
            _ = f.id, f.name, f.category, f.dataset, f.description
        return f


def search_fields(query: str) -> list[DataField]:
    """Search for fields by name, description, or dataset."""
    with get_db() as db:
        q = f"%{query}%"
        fields = (
            db.query(DataField)
            .filter(
                or_(
                    DataField.name.ilike(q),
                    DataField.description.ilike(q),
                    DataField.dataset.ilike(q),
                )
            )
            .order_by(DataField.name)
            .all()
        )
        for f in fields:
            _ = f.id, f.name, f.category, f.dataset, f.description
        return fields


def get_random_fields(count: int) -> list[str]:
    """Get a random sample of field names."""
    with get_db() as db:
        names = [r[0] for r in db.query(DataField.name).all()]
        if not names:
            return []
        if len(names) <= count:
            return names
        return random.sample(names, count)


def get_fields_by_category(category: str) -> list[str]:
    """Get field names for a specific category."""
    with get_db() as db:
        results = (
            db.query(DataField.name)
            .filter(DataField.category.ilike(f"%{category}%"))
            .all()
        )
        return [r[0] for r in results]
