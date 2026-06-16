"""SQLAlchemy models for the Alpha Mining Machine."""

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Text,
    DateTime,
    ForeignKey,
)
from sqlalchemy.orm import relationship

from database.database import Base


class Experiment(Base):
    """An alpha research experiment with lineage tracking."""

    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True)

    theme = Column(String(100), nullable=False)

    expression = Column(Text, nullable=False)

    parent_id = Column(Integer, ForeignKey("experiments.id"))

    generation = Column(Integer, default=0)

    status = Column(String(50), default="generated")

    sharpe = Column(Float)
    fitness = Column(Float)
    turnover = Column(Float)
    returns = Column(Float)

    score = Column(Float)
    classification = Column(String(50))
    is_archived = Column(Integer, default=0)  # Use Integer (0/1) for SQLite boolean compat

    structure_hash = Column(String(32), index=True)

    notes = Column(Text)

    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    parent = relationship(
        "Experiment",
        remote_side=[id],
        backref="children",
        lazy="select"
    )

    def __repr__(self) -> str:
        return f"<Experiment #{self.id} theme='{self.theme}'>"


class Operator(Base):
    """A WorldQuant Brain operator from the /operators endpoint."""

    __tablename__ = "operators"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    category = Column(String(100))
    definition = Column(Text)
    description = Column(Text)
    scope = Column(String(50))
    level = Column(String(50))

    def __repr__(self) -> str:
        return f"<Operator '{self.name}' category='{self.category}'>"


class DataField(Base):
    """A WorldQuant Brain data field from the /data-fields endpoint."""

    __tablename__ = "data_fields"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True, index=True)
    category = Column(String(100))
    description = Column(Text)
    dataset = Column(String(100))
    field_type = Column(String(50))
    instrument_type = Column(String(50))
    region = Column(String(50))
    universe = Column(String(50))
    delay = Column(Integer)

    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc)
    )

    def __repr__(self) -> str:
        return f"<DataField '{self.name}' dataset='{self.dataset}'>"


class ExperimentEmbedding(Base):
    """Stored embedding vector for an experiment (Phase 8)."""

    __tablename__ = "experiment_embeddings"

    id = Column(Integer, primary_key=True)
    experiment_id = Column(
        Integer,
        ForeignKey("experiments.id"),
        unique=True,
        nullable=False,
    )
    embedding = Column(Text)  # JSON-serialized vector

    experiment = relationship("Experiment", backref="embedding_record")

    def __repr__(self) -> str:
        return f"<ExperimentEmbedding exp_id={self.experiment_id}>"