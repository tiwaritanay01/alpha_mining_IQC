"""Embedding service for experiment similarity search.

Uses a repository pattern — backend is swappable.
Current implementation: sentence-transformers (local) with SQLite storage.
Future: pgvector with PostgreSQL.
"""

import json
import logging
import math
from typing import Optional

from database.database import get_db
from database.models import ExperimentEmbedding

logger = logging.getLogger(__name__)

# Lazy-loaded model to avoid import cost on every CLI invocation
_model = None


def _get_model():
    """Lazy-load the sentence-transformers model."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded embedding model: all-MiniLM-L6-v2")
        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
            raise
    return _model


class EmbeddingService:
    """Repository-pattern embedding service with cosine similarity search.

    Current backend: sentence-transformers (local, offline).
    Embeddings stored as JSON blobs in SQLite.
    """

    def embed_text(self, text: str) -> list[float]:
        """Compute the embedding vector for a text string.

        Args:
            text: Input text to embed (theme + expression + notes).

        Returns:
            List of floats representing the embedding vector.
        """
        model = _get_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def store_embedding(
        self, exp_id: int, embedding: list[float]
    ) -> None:
        """Store or update an embedding for an experiment.

        Args:
            exp_id: Experiment ID.
            embedding: Vector as a list of floats.
        """
        serialized = json.dumps(embedding)

        with get_db() as db:
            existing = (
                db.query(ExperimentEmbedding)
                .filter(ExperimentEmbedding.experiment_id == exp_id)
                .first()
            )

            if existing:
                existing.embedding = serialized
            else:
                record = ExperimentEmbedding(
                    experiment_id=exp_id,
                    embedding=serialized,
                )
                db.add(record)

            db.flush()

    def get_embedding(self, exp_id: int) -> Optional[list[float]]:
        """Retrieve the stored embedding for an experiment.

        Args:
            exp_id: Experiment ID.

        Returns:
            The embedding vector, or None if not found.
        """
        with get_db() as db:
            record = (
                db.query(ExperimentEmbedding)
                .filter(ExperimentEmbedding.experiment_id == exp_id)
                .first()
            )

            if record and record.embedding:
                return json.loads(record.embedding)
            return None

    def find_similar(
        self, exp_id: int, top_k: int = 10
    ) -> list[tuple[int, float]]:
        """Find the most similar experiments by cosine similarity.

        Args:
            exp_id: The reference experiment ID.
            top_k: Number of similar results to return.

        Returns:
            List of (experiment_id, similarity_score) tuples,
            sorted by similarity descending. Excludes the reference experiment.
        """
        target_embedding = self.get_embedding(exp_id)
        if target_embedding is None:
            logger.warning(
                "No embedding found for experiment %d. Run 'embed-all' first.",
                exp_id,
            )
            return []

        with get_db() as db:
            all_records = db.query(ExperimentEmbedding).all()

            # Eagerly load
            records_data = []
            for r in all_records:
                records_data.append((r.experiment_id, r.embedding))

        scores: list[tuple[int, float]] = []
        for other_id, embedding_json in records_data:
            if other_id == exp_id:
                continue
            try:
                other_embedding = json.loads(embedding_json)
                score = _cosine_similarity(target_embedding, other_embedding)
                scores.append((other_id, score))
            except (json.JSONDecodeError, TypeError):
                continue

        # Sort by similarity descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)
