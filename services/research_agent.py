"""RAG Research Agent for alpha experiment analysis.

Pipeline:
    1. Embed the research question
    2. Find similar experiments via embeddings
    3. Collect metrics and notes for context
    4. Build a structured prompt
    5. Send to LLM
    6. Return research summary

Supports multiple LLM backends via environment variable LLM_PROVIDER:
    - openai: OpenAI API (requires LLM_API_KEY)
    - ollama: Local Ollama instance via OpenAI-compatible endpoint
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv

from services.experiment_service import get_experiment

load_dotenv()

logger = logging.getLogger(__name__)

# Supported provider names
VALID_PROVIDERS = ("openai", "ollama")

# Default Ollama settings
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"


class ResearchAgent:
    """RAG-based research agent for alpha mining insights.

    Args:
        embedding_service: An EmbeddingService instance for similarity search.
        top_k: Number of similar experiments to include in context.
    """

    def __init__(self, embedding_service, top_k: int = 15) -> None:
        self.embedding_service = embedding_service
        self.top_k = top_k
        self.provider = os.getenv("LLM_PROVIDER", "openai").lower().strip()
        self.api_key = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        self.model = os.getenv("LLM_MODEL", "")
        self.ollama_base_url = os.getenv(
            "OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL
        ).rstrip("/")

        # Apply provider-specific model defaults
        if not self.model:
            if self.provider == "ollama":
                self.model = DEFAULT_OLLAMA_MODEL
            else:
                self.model = "gpt-4o-mini"

    def validate_config(self) -> tuple[bool, str]:
        """Validate the LLM configuration.

        Returns:
            Tuple of (is_valid, message).
        """
        if self.provider not in VALID_PROVIDERS:
            return (
                False,
                f"Unknown LLM_PROVIDER '{self.provider}'. "
                f"Supported: {', '.join(VALID_PROVIDERS)}",
            )

        if self.provider == "openai" and not self.api_key:
            return (
                False,
                "LLM_PROVIDER=openai requires LLM_API_KEY or OPENAI_API_KEY in .env",
            )

        if self.provider == "ollama" and not self.ollama_base_url:
            return (
                False,
                "LLM_PROVIDER=ollama requires OLLAMA_BASE_URL in .env "
                f"(default: {DEFAULT_OLLAMA_BASE_URL})",
            )

        return True, f"Provider: {self.provider} | Model: {self.model}"

    def test_connectivity(self) -> tuple[bool, str]:
        """Test connectivity to the configured LLM provider.

        Returns:
            Tuple of (success, message).
        """
        import requests

        valid, msg = self.validate_config()
        if not valid:
            return False, msg

        try:
            if self.provider == "openai":
                response = requests.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=10,
                )
                if response.status_code == 200:
                    return True, "OpenAI API reachable. Authentication OK."
                return (
                    False,
                    f"OpenAI API returned {response.status_code}: "
                    f"{response.text[:150]}",
                )

            elif self.provider == "ollama":
                # Check Ollama is running
                response = requests.get(
                    f"{self.ollama_base_url}/api/tags",
                    timeout=10,
                )
                if response.status_code != 200:
                    return (
                        False,
                        f"Ollama not reachable at {self.ollama_base_url} "
                        f"(status {response.status_code})",
                    )

                # Check requested model is available
                data = response.json()
                available = [
                    m.get("name", "") for m in data.get("models", [])
                ]
                # Ollama model names can include tags like ":latest"
                model_found = any(
                    self.model in name or name.startswith(self.model.split(":")[0])
                    for name in available
                )
                if model_found:
                    return (
                        True,
                        f"Ollama reachable. Model '{self.model}' available.\n"
                        f"Endpoint: {self.ollama_base_url}",
                    )
                else:
                    avail_str = ", ".join(available[:10]) or "(none)"
                    return (
                        False,
                        f"Ollama reachable but model '{self.model}' not found.\n"
                        f"Available models: {avail_str}\n"
                        f"Pull it with: ollama pull {self.model}",
                    )

        except requests.ConnectionError:
            if self.provider == "ollama":
                return (
                    False,
                    f"Cannot connect to Ollama at {self.ollama_base_url}.\n"
                    "Is Ollama running? Start it with: ollama serve",
                )
            return False, "Cannot connect to OpenAI API."
        except requests.Timeout:
            return False, f"Connection to {self.provider} timed out."
        except Exception as e:
            return False, f"Unexpected error: {e}"

    def ask(self, question: str) -> str:
        """Answer a research question using the RAG pipeline.

        Args:
            question: Natural-language research question.

        Returns:
            Research summary string.
        """
        # Validate config first
        valid, msg = self.validate_config()
        if not valid:
            logger.error("LLM config invalid: %s", msg)
            return f"LLM configuration error: {msg}"

        # Step 1: Embed the question and find similar experiments
        try:
            question_embedding = self.embedding_service.embed_text(question)
        except Exception as e:
            logger.error("Failed to embed question: %s", e)
            return self._fallback_analysis(question)

        # Find similar by embedding the question text and comparing
        # We need to search across all stored embeddings
        similar = self._find_similar_to_text(question_embedding)

        if not similar:
            return self._fallback_analysis(question)

        # Step 2-3: Collect metrics and notes
        context_experiments = []
        for exp_id, score in similar[:self.top_k]:
            exp = get_experiment(exp_id)
            if exp:
                context_experiments.append((exp, score))

        # Step 4: Build prompt
        prompt = self._build_prompt(question, context_experiments)

        # Step 5: Send to LLM
        answer = self._call_llm(prompt)

        if not answer:
            # Fallback: return structured context without LLM
            return self._format_context_as_answer(question, context_experiments)

        return answer

    def _find_similar_to_text(
        self, embedding: list[float]
    ) -> list[tuple[int, float]]:
        """Find experiments similar to an embedding vector."""
        import json
        import math

        from database.database import get_db
        from database.models import ExperimentEmbedding

        with get_db() as db:
            all_records = db.query(ExperimentEmbedding).all()
            records_data = [(r.experiment_id, r.embedding) for r in all_records]

        scores = []
        for exp_id, emb_json in records_data:
            try:
                other = json.loads(emb_json)
                dot = sum(a * b for a, b in zip(embedding, other))
                norm_a = math.sqrt(sum(a * a for a in embedding))
                norm_b = math.sqrt(sum(b * b for b in other))
                if norm_a > 0 and norm_b > 0:
                    score = dot / (norm_a * norm_b)
                    scores.append((exp_id, score))
            except (json.JSONDecodeError, TypeError):
                continue

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:self.top_k]

    def _build_prompt(
        self,
        question: str,
        context: list[tuple],
    ) -> str:
        """Build the LLM prompt with experiment context."""
        context_str = ""
        for exp, score in context:
            metrics = ""
            if exp.sharpe is not None:
                metrics = (
                    f"Sharpe={exp.sharpe:.4f}, "
                    f"Fitness={exp.fitness:.4f}, "
                    f"Turnover={exp.turnover:.4f}"
                )
            else:
                metrics = "Not yet scored"

            context_str += (
                f"---\n"
                f"Experiment #{exp.id} (similarity: {score:.3f})\n"
                f"Theme: {exp.theme}\n"
                f"Expression: {exp.expression}\n"
                f"Generation: {exp.generation} | Status: {exp.status}\n"
                f"Metrics: {metrics}\n"
                f"Notes: {exp.notes or 'None'}\n\n"
            )

        return (
            "You are a quantitative research analyst specializing in alpha mining "
            "for WorldQuant Brain IQC. Analyze the following experiments and answer "
            "the research question.\n\n"
            f"## Research Question\n{question}\n\n"
            f"## Related Experiments\n{context_str}\n"
            "## Instructions\n"
            "1. Identify patterns in the data\n"
            "2. Note which themes/operators perform well or poorly\n"
            "3. Suggest hypotheses for why certain approaches work\n"
            "4. Recommend specific next experiments to try\n"
            "5. Be concise and actionable\n"
        )

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call the configured LLM provider."""
        # Store prompt for _call_openai_compatible
        self._current_prompt = prompt

        try:
            if self.provider == "openai":
                return self._call_openai_compatible(
                    base_url="https://api.openai.com/v1",
                    api_key=self.api_key,
                )
            elif self.provider == "ollama":
                return self._call_openai_compatible(
                    base_url=f"{self.ollama_base_url}/v1",
                    api_key="ollama",  # Ollama accepts any key
                )
            else:
                logger.warning("Unknown LLM provider: %s", self.provider)
                return None
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return None

    def _call_openai_compatible(
        self,
        base_url: str,
        api_key: str,
    ) -> Optional[str]:
        """Call any OpenAI-compatible chat completions endpoint.

        Works with both OpenAI API and Ollama's OpenAI-compatible endpoint.

        Args:
            base_url: API base URL (e.g. https://api.openai.com/v1 or
                      http://localhost:11434/v1).
            api_key: API key (for Ollama, any non-empty string works).

        Returns:
            The assistant's response text, or None on failure.
        """
        import requests

        url = f"{base_url}/chat/completions"

        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a quant research analyst."},
                {"role": "user", "content": self._current_prompt},
            ],
            "temperature": 0.7,
        }

        # Only set max_tokens for OpenAI (Ollama handles it differently)
        if self.provider == "openai":
            payload["max_tokens"] = 1500

        timeout = 120 if self.provider == "ollama" else 60

        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=timeout,
            )
        except requests.ConnectionError:
            if self.provider == "ollama":
                logger.error(
                    "Cannot connect to Ollama at %s. Is it running?",
                    self.ollama_base_url,
                )
            else:
                logger.error("Cannot connect to %s", base_url)
            return None
        except requests.Timeout:
            logger.error("Request to %s timed out after %ds", base_url, timeout)
            return None

        if response.status_code == 200:
            data = response.json()
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                logger.error("Unexpected response structure: %s", str(data)[:200])
                return None

        logger.error(
            "%s API error: %d -- %s",
            self.provider.capitalize(),
            response.status_code,
            response.text[:200],
        )
        return None

    def _fallback_analysis(self, question: str) -> str:
        """Provide analysis without LLM when embeddings aren't available."""
        return (
            "Embeddings not available. Run 'embed-all' first to enable "
            "similarity search.\n\n"
            "To use the full RAG pipeline, configure your LLM provider in .env:\n"
            "  LLM_PROVIDER=ollama  (local, no API key needed)\n"
            "  LLM_PROVIDER=openai  (requires LLM_API_KEY)"
        )

    def _format_context_as_answer(
        self,
        question: str,
        context: list[tuple],
    ) -> str:
        """Format experiment context as a structured answer (no LLM fallback)."""
        lines = [
            "Context-Based Analysis (LLM unavailable)\n",
            f"Question: {question}\n",
            f"Found {len(context)} related experiments:\n",
        ]

        scored = [(exp, s) for exp, s in context if exp.sharpe is not None]
        unscored = [(exp, s) for exp, s in context if exp.sharpe is None]

        if scored:
            avg_sharpe = sum(e.sharpe for e, _ in scored) / len(scored)
            best = max(scored, key=lambda x: x[0].sharpe)
            lines.append(f"  Average Sharpe of related: {avg_sharpe:.4f}")
            lines.append(
                f"  Best related: #{best[0].id} "
                f"(Sharpe={best[0].sharpe:.4f}, Theme='{best[0].theme}')"
            )
            lines.append(f"  {len(unscored)} related experiments not yet scored")

            # Theme distribution
            themes = {}
            for exp, _ in scored:
                themes[exp.theme] = themes.get(exp.theme, 0) + 1
            lines.append("\nTheme distribution:")
            for theme, count in sorted(themes.items(), key=lambda x: -x[1]):
                lines.append(f"  {theme}: {count} experiments")

        lines.append(
            "\nConfigure LLM in .env for deeper AI-powered analysis."
        )

        return "\n".join(lines)
