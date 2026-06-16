"""WorldQuant Brain API client with authentication, simulation, and rate limiting."""

import logging
import os
import time
import threading
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class WorldQuantClient:
    """Client for the WorldQuant Brain API.

    Supports:
        - Session-based authentication
        - Operator fetching
        - Simulation submission and polling
        - Rate limiting with exponential backoff
        - Retry logic on server errors
    """

    BASE_URL = "https://api.worldquantbrain.com"
    MAX_RETRIES = 5
    POLL_INTERVAL = 1.0  # seconds between poll requests
    POLL_TIMEOUT = 300   # max seconds to poll before giving up
    MIN_REQUEST_INTERVAL = 1.5  # minimum seconds between requests

    def __init__(self) -> None:
        self.username = os.getenv("WQ_USERNAME", "")
        self.password = os.getenv("WQ_PASSWORD", "")
        self.session = requests.Session()
        self._last_request_time: float = 0.0
        self._throttle_lock = threading.Lock()
        self._authenticated = False

    # ── RATE LIMITING ─────────────────────────────────────────────────────

    def _throttle(self) -> None:
        """Enforce minimum interval between requests."""
        with self._throttle_lock:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.MIN_REQUEST_INTERVAL:
                time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
            self._last_request_time = time.time()

    def _request(
        self,
        method: str,
        url: str,
        retries: int = 0,
        **kwargs,
    ) -> Optional[requests.Response]:
        """Make an HTTP request with throttling and retry logic."""
        self._throttle()

        try:
            response = self.session.request(method, url, **kwargs)

            # Rate limited — exponential backoff
            if response.status_code == 429:
                if retries < self.MAX_RETRIES:
                    wait = 2 ** (retries + 1)
                    logger.warning(
                        "Rate limited (429). Waiting %ds before retry %d/%d",
                        wait, retries + 1, self.MAX_RETRIES,
                    )
                    time.sleep(wait)
                    return self._request(method, url, retries=retries + 1, **kwargs)
                logger.error("Rate limited — max retries exhausted.")
                return None

            # Unauthorized — re-authenticate and retry once
            if response.status_code == 401 and retries == 0:
                logger.warning("Got 401 Unauthorized. Re-authenticating...")
                if self.authenticate():
                    return self._request(method, url, retries=1, **kwargs)
                logger.error("Re-authentication failed.")
                return None

            # Server error — retry
            if response.status_code >= 500:
                if retries < self.MAX_RETRIES:
                    wait = 2 ** retries
                    logger.warning(
                        "Server error (%d). Retrying in %ds (%d/%d)",
                        response.status_code, wait, retries + 1, self.MAX_RETRIES,
                    )
                    time.sleep(wait)
                    return self._request(method, url, retries=retries + 1, **kwargs)
                logger.error("Server error — max retries exhausted.")
                return None

            return response

        except requests.RequestException as e:
            logger.error("Request failed: %s", e)
            if retries < self.MAX_RETRIES:
                wait = 2 ** retries
                time.sleep(wait)
                return self._request(method, url, retries=retries + 1, **kwargs)
            return None

    # ── AUTHENTICATION ────────────────────────────────────────────────────

    def authenticate(self) -> bool:
        """
        Authenticate with WorldQuant Brain.

        Uses HTTP Basic Auth and establishes a session.
        """

        if not self.username or not self.password:
            logger.error(
                "WQ_USERNAME and WQ_PASSWORD must be set in .env"
            )
            return False

        url = f"{self.BASE_URL}/authentication"

        try:
            # Browser sessions appear to use session cookies after auth.
            # First authenticate using Basic Auth.
            self.session.auth = (
                self.username,
                self.password,
            )

            response = self._request(
                "POST",
                url,
            )

            if response is None:
                logger.error("No response received.")
                return False

            logger.info(
                "Auth response: %s",
                response.status_code,
            )

            logger.debug("Auth response: %d — %s", response.status_code, response.text[:200])

            if response.status_code in (200, 201):
                self._authenticated = True
                logger.info(
                    "Authenticated successfully."
                )
                return True

            logger.error(
                "Authentication failed: %s — %s",
                response.status_code,
                response.text[:500],
            )

            return False

        except Exception as e:
            logger.exception(
                "Authentication error: %s",
                e,
            )
            return False

    # ── OPERATORS ─────────────────────────────────────────────────────────

    def fetch_operators(self) -> list[dict]:
        """Fetch all operators from the /operators endpoint.

        Returns a list of operator dicts.
        """
        url = f"{self.BASE_URL}/operators"

        response = self._request("GET", url)

        if response is None:
            return []

        if response.status_code != 200:
            logger.error("Failed to fetch operators: %s - %s", response.status_code, response.text[:500])
            return []

        data = response.json()
        logger.info("Fetched %d operators", len(data))
        
        return data

    # ── DATA FIELDS ───────────────────────────────────────────────────────

    def fetch_data_fields(self) -> list[dict]:
        """Fetch all data fields from the /data-fields endpoint.

        Returns a list of data field dicts.
        """
        url = f"{self.BASE_URL}/data-fields"
        
        response = self._request(
            "GET",
            url,
            params={
                "instrumentType": "EQUITY",
                "region": "USA",
                "delay": 1,
                "universe": "TOP3000"
            },
        )

        if response is None or response.status_code != 200:
            logger.error("Failed to fetch data fields.")
            return []

        data = response.json()
        all_fields = []

        if isinstance(data, list):
            all_fields.extend(data)
        elif isinstance(data, dict):
            all_fields.extend(data.get("results", data.get("data", [])))

        logger.info("Fetched %d data fields total.", len(all_fields))
        return all_fields

    # ── SIMULATION ────────────────────────────────────────────────────────

    def submit_simulation(
        self,
        expression: str,
        region: str = "USA",
        universe: str = "TOP3000",
        delay: int = 1,
        decay: int = 6,
        neutralization: str = "SUBINDUSTRY",
        truncation: float = 0.08,
    ) -> Optional[str]:
        """Submit a simulation to WorldQuant Brain.

        Args:
            expression: Alpha expression to simulate.
            region: Trading region.
            universe: Stock universe.
            delay: Signal delay in days.
            decay: Decay parameter.
            neutralization: Neutralization method.
            truncation: Truncation parameter.

        Returns:
            The progress URL to poll, or None on failure.
        """
        url = f"{self.BASE_URL}/simulations"

        payload = {
            "type": "REGULAR",
            "settings": {
                "instrumentType": "EQUITY",
                "region": region,
                "universe": universe,
                "delay": delay,
                "decay": decay,
                "neutralization": neutralization,
                "truncation": truncation,
                "pasteurization": "ON",
                "unitHandling": "VERIFY",
                "nanHandling": "OFF",
                "language": "FASTEXPR",
                "visualization": False,
            },
            "regular": expression,
        }

        response = self._request("POST", url, json=payload)

        if response is None:
            return None

        if response.status_code in (201, 202):
            # Simulation queued — return the progress URL
            progress_url = response.headers.get("Location", "")
            if not progress_url:
                # Try extracting from response body
                data = response.json() if response.text else {}
                progress_url = data.get("url", data.get("progressUrl", ""))
            logger.info("Simulation submitted. Progress URL: %s", progress_url)
            return progress_url

        logger.error(
            "Simulation submission failed: %d — %s",
            response.status_code,
            response.text[:300],
        )
        return None

    def poll_simulation(self, progress_url: str) -> Optional[dict]:
        """Poll a simulation until completion.

        Args:
            progress_url: The URL to poll for progress.

        Returns:
            The simulation result dict, or None on failure/timeout.
        """
        if not progress_url:
            return None

        # Ensure full URL
        if not progress_url.startswith("http"):
            if "/" not in progress_url:
                progress_url = f"/simulations/{progress_url}"
            progress_url = f"{self.BASE_URL}/{progress_url.lstrip('/')}"

        start_time = time.time()

        while (time.time() - start_time) < self.POLL_TIMEOUT:
            response = self._request("GET", progress_url)

            if response is None:
                return None

            if response.status_code == 200:
                data = response.json()

                # Check if complete
                status = data.get("status", "").upper()
                if status in ("DONE", "COMPLETE", "COMPLETED"):
                    logger.info("Simulation complete.")
                    # Extract alpha/result data
                    alpha_url = data.get("alpha", data.get("alphaUrl", ""))
                    if alpha_url:
                        return self.fetch_alpha(alpha_url)
                    return data

                if status in ("ERROR", "FAILED"):
                    logger.error("Simulation failed: %s", data.get("message", ""))
                    return None

                # Still running — log progress
                progress = data.get("progress", "?")
                logger.info("Simulation progress: %s", progress)

            elif response.status_code == 404:
                logger.error("Simulation not found at %s", progress_url)
                return None

            time.sleep(self.POLL_INTERVAL)

        logger.error("Simulation polling timed out after %ds", self.POLL_TIMEOUT)
        return None

    def poll_simulations_batch(self, progress_urls: dict[int, str], max_workers: int = 10) -> dict[int, dict]:
        """Poll multiple simulations concurrently using a thread pool.

        Args:
            progress_urls: Dict mapping an arbitrary ID (e.g. experiment ID) to its progress URL.
            max_workers: Number of concurrent polling threads.

        Returns:
            Dict mapping the same IDs to their simulation result dicts (or None on failure).
        """
        import concurrent.futures

        results = {}
        if not progress_urls:
            return results

        logger.info("Batch polling %d simulations with %d workers...", len(progress_urls), max_workers)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {
                executor.submit(self.poll_simulation, url): exp_id
                for exp_id, url in progress_urls.items()
            }
            
            for future in concurrent.futures.as_completed(future_to_id):
                exp_id = future_to_id[future]
                try:
                    result = future.result()
                    results[exp_id] = result
                except Exception as e:
                    logger.error("Simulation polling thread failed for ID %s: %s", exp_id, e)
                    results[exp_id] = None
                    
        return results

    # ── ALPHA FETCH ───────────────────────────────────────────────────────

    def fetch_alpha(self, alpha_url: str) -> Optional[dict]:
        """Fetch alpha details/metrics from a URL.

        Args:
            alpha_url: Full URL or path to the alpha resource.

        Returns:
            Alpha data dict, or None on failure.
        """
        if not alpha_url.startswith("http"):
            if "/" not in alpha_url:
                alpha_url = f"/alphas/{alpha_url}"
            alpha_url = f"{self.BASE_URL}/{alpha_url.lstrip('/')}"

        response = self._request("GET", alpha_url)

        if response is None or response.status_code != 200:
            logger.error("Failed to fetch alpha from %s", alpha_url)
            return None

        data = response.json()
        logger.info("Fetched alpha data: %s", str(data)[:200])
        return data
