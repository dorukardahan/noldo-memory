"""Security middleware for the Memory API.

Provides:
    - API key authentication (X-API-Key header)
    - Rate limiting (per-IP, in-memory sliding window)
    - Audit logging (structured request/response logging)
"""

from __future__ import annotations

import logging
import secrets
import time
from collections import defaultdict
from typing import Callable, Set

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)
audit_logger = logging.getLogger("audit")


# ---------------------------------------------------------------------------
# API Key Authentication
# ---------------------------------------------------------------------------

def _normalize_agent_scope(agent: object) -> str | None:
    """Normalize optional per-key agent scope."""
    if agent is None:
        return None
    value = str(agent).strip().lower()
    return value or None


def _load_extra_keys(keys_path: str) -> list[dict]:
    """Load additional API keys from a JSON file (if it exists).

    Format:
    {
      "keys": [
        {"key": "...", "expires_at": null|unix_ts, "label": "...", "agent": "main"},
        {"key": "..."}
      ]
    }

    - If ``agent`` is present, key is restricted to that agent.
    - If ``agent`` is absent, key is treated as admin (all agents).
    """
    import json
    from pathlib import Path

    p = Path(keys_path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
        raw_keys = data.get("keys", [])
        if not isinstance(raw_keys, list):
            logger.warning("Invalid extra key format in %s: 'keys' must be a list", keys_path)
            return []

        keys: list[dict] = []
        for entry in raw_keys:
            if not isinstance(entry, dict):
                continue
            ekey = entry.get("key")
            if not isinstance(ekey, str) or not ekey:
                continue

            normalized = {
                "key": ekey,
                "expires_at": entry.get("expires_at"),
                "label": entry.get("label"),
            }
            if "agent" in entry:
                normalized["agent"] = _normalize_agent_scope(entry.get("agent"))

            keys.append(normalized)

        return keys
    except Exception as exc:
        logger.warning("Failed to load extra keys from %s: %s", keys_path, exc)
        return []


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Require X-API-Key header on all non-exempt paths.

    Supports multiple keys: primary (from file) + extras (from JSON).
    """

    EXEMPT_PATHS: Set[str] = {"/v1/health", "/docs", "/openapi.json", "/redoc"}

    def __init__(self, app, api_key: str, extra_keys_path: str | None = None):
        super().__init__(app)
        self.api_key = api_key
        self.extra_keys_path = extra_keys_path

    def _validate_key(self, key: str) -> tuple[bool, str | None]:
        """Validate key and return (is_valid, allowed_agent)."""
        if secrets.compare_digest(key, self.api_key):
            return True, None  # primary key is admin

        if self.extra_keys_path:
            now = time.time()
            for entry in _load_extra_keys(self.extra_keys_path):
                ekey = entry.get("key", "")
                expires = entry.get("expires_at")
                if expires is not None and expires < now:
                    continue  # expired
                if ekey and secrets.compare_digest(key, ekey):
                    if "agent" in entry:
                        # restricted key: can only access this agent
                        return True, _normalize_agent_scope(entry.get("agent"))
                    # legacy/admin extra key
                    return True, None

        return False, None

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        key = request.headers.get("X-API-Key", "")
        is_valid, allowed_agent = self._validate_key(key) if key else (False, None)
        if not key or not is_valid:
            audit_logger.warning(
                "AUTH_FAIL ip=%s path=%s",
                request.client.host if request.client else "unknown",
                request.url.path,
            )
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )

        # None -> admin key (all agents), "<agent>" -> restricted key
        request.state.allowed_agent = allowed_agent

        return await call_next(request)


# ---------------------------------------------------------------------------
# Rate Limiting
# ---------------------------------------------------------------------------

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple sliding-window rate limiter (per-IP, in-memory)."""

    def __init__(self, app, max_requests: int = 120, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "0.0.0.0"
        now = time.time()

        # Prune old entries
        hits = self._hits[client_ip]
        self._hits[client_ip] = [t for t in hits if now - t < self.window]

        if len(self._hits[client_ip]) >= self.max_requests:
            audit_logger.warning(
                "RATE_LIMIT ip=%s path=%s count=%d",
                client_ip, request.url.path, len(self._hits[client_ip]),
            )
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": str(self.window)},
            )

        self._hits[client_ip].append(now)
        return await call_next(request)


# ---------------------------------------------------------------------------
# Audit Logging
# ---------------------------------------------------------------------------

class AuditLogMiddleware(BaseHTTPMiddleware):
    """Log every request with structured fields for forensic analysis."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.time()
        client_ip = request.client.host if request.client else "unknown"

        # Extract agent from query params or body (best-effort)
        agent = request.query_params.get("agent", "-")

        response = await call_next(request)
        elapsed_ms = round((time.time() - start) * 1000, 1)

        audit_logger.info(
            "method=%s path=%s status=%d ip=%s agent=%s elapsed_ms=%.1f",
            request.method,
            request.url.path,
            response.status_code,
            client_ip,
            agent,
            elapsed_ms,
        )

        return response
