"""Security middleware for the Memory API.

Provides:
    - API key authentication (X-API-Key header)
    - Rate limiting (per-IP, in-memory sliding window)
    - Audit logging (structured request/response logging)
"""

from __future__ import annotations

import logging
import secrets
import sys
import time
from collections import defaultdict
from typing import Callable, Set
from uuid import UUID, uuid4

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import Scope

from .config import Config
from .metrics import collector, normalize_metric_path

logger = logging.getLogger(__name__)
audit_logger = logging.getLogger("audit")


def _attempt_id(scope: Scope) -> str:
    state = scope.setdefault("state", {})
    if "request_id" in state:
        return state["request_id"]
    values = [value.decode("latin-1") for key, value in scope.get("headers", ())
              if key.lower() == b"x-request-id"]
    candidate = values[0] if len(values) == 1 else ""
    try:
        parsed = UUID(candidate) if len(candidate) == 36 else None
        valid = parsed is not None and parsed.version == 4 and str(parsed) == candidate
    except (ValueError, TypeError, AttributeError):
        valid = False
    state["request_id"] = candidate if valid else str(uuid4())
    return state["request_id"]


def _body_limit(scope: Scope, configured=None) -> int:
    """Read live configuration without importing the API (and creating a cycle)."""
    config = configured() if callable(configured) else configured
    if config is None:
        config = getattr(getattr(scope.get("app"), "state", None), "config", None)
    if config is None:
        config = getattr(sys.modules.get("agent_memory.api"), "_config", None)
    limit = getattr(config, "api_max_body_bytes", Config.api_max_body_bytes)
    if type(limit) is not int or not 1024 <= limit <= 1048576:
        raise ValueError("invalid_config:api_max_body_bytes")
    return limit


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

    EXEMPT_PATHS: Set[str] = {"/v1/health", "/v1/health/live", "/v1/health/doctor",
                              "/docs", "/openapi.json", "/redoc"}

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
        started_at = time.perf_counter()
        request_id = _attempt_id(request.scope)

        def _record_request_metrics(response: Response) -> Response:
            response.headers["X-Request-ID"] = request_id
            duration_seconds = time.perf_counter() - started_at
            collector.record_request(
                request.method,
                request.url.path,
                response.status_code,
                duration_seconds,
            )
            return response

        if request.url.path in self.EXEMPT_PATHS:
            response = await call_next(request)
            return _record_request_metrics(response)

        key = request.headers.get("X-API-Key", "")
        is_valid, allowed_agent = self._validate_key(key) if key else (False, None)
        if not key or not is_valid:
            audit_logger.warning(
                "AUTH_FAIL ip=%s path=%s",
                request.client.host if request.client else "unknown",
                normalize_metric_path(request.url.path),
            )
            return _record_request_metrics(JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            ))

        # None -> admin key (all agents), "<agent>" -> restricted key
        request.state.allowed_agent = allowed_agent

        response = await call_next(request)
        return _record_request_metrics(response)


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
        request_id = _attempt_id(request.scope)
        client_ip = request.client.host if request.client else "0.0.0.0"
        now = time.time()

        # Prune old entries and clean up empty keys to prevent memory leak
        hits = self._hits[client_ip]
        self._hits[client_ip] = [t for t in hits if now - t < self.window]

        if not self._hits[client_ip]:
            del self._hits[client_ip]
            # Re-add for the current request below
            self._hits[client_ip] = []

        if len(self._hits[client_ip]) >= self.max_requests:
            audit_logger.warning(
                "RATE_LIMIT ip=%s path=%s count=%d",
                client_ip, normalize_metric_path(request.url.path), len(self._hits[client_ip]),
            )
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": str(self.window), "X-Request-ID": request_id},
            )

        self._hits[client_ip].append(now)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# ---------------------------------------------------------------------------
# Audit Logging
# ---------------------------------------------------------------------------

class AuditLogMiddleware:
    """Bound received bytes before dispatch and emit metadata-only audit logs.

    Pure ASGI buffering avoids BaseHTTPMiddleware receive/disconnect races. The
    buffer never exceeds the configured cap and downstream sees the exact bytes.
    Existing API registration makes controls effective even without API auth.
    """

    def __init__(self, app, config=None):
        self.app = app
        self.config = config

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = time.perf_counter()
        request_id = _attempt_id(scope)
        limit = _body_limit(scope, self.config)
        body = bytearray()
        status = 500

        async def controlled_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message = dict(message)
                headers = [(key, value) for key, value in message.get("headers", ())
                           if key.lower() != b"x-request-id"]
                headers.append((b"x-request-id", request_id.encode("ascii")))
                message["headers"] = headers
            await send(message)

        try:
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                if message["type"] != "http.request":
                    continue
                chunk = message.get("body", b"")
                if len(body) + len(chunk) > limit:
                    response = JSONResponse(status_code=413, content={"detail": "body_too_large"})
                    await response(scope, receive, controlled_send)
                    return
                body.extend(chunk)
                if not message.get("more_body", False):
                    break
            delivered = False

            async def replay_receive():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()

            await self.app(scope, replay_receive, controlled_send)
        finally:
            elapsed = time.perf_counter() - started
            path = normalize_metric_path(scope.get("path", ""))
            audit_logger.info("method=%s path=%s status=%d elapsed_ms=%.1f",
                              scope.get("method", "GET"), path, status, elapsed * 1000)


# Optional outer registration also covers authentication/CORS short circuits.
RequestControlMiddleware = AuditLogMiddleware
