"""Configuration for the Agent Memory system.

Loads from environment variables with sensible defaults.
Optionally reads a config.json file.

Note:
    Environment variables use the ``AGENT_MEMORY_*`` prefix.
    Legacy ``ASUMAN_MEMORY_*`` prefixes are also accepted for backward compatibility.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """Central configuration for all memory sub-systems."""

    # OpenRouter embedding
    openrouter_api_key: str = ""
    embedding_model: str = "qwen/qwen3-embedding-8b"
    embedding_dimensions: int = 4096
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Storage — default path; falls back to legacy ~/.asuman/ if it exists
    db_path: str = ""  # resolved in load_config()

    # API
    # Security: bind to localhost by default. Override with AGENT_MEMORY_HOST if needed.
    api_host: str = "127.0.0.1"
    api_port: int = 8787
    api_key: str = ""  # Required for authenticated access

    # Search weights (4-layer hybrid search) — rebalanced 2026-02-17 [S2]
    weight_semantic: float = 0.55
    weight_keyword: float = 0.25
    weight_recency: float = 0.10
    weight_strength: float = 0.10

    # Sessions
    sessions_dir: str = str(Path.home() / ".openclaw" / "agents" / "main" / "sessions")

    # Ingest
    chunk_gap_hours: float = 4.0
    max_message_len: int = 2000
    batch_size: int = 50

    # Embedding retry
    embed_max_retries: int = 3
    embed_cache_size: int = 1024

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty == OK)."""
        errors: list[str] = []
        if not self.openrouter_api_key:
            errors.append("OPENROUTER_API_KEY is required")
        if self.embedding_dimensions < 1:
            errors.append("AGENT_MEMORY_DIMENSIONS must be >= 1")
        if self.api_port < 1 or self.api_port > 65535:
            errors.append("AGENT_MEMORY_PORT must be 1-65535")
        return errors


def load_config(config_path: Optional[str] = None) -> Config:
    """Build a Config from environment variables, optionally overlaid with a JSON file.

    Environment variables (all optional except OPENROUTER_API_KEY):
        OPENROUTER_API_KEY
        AGENT_MEMORY_DB
        AGENT_MEMORY_MODEL
        AGENT_MEMORY_PORT
        AGENT_MEMORY_DIMENSIONS
        AGENT_MEMORY_HOST
        AGENT_MEMORY_SESSIONS_DIR
    """
    cfg = Config()

    # --- JSON file overlay ------------------------------------------------
    json_path = config_path or os.environ.get("AGENT_MEMORY_CONFIG") or os.environ.get("ASUMAN_MEMORY_CONFIG")
    if json_path and Path(json_path).is_file():
        with open(json_path, "r") as fh:
            data = json.load(fh)
        for key, val in data.items():
            if hasattr(cfg, key):
                expected_type = type(getattr(cfg, key))
                try:
                    setattr(cfg, key, expected_type(val))
                except (ValueError, TypeError):
                    pass  # skip bad values

    # --- Environment variable overlay -------------------------------------
    # New AGENT_MEMORY_* vars take priority; legacy ASUMAN_MEMORY_* accepted as fallback
    env_map: dict[str, tuple[str, type]] = {
        "OPENROUTER_API_KEY": ("openrouter_api_key", str),
        "AGENT_MEMORY_DB": ("db_path", str),
        "AGENT_MEMORY_MODEL": ("embedding_model", str),
        "AGENT_MEMORY_PORT": ("api_port", int),
        "AGENT_MEMORY_DIMENSIONS": ("embedding_dimensions", int),
        "AGENT_MEMORY_HOST": ("api_host", str),
        "AGENT_MEMORY_SESSIONS_DIR": ("sessions_dir", str),
        "OPENROUTER_BASE_URL": ("openrouter_base_url", str),
        "AGENT_MEMORY_API_KEY": ("api_key", str),
    }

    # Legacy env var fallbacks (checked only if new name is not set)
    legacy_map: dict[str, str] = {
        "ASUMAN_MEMORY_DB": "AGENT_MEMORY_DB",
        "ASUMAN_MEMORY_MODEL": "AGENT_MEMORY_MODEL",
        "ASUMAN_MEMORY_PORT": "AGENT_MEMORY_PORT",
        "ASUMAN_MEMORY_DIMENSIONS": "AGENT_MEMORY_DIMENSIONS",
        "ASUMAN_MEMORY_HOST": "AGENT_MEMORY_HOST",
        "ASUMAN_SESSIONS_DIR": "AGENT_MEMORY_SESSIONS_DIR",
        "ASUMAN_MEMORY_CONFIG": "AGENT_MEMORY_CONFIG",
    }

    for env_key, (attr, cast) in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            try:
                setattr(cfg, attr, cast(val))
            except (ValueError, TypeError):
                pass

    # Apply legacy env vars only where the new equivalent was not set
    for legacy_key, new_key in legacy_map.items():
        if os.environ.get(new_key) is None:
            val = os.environ.get(legacy_key)
            if val is not None and new_key in env_map:
                attr, cast = env_map[new_key]
                try:
                    setattr(cfg, attr, cast(val))
                except (ValueError, TypeError):
                    pass

    # --- Default db_path resolution ---------------------------------------
    if not cfg.db_path:
        new_dir = Path.home() / ".agent-memory"
        legacy_dir = Path.home() / ".asuman"
        if legacy_dir.exists() and not new_dir.exists():
            cfg.db_path = str(legacy_dir / "memory.sqlite")
        else:
            cfg.db_path = str(new_dir / "memory.sqlite")

    return cfg
