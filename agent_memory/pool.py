"""Connection pool for per-agent memory databases.

Each agent gets its own SQLite database file:
    - main/default: {base_dir}/memory.sqlite
    - named agent:  {base_dir}/memory-{agent_id}.sqlite

The pool lazily creates MemoryStorage instances on first access.
Schema is auto-created by MemoryStorage.__init__.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from .storage import MemoryStorage

logger = logging.getLogger(__name__)

# Valid agent ID: lowercase alphanumeric, hyphens, underscores, 1-64 chars.
_AGENT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class StoragePool:
    """Manages per-agent MemoryStorage instances."""

    def __init__(self, base_dir: str, dimensions: int) -> None:
        self.base_dir = Path(base_dir)
        self.dimensions = dimensions
        self._storages: Dict[str, MemoryStorage] = {}

    @staticmethod
    def normalize_key(agent_id: Optional[str]) -> str:
        """Normalize and validate agent parameter to a cache key.

        Raises ValueError for invalid agent IDs (path traversal, reserved words).
        """
        if not agent_id or agent_id == "main":
            return "main"
        agent_id = agent_id.strip().lower()
        if agent_id == "all":
            raise ValueError("'all' is a reserved agent ID")
        if not _AGENT_ID_RE.match(agent_id):
            raise ValueError(
                f"Invalid agent ID '{agent_id}': must match [a-z0-9][a-z0-9_-]{{0,63}}"
            )
        return agent_id

    def _db_path(self, key: str) -> str:
        if key == "main":
            return str(self.base_dir / "memory.sqlite")
        return str(self.base_dir / f"memory-{key}.sqlite")

    def get(self, agent_id: Optional[str] = None) -> MemoryStorage:
        """Get or create a MemoryStorage for the given agent."""
        key = self.normalize_key(agent_id)
        if key not in self._storages:
            db_path = self._db_path(key)
            self._storages[key] = MemoryStorage(
                db_path=db_path,
                dimensions=self.dimensions,
            )
            logger.info("StoragePool: opened %s -> %s", key, db_path)
        return self._storages[key]

    def get_all_agents(self) -> List[str]:
        """Discover all agent IDs from existing database files."""
        agents: List[str] = []
        if (self.base_dir / "memory.sqlite").exists():
            agents.append("main")
        for f in sorted(self.base_dir.glob("memory-*.sqlite")):
            # Skip WAL/SHM and backup files
            if f.suffix != ".sqlite":
                continue
            if ".bak" in f.name:
                continue
            agent_id = f.stem.replace("memory-", "", 1)
            if agent_id:
                agents.append(agent_id)
        return agents

    def get_all_storages(self) -> Dict[str, MemoryStorage]:
        """Get storages for all discovered agents."""
        for agent_id in self.get_all_agents():
            self.get(agent_id)
        return dict(self._storages)

    def close_all(self) -> None:
        """Close all open database connections."""
        for storage in self._storages.values():
            storage.close()
        self._storages.clear()
