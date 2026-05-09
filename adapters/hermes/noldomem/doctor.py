#!/usr/bin/env python3
"""Basic NoldoMem Hermes adapter diagnostics.

The doctor intentionally prints only presence/status information. It never
prints API keys or memory payloads.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

from pathlib import Path


def _load_provider():
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root.parent))
    import noldomem  # type: ignore

    return noldomem.NoldoMemProvider()


def main() -> int:
    provider = _load_provider()
    available = provider.is_available()
    print(f"provider_available={str(available).lower()}")

    cfg = provider.load_config()
    print(f"base_url={cfg.base_url}")
    print(f"agent={cfg.agent}")
    print(f"namespace={cfg.namespace}")
    print(f"api_key_present={str(bool(cfg.api_key)).lower()}")

    try:
        url = cfg.base_url.rstrip("/") + "/v1/health"
        with urllib.request.urlopen(url, timeout=min(cfg.timeout_seconds, 2.0)) as resp:
            body = json.loads(resp.read().decode("utf-8") or "{}")
        print(f"health_ok={str(bool(body)).lower()}")
    except (OSError, urllib.error.URLError, ValueError) as exc:
        print("health_ok=false")
        print(f"health_error={type(exc).__name__}")

    return 0 if available else 1


if __name__ == "__main__":
    raise SystemExit(main())
