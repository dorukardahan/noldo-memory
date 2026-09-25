#!/usr/bin/env python3
"""Privacy-safe NoldoMem Hermes adapter diagnostics."""

from __future__ import annotations

import argparse
import ipaddress
import os
import sys

from pathlib import Path
from typing import Optional, Sequence
from urllib.parse import urlsplit


def _load_provider():
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root.parent))
    import noldomem  # type: ignore

    return noldomem.NoldoMemProvider()


def _endpoint_metadata(base_url: str) -> tuple[str, str]:
    try:
        parsed = urlsplit(base_url)
        scheme = parsed.scheme.lower()
        if scheme not in {"http", "https"}:
            return "other", "invalid"
        hostname = parsed.hostname
        if not hostname:
            return scheme, "invalid"
        if hostname.lower() == "localhost":
            return scheme, "loopback"
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            return scheme, "remote"
        if address.is_loopback:
            return scheme, "loopback"
        if address.is_private:
            return scheme, "private"
        return scheme, "remote"
    except ValueError:
        return "other", "invalid"


def _bool_text(value: bool) -> str:
    return str(value).lower()


def _safe_error_type(value: object) -> str:
    name = str(value or "")
    classifications = {
        "JSONDecodeError": "InvalidResponse",
        "ValueError": "InvalidResponse",
        "_ReadinessPayloadTooLarge": "ResponseTooLarge",
        "_ReadinessDeadlineExceeded": "DeadlineExceeded",
        "_ReadinessDeadlineUnavailable": "DeadlineUnavailable",
    }
    if name in classifications:
        return classifications[name]
    allowed = {
        "HTTPError",
        "InvalidResponse",
        "DeadlineExceeded",
        "DeadlineUnavailable",
        "OSError",
        "ReadinessError",
        "ResponseTooLarge",
        "TimeoutError",
        "URLError",
    }
    return name if name in allowed else "ReadinessError"


def _resolve_hermes_memory_toolset(config: dict, platform: str) -> bool:
    """Use the host's effective resolver, not a naive platform list membership test."""
    from hermes_cli.tools_config import _get_platform_tools

    agent = config.get("agent") or {}
    if not isinstance(agent, dict):
        raise ValueError("invalid agent configuration")
    disabled = agent.get("disabled_toolsets") or []
    if isinstance(disabled, str):
        disabled = [part.strip() for part in disabled.split(",")]
    if not isinstance(disabled, list):
        raise ValueError("invalid disabled toolsets")
    memory = config.get("memory") or {}
    if not isinstance(memory, dict):
        raise ValueError("invalid memory configuration")
    # The optional compatibility flag is host-version-dependent. A config bit
    # alone cannot prove whether this host really exposes provider tools.
    if "memory" in disabled and memory.get("external_tools_enabled_when_memory_toolset_disabled") is True:
        raise ValueError("host compatibility gate cannot be projected")
    return "memory" not in disabled and "memory" in _get_platform_tools(config, platform)


def _resolve_hermes_builtin_mirror(config: dict, platform: str) -> bool:
    """A successful native memory write is mirrored even with provider tools off."""
    from tools.memory_tool import get_builtin_memory_store_flags

    return _resolve_hermes_memory_toolset(config, platform) and any(get_builtin_memory_store_flags(config))


def _host_write_diagnostic(cfg, platform: str, configured: bool) -> int:
    """Read local host configuration only; print a bounded, payload-free receipt."""
    sync = cfg.sync_turns_enabled
    requested = cfg.tools_enabled
    print(f"turn_sync_enabled={_bool_text(sync)}")
    print(f"provider_tools_requested={_bool_text(requested)}")
    try:
        import yaml

        home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes").expanduser()
        path = home / "config.yaml"
        if path.stat().st_size > 262144:
            raise ValueError("host config too large")
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("invalid host config")
        memory = config.get("memory") or {}
        if not isinstance(memory, dict):
            raise ValueError("invalid memory configuration")
        selected = memory.get("provider") == "noldomem"
        if selected:
            exposed = requested and _resolve_hermes_memory_toolset(config, platform)
            mirror = _resolve_hermes_builtin_mirror(config, platform)
        else:
            exposed = mirror = False
    except Exception:
        print("host_provider_selected=unknown")
        print("provider_tools_exposed_for_platform=unknown")
        print("built_in_mirror_available=unknown")
        print("durable_write_path_available=unknown")
        print("host_write_status=unknown")
        return 3

    durable = configured and selected and (sync or exposed or mirror)
    if not selected:
        status = "host_provider_not_selected"
    elif not configured:
        status = "provider_unconfigured"
    elif durable:
        status = "write_path_available"
    elif requested:
        status = "no_write_path"
    else:
        status = "intentional_read_only"
    print(f"host_provider_selected={_bool_text(selected)}")
    print(f"provider_tools_exposed_for_platform={_bool_text(configured and selected and exposed)}")
    print(f"built_in_mirror_available={_bool_text(configured and selected and mirror)}")
    print(f"durable_write_path_available={_bool_text(durable)}")
    print(f"host_write_status={status}")
    return 3 if status in {"no_write_path", "host_provider_not_selected"} else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Check NoldoMem adapter configuration and optional readiness.")
    parser.add_argument("--live", action="store_true", help="Perform one bounded live readiness probe.")
    parser.add_argument("--timeout", type=float, default=2.0, help="Live probe timeout, capped at 2 seconds.")
    parser.add_argument("--host", choices=["hermes"], help="Opt in to local Hermes host write-path diagnostics.")
    parser.add_argument("--platform", help="Hermes platform key (required with --host).")
    args = parser.parse_args(argv)
    if bool(args.host) != bool(args.platform):
        parser.error("--host and --platform must be used together")

    provider = _load_provider()
    configured = provider.is_available()
    cfg = provider.load_config()
    endpoint_scheme, endpoint_scope = _endpoint_metadata(cfg.base_url)
    print(f"provider_configured={_bool_text(configured)}")
    print(f"api_key_present={_bool_text(bool(cfg.api_key))}")
    print(f"endpoint_scheme={endpoint_scheme}")
    print(f"endpoint_scope={endpoint_scope}")

    host_code = _host_write_diagnostic(cfg, args.platform, configured) if args.host else 0

    if not args.live:
        print("readiness_probe=skipped")
        return host_code if configured else 1

    print("readiness_probe=completed")
    if not configured:
        print("readiness_ready=false")
        print("readiness_status=unconfigured")
        return 1

    health = provider.probe_readiness(timeout_seconds=args.timeout)
    ready = health.get("ready") is True
    print(f"readiness_ready={_bool_text(ready)}")
    print(f"readiness_status={health.get('status', 'unavailable')}")
    if isinstance(health.get("storage_ok"), bool):
        print(f"readiness_storage_ok={_bool_text(health['storage_ok'])}")
    if isinstance(health.get("embedding_ok"), bool):
        print(f"readiness_embedding_ok={_bool_text(health['embedding_ok'])}")
    uptime_seconds = health.get("uptime_seconds")
    if isinstance(uptime_seconds, (int, float)) and not isinstance(uptime_seconds, bool):
        print(f"readiness_uptime_seconds={max(0.0, float(uptime_seconds)):.1f}")
    if health.get("error_type"):
        print(f"readiness_error_type={_safe_error_type(health['error_type'])}")
    return host_code if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
