import json
import re
from pathlib import Path

VALID_MEMORY_TYPES = {"fact", "preference", "rule", "conversation", "lesson", "other"}


def test_hook_pack_entries_are_installable():
    repo_root = Path(__file__).resolve().parent.parent
    hooks_root = repo_root / "hooks"
    manifest = json.loads((hooks_root / "package.json").read_text())
    hook_entries = manifest["openclaw"]["hooks"]

    assert hook_entries, "openclaw.hooks must declare at least one hook"

    for entry in hook_entries:
        hook_dir = (hooks_root / entry).resolve()
        assert hook_dir.is_dir(), f"hook directory missing: {entry}"
        assert (hook_dir / "HOOK.md").is_file(), f"HOOK.md missing: {entry}"
        assert (hook_dir / "handler.js").is_file(), f"handler.js missing: {entry}"

        example_path = hook_dir / "handler.js.example"
        if example_path.is_file():
            assert (hook_dir / "handler.js").read_text() == example_path.read_text(), (
                f"handler.js drifted from handler.js.example: {entry}"
            )


def test_hook_pack_only_sends_public_memory_types():
    repo_root = Path(__file__).resolve().parent.parent
    hooks_root = repo_root / "hooks"
    hook_sources = list(hooks_root.glob("*/handler.js")) + list(
        hooks_root.glob("*/handler.js.example")
    )

    assert hook_sources

    for source in hook_sources:
        for match in re.finditer(r'memory_type:\s*"([^"]+)"', source.read_text()):
            assert match.group(1) in VALID_MEMORY_TYPES, (
                f"{source} sends invalid memory_type={match.group(1)!r}"
            )
