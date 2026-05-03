import json
import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_plugin_package_version_tracks_project_version() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    plugin_package = json.loads((ROOT / "plugin" / "package.json").read_text())
    plugin_manifest = json.loads((ROOT / "plugin" / "openclaw.plugin.json").read_text())
    plugin_index = (ROOT / "plugin" / "index.js").read_text()

    project_version = pyproject["project"]["version"]

    assert plugin_package["version"] == project_version
    assert plugin_manifest["version"] == project_version
    assert f'NOLDOMEM_PLUGIN_VERSION = "{project_version}"' in plugin_index


def test_openclaw_pack_versions_track_project_version() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    hooks_package = json.loads((ROOT / "hooks" / "package.json").read_text())

    assert hooks_package["version"] == pyproject["project"]["version"]


def test_hook_pack_is_declared_as_esm() -> None:
    hooks_package = json.loads((ROOT / "hooks" / "package.json").read_text())
    hook_files = list((ROOT / "hooks").glob("*/handler.js"))

    assert hooks_package["type"] == "module"
    assert hook_files
    assert all(
        re.search(r"\bexport\s+default\b", path.read_text())
        for path in hook_files
    )
