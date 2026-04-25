import json
from pathlib import Path


def test_openclaw_plugin_pack_is_installable():
    repo_root = Path(__file__).resolve().parent.parent
    plugin_root = repo_root / "plugin"

    manifest = json.loads((plugin_root / "openclaw.plugin.json").read_text())
    package = json.loads((plugin_root / "package.json").read_text())

    assert manifest["id"] == "noldomem"
    assert package["openclaw"]["plugin"] is True
    assert "dependencies" not in package
    assert (plugin_root / package["main"]).is_file()

    schema = manifest["configSchema"]["properties"]
    assert schema["apiKeyFile"]["default"] == "~/.noldomem/memory-api-key"
    assert schema["enableAutoRecall"]["default"] is False


def test_plugin_recall_omits_default_namespace_for_cross_namespace_search():
    repo_root = Path(__file__).resolve().parent.parent
    tools_source = (repo_root / "plugin" / "src" / "tools.js").read_text()
    recall_source = tools_source.split("// ── noldomem_store ──", 1)[0]

    assert "namespace: params.namespace || cfg.defaultNamespace" not in recall_source
    assert "if (namespace) body.namespace = namespace;" in recall_source
    assert 'if (normalized === "all") return "all";' in recall_source
