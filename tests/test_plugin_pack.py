import json
from pathlib import Path


def test_openclaw_plugin_pack_is_installable():
    repo_root = Path(__file__).resolve().parent.parent
    plugin_root = repo_root / "plugin"

    manifest = json.loads((plugin_root / "openclaw.plugin.json").read_text())
    package = json.loads((plugin_root / "package.json").read_text())

    assert manifest["id"] == "noldomem"
    assert package["openclaw"]["plugin"] is True
    assert package["openclaw"]["extensions"] == ["./index.js"]
    assert package["openclaw"]["install"] == {"minHostVersion": ">=2026.5.2"}
    assert package["openclaw"]["compat"] == {"pluginApi": ">=2026.5.2"}
    assert package["openclaw"]["build"] == {
        "openclawVersion": "2026.5.2",
        "pluginSdkVersion": "2026.5.2",
    }
    assert "dependencies" not in package
    assert (plugin_root / package["main"]).is_file()

    schema = manifest["configSchema"]["properties"]
    assert schema["apiKeyFile"]["default"] == "~/.noldomem/memory-api-key"
    assert schema["enableAutoRecall"]["default"] is False
    assert schema["enableOperationalCapture"]["default"] is True
    assert schema["enableCompactionCapture"]["default"] is True
    assert schema["enableSubagentCapture"]["default"] is True


def test_plugin_recall_omits_default_namespace_for_cross_namespace_search():
    repo_root = Path(__file__).resolve().parent.parent
    tools_source = (repo_root / "plugin" / "src" / "tools.js").read_text()
    recall_source = tools_source.split("// ── noldomem_store ──", 1)[0]

    assert "namespace: params.namespace || cfg.defaultNamespace" not in recall_source
    assert "if (namespace) body.namespace = namespace;" in recall_source
    assert 'if (normalized === "all") return "all";' in recall_source


def test_native_plugin_registers_current_openclaw_typed_hooks():
    repo_root = Path(__file__).resolve().parent.parent
    hooks_source = (repo_root / "plugin" / "src" / "hooks.js").read_text()

    assert 'api.on("after_tool_call"' in hooks_source
    assert 'api.on("before_compaction"' in hooks_source
    assert 'api.on("subagent_ended"' in hooks_source
