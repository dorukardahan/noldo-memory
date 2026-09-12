import json
import subprocess
from pathlib import Path


def test_openclaw_plugin_pack_is_installable():
    repo_root = Path(__file__).resolve().parent.parent
    plugin_root = repo_root / "plugin"

    manifest = json.loads((plugin_root / "openclaw.plugin.json").read_text())
    package = json.loads((plugin_root / "package.json").read_text())

    assert manifest["id"] == "noldomem"
    assert manifest["contracts"]["tools"] == [
        "noldomem_recall",
        "noldomem_store",
        "noldomem_pin",
        "noldomem_forget",
        "noldomem_relearn_source",
    ]
    assert manifest["activation"]["onStartup"] is True

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
    assert 'Use all for cross-agent recall' not in recall_source


def test_native_plugin_registers_current_openclaw_typed_hooks():
    repo_root = Path(__file__).resolve().parent.parent
    hooks_source = (repo_root / "plugin" / "src" / "hooks.js").read_text()

    assert 'api.on("after_tool_call"' in hooks_source
    assert 'api.on("before_compaction"' in hooks_source
    assert 'api.on("subagent_ended"' in hooks_source


def test_operational_capture_uses_exact_normalized_self_tool_allowlist():
    repo_root = Path(__file__).resolve().parent.parent
    script = r"""
import { registerNativeLifecycleCapture } from "./plugin/src/hooks.js";
let handler;
const api = { on(name, callback) { if (name === "after_tool_call") handler = callback; } };
const stores = [];
const client = { async store(body) { stores.push(body); } };
registerNativeLifecycleCapture(api, client, {
  enableOperationalCapture: true, enableCompactionCapture: false,
  enableSubagentCapture: false, defaultNamespace: "default",
});
const supported = ["recall", "store", "pin"].flatMap((action) => [
  `noldomem_${action}`, `plugin:noldomem_${action}`,
  `noldomem/noldomem_${action}`, `memory.noldomem_${action}`,
]);
for (const toolName of [...supported, "  PLUGIN:NOLDOMEM_RECALL  ", "Memory.NoldoMem_Store"]) {
  await handler({ toolName, params: { command: "git push" }, result: "failed" }, { agentId: "test" });
}
if (stores.length !== 0) throw new Error(`self tools captured ${stores.length} times`);
const unrelated = [
  "other:noldomem_recall", "vendor/noldomem_store",
  "unrelated.memory.noldomem_pin", "plugin:noldomem_recall_extra",
  "plugin:noldomemrecall",
];
for (const toolName of unrelated) {
  await handler({ toolName, params: { command: "git push" }, result: "completed" }, { agentId: "test" });
}
if (stores.length !== unrelated.length) throw new Error(`unrelated captures: ${stores.length}`);
stores.length = 0;
await handler({ toolName: "terminal", params: { command: "git push" }, result: "push completed" }, { agentId: "test" });
if (stores.length !== 1) throw new Error(`normal operational captures: ${stores.length}`);
if (stores[0].source !== "plugin-after-tool-call") throw new Error("unexpected capture source");
"""
    subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )


def test_operational_capture_redacts_params_result_error_and_compacts_whitespace():
    repo_root = Path(__file__).resolve().parent.parent
    script = r"""
import { registerNativeLifecycleCapture } from "./plugin/src/hooks.js";
let handler;
const stores = [];
const logs = [];
const originalWarn = console.warn;
console.warn = (...args) => logs.push(args.join(" "));
const api = { on(name, callback) { if (name === "after_tool_call") handler = callback; } };
const client = { async store(body) { stores.push(body); } };
registerNativeLifecycleCapture(api, client, {
  enableOperationalCapture: true, enableCompactionCapture: false,
  enableSubagentCapture: false, defaultNamespace: "default",
});
const sentinels = [
  "startsWithS-SENTINEL", "valueHasS-SENTINEL", "space secret SENTINEL",
  "escaped secret SENTINEL", "nested-array-SENTINEL", "cycle-SENTINEL",
];
await handler({ toolName: "terminal",
  params: {
    command: "git push", password: sentinels[0],
    nested: { API_KEY: sentinels[1], safe: "structured capture stays" },
    items: [{ "api-key": sentinels[4] }, { safe: "array capture stays" }],
  },
  result: { token: sentinels[2], nested: { passwd: sentinels[3], status: "failed safely" } },
}, { agentId: "test" });
await handler({ toolName: "terminal", params: { command: "git push", safe: "error capture stays" },
  error: { message: "failed", details: [{ pwd: sentinels[4] }, { safe: "error detail stays" }] },
}, { agentId: "test" });
await handler({ toolName: "terminal",
  params: `git push "password"="${sentinels[2]}" 'api-key'='${sentinels[3]}' many   spaces\ninside`,
  result: `failed token="${sentinels[0]} with spaces" secret='${sentinels[1]} and spaces' password="escaped \\"${sentinels[3]}\\" value"`,
}, { agentId: "test" });
const cyclic = { command: "git push", password: sentinels[5], safe: "cycle capture stays" };
cyclic.self = cyclic;
await handler({ toolName: "terminal", params: cyclic, result: "completed normally" }, { agentId: "test" });
console.warn = originalWarn;
if (stores.length !== 4) throw new Error(`captures: ${stores.length}`);
const persisted = JSON.stringify(stores);
const logged = logs.join(" ");
for (const sentinel of sentinels) {
  if (persisted.includes(sentinel)) throw new Error(`sentinel persisted: ${sentinel}`);
  if (logged.includes(sentinel)) throw new Error(`sentinel logged: ${sentinel}`);
}
for (const marker of ["<redacted>", "structured capture stays", "array capture stays",
  "error capture stays", "error detail stays", "cycle capture stays", "completed normally"]) {
  if (!persisted.includes(marker)) throw new Error(`missing capture marker: ${marker}`);
}
if (!persisted.includes("many spaces inside")) throw new Error("whitespace not compacted");
"""
    subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )


def test_operational_capture_sanitizes_json_strings_and_shared_reference_dags():
    repo_root = Path(__file__).resolve().parent.parent
    script = r"""
import { registerNativeLifecycleCapture } from "./plugin/src/hooks.js";
let handler;
const stores = [];
const logs = [];
const originalWarn = console.warn;
console.warn = (...args) => logs.push(args.join(" "));
const api = { on(name, callback) { if (name === "after_tool_call") handler = callback; } };
const client = { async store(body) { stores.push(body); } };
registerNativeLifecycleCapture(api, client, {
  enableOperationalCapture: true, enableCompactionCapture: false,
  enableSubagentCapture: false, defaultNamespace: "default",
});
const sentinels = ["json-param-SENTINEL", "json-result-SENTINEL", "json-error-SENTINEL",
  "json-array-SENTINEL", "json-deep-SENTINEL", "dag-SENTINEL", "cycle-SENTINEL",
  "top-result-object-SENTINEL", "top-result-array-SENTINEL",
  "top-error-object-SENTINEL", "top-error-array-SENTINEL"];
await handler({ toolName: "terminal", params: {
  command: "git push",
  encoded: JSON.stringify({ password: sentinels[0], safe: "json params stay",
    nestedEncoded: JSON.stringify([{ token: sentinels[4], safe: "deep json stays" }]) }),
  array: [JSON.stringify([{ api_key: sentinels[3], safe: "json array stays" }]), '"ordinary JSON primitive"'],
}, result: JSON.stringify({ secret: sentinels[1], safe: "json result stays" }) }, { agentId: "test" });
await handler({ toolName: "terminal", params: { command: "git push" },
  error: { message: "failed", encoded: JSON.stringify({ pwd: sentinels[2], safe: "json error stays" }) },
}, { agentId: "test" });
await handler({ toolName: "terminal", params: { command: "git push" },
  result: JSON.stringify({ status: "completed", nestedObject: JSON.stringify({ secret: sentinels[7], safe: "top result object stays" }),
    nestedArray: JSON.stringify([{ password: sentinels[8], safe: "top result array stays" }]) }),
}, { agentId: "test" });
await handler({ toolName: "terminal", params: { command: "git push" },
  error: JSON.stringify({ message: "failed", nestedObject: JSON.stringify({ token: sentinels[9], safe: "top error object stays" }),
    nestedArray: JSON.stringify([{ api_key: sentinels[10], safe: "top error array stays" }]) }),
}, { agentId: "test" });
await handler({ toolName: "terminal", params: { command: "git push" },
  result: "ordinary completed text stays unchanged",
}, { agentId: "test" });
const shared = { token: sentinels[5], safe: "shared safe stays" };
await handler({ toolName: "terminal", params: { command: "git push", first: shared, second: shared },
  result: "completed" }, { agentId: "test" });
const cyclic = { command: "git push", secret: sentinels[6], safe: "cycle safe stays" };
cyclic.self = cyclic;
await handler({ toolName: "terminal", params: cyclic, result: "completed" }, { agentId: "test" });
console.warn = originalWarn;
if (stores.length !== 7) throw new Error(`captures: ${stores.length}`);
const persisted = JSON.stringify(stores);
const logged = logs.join(" ");
for (const sentinel of sentinels) {
  if (persisted.includes(sentinel)) throw new Error(`sentinel persisted: ${sentinel}`);
  if (logged.includes(sentinel)) throw new Error(`sentinel logged: ${sentinel}`);
}
for (const marker of ["json params stay", "deep json stays", "json array stays", "json result stays",
  "json error stays", "cycle safe stays", "ordinary JSON primitive", "top result object stays",
  "top result array stays", "top error object stays", "top error array stays"]) {
  if (!persisted.includes(marker)) throw new Error(`missing marker: ${marker}`);
}
if (!stores.some(({ text }) => text.includes("Result: ordinary completed text stays unchanged"))) {
  throw new Error(`ordinary top-level text changed: ${persisted}`);
}
if ((persisted.match(/shared safe stays/g) || []).length !== 2) {
  throw new Error(`shared DAG occurrences were not both preserved: ${persisted}`);
}
if ((persisted.match(/token/g) || []).length < 2) throw new Error("shared secret keys not preserved/redacted twice");
"""
    subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )


def test_docs_keep_compaction_capture_on_openclaw_520_timeout_default():
    repo_root = Path(__file__).resolve().parent.parent
    root_readme = (repo_root / "README.md").read_text()
    plugin_readme = (repo_root / "plugin" / "README.md").read_text()

    for text in (root_readme, plugin_readme):
        assert '"before_compaction": 30000' in text
        assert ('"before_compaction": ' + '10000') not in text

    assert "OpenClaw 2026.5.20" in root_readme
    assert "OpenClaw 2026.5.20" in plugin_readme


def test_plugin_pin_uses_public_pin_api_id_contract():
    repo_root = Path(__file__).resolve().parent.parent
    tools_source = (repo_root / "plugin" / "src" / "tools.js").read_text()
    pin_source = tools_source.split("// ── noldomem_pin ──", 1)[1]

    assert "id: params.memory_id" in pin_source
    assert "memory_id: params.memory_id" not in pin_source
