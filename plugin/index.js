/**
 * NoldoMem Plugin for OpenClaw
 *
 * Exposes NoldoMem long-term memory as native agent tools:
 * - noldomem_recall: Search memories (agent can call proactively)
 * - noldomem_store: Save important information
 * - noldomem_pin: Pin critical memories
 *
 * Optional hooks:
 * - before_prompt_build: Auto-inject relevant memories before every response
 * - agent_end: Auto-capture important user messages
 *
 * This plugin does NOT take over memory-core's exclusive slots.
 * It coexists with existing NoldoMem hooks and native memory-core.
 *
 * @author dorukardahan
 */

import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { buildClient } from "./src/client.js";
import { registerTools } from "./src/tools.js";
import {
  registerAutoRecall,
  registerAutoCapture,
  registerNativeLifecycleCapture,
} from "./src/hooks.js";

export const NOLDOMEM_PLUGIN_VERSION = "1.26.0";

export default definePluginEntry({
  id: "noldomem",
  name: "NoldoMem",
  description:
    "NoldoMem long-term memory tools - recall, store, pin. " +
    "Gives agents proactive access to 10K+ memories via NoldoMem API.",

  register(api) {
    const rawCfg = api.pluginConfig || {};
    const cfg = {
      baseUrl: rawCfg.baseUrl || "http://127.0.0.1:8787",
      apiKeyFile:
        rawCfg.apiKeyFile ||
        process.env.AGENT_MEMORY_API_KEY_FILE ||
        `${process.env.HOME || "~"}/.noldomem/memory-api-key`,
      defaultNamespace: rawCfg.defaultNamespace || "default",
      enableAutoRecall: rawCfg.enableAutoRecall ?? false,
      enableAutoCapture: rawCfg.enableAutoCapture ?? false,
      enableOperationalCapture: rawCfg.enableOperationalCapture ?? true,
      enableCompactionCapture: rawCfg.enableCompactionCapture ?? true,
      enableSubagentCapture: rawCfg.enableSubagentCapture ?? true,
      recallLimit: rawCfg.recallLimit ?? 5,
      recallMaxTokens: rawCfg.recallMaxTokens ?? 2000,
      captureMaxItems: rawCfg.captureMaxItems ?? 3,
      requestTimeoutMs: rawCfg.requestTimeoutMs ?? 15000,
    };

    const client = buildClient(cfg);

    // Always register tools — this is the core value
    registerTools(api, client, cfg);
    api.logger.info(
      `noldomem: tools registered (recall, store, pin) -> ${cfg.baseUrl}`
    );

    // Optional: auto-recall before every response
    if (cfg.enableAutoRecall) {
      registerAutoRecall(api, client, cfg);
      api.logger.info("noldomem: auto-recall enabled (before_prompt_build)");
    }

    // Optional: auto-capture after each turn
    if (cfg.enableAutoCapture) {
      registerAutoCapture(api, client, cfg);
      api.logger.info("noldomem: auto-capture enabled (agent_end)");
    }

    registerNativeLifecycleCapture(api, client, cfg);

    // Health check service
    api.registerService({
      id: "noldomem",
      async start() {
        try {
          const health = await client.health();
          api.logger.info(
            `noldomem: connected (status=${health.status})`
          );
        } catch (err) {
          api.logger.warn(
            `noldomem: health check failed — ${err.message || err}`
          );
        }
      },
      stop() {
        api.logger.info("noldomem: stopped");
      },
    });
  },
});
