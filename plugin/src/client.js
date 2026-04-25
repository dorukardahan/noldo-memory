/**
 * NoldoMem HTTP client — thin wrapper around the NoldoMem REST API.
 */

import { readFileSync } from "node:fs";
import os from "node:os";

function expandHome(filePath) {
  if (typeof filePath !== "string") return filePath;
  if (filePath === "~") return os.homedir();
  if (filePath.startsWith("~/")) return `${os.homedir()}${filePath.slice(1)}`;
  return filePath;
}

export function buildClient(cfg) {
  let _apiKey = "";
  try {
    _apiKey = readFileSync(expandHome(cfg.apiKeyFile), "utf-8").trim();
  } catch (e) {
    console.warn("[noldomem-plugin] API key read error:", e.message || e);
  }

  async function request(path, method, body) {
    const url = `${cfg.baseUrl}${path}`;
    const init = {
      method,
      headers: {
        "Content-Type": "application/json",
      },
      signal: AbortSignal.timeout(cfg.requestTimeoutMs ?? 15000),
    };
    if (_apiKey) init.headers["X-API-Key"] = _apiKey;
    if (body) init.body = JSON.stringify(body);

    const res = await fetch(url, init);
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`NoldoMem ${path} failed: ${res.status} ${text.slice(0, 200)}`);
    }
    return res.json();
  }

  return {
    recall: (body) => request("/v1/recall", "POST", body),
    capture: (body) => request("/v1/capture", "POST", body),
    store: (body) => request("/v1/store", "POST", body),
    pin: (body) => request("/v1/pin", "POST", body),
    stats: (agent) =>
      request(`/v1/stats${agent ? `?agent=${encodeURIComponent(agent)}` : ""}`, "GET"),
    health: () => request("/v1/health", "GET"),
  };
}
