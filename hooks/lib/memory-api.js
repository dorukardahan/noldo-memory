/**
 * Small non-blocking NoldoMem writer for OpenClaw hooks.
 *
 * Hooks should not make agent responses wait on post-response memory writes.
 * This helper starts bounded background writes and logs failures without
 * surfacing them to the hook caller.
 */

export function createMemoryPoster({
  baseUrl,
  apiKey,
  defaultTimeoutMs = 5000,
  maxInFlight = 8,
  label = "noldomem-hook",
  logger = console,
} = {}) {
  let inFlight = 0;
  let dropped = 0;

  function postBackground(path, payload, options = {}) {
    if (!apiKey) return false;
    if (inFlight >= maxInFlight) {
      dropped += 1;
      logger.warn(
        `[${label}] memory write queue full; dropped ${path} (inFlight=${inFlight}, dropped=${dropped})`
      );
      return false;
    }

    inFlight += 1;
    const timeoutMs = options.timeoutMs ?? defaultTimeoutMs;
    const requestLabel = options.label || label;

    fetch(`${baseUrl}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-API-Key": apiKey },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(timeoutMs),
    })
      .then(async (res) => {
        if (!res.ok) {
          const text = await res.text().catch(() => "");
          logger.warn(
            `[${requestLabel}] API error ${res.status}: ${text.slice(0, 160)}`
          );
        }
      })
      .catch((err) => {
        logger.warn(`[${requestLabel}] write failed: ${err.message || err}`);
      })
      .finally(() => {
        inFlight = Math.max(0, inFlight - 1);
      });

    return true;
  }

  return {
    postBackground,
    getStats: () => ({ inFlight, dropped }),
  };
}
