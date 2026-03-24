/**
 * Resilient dynamic import wrapper for hook dependencies.
 * If a lib module fails to load (syntax error, missing dep, etc.),
 * returns a proxy that logs the error and returns safe no-op values.
 *
 * Usage:
 *   const { recordToolCall } = await safeImport("../lib/shared-state.js", ["recordToolCall"]);
 *
 * Created: 2026-03-24 [ESM import fragility fix]
 */

const _cache = new Map();

/**
 * Dynamically import a module with graceful fallback.
 * On failure, returns an object with no-op stubs for each named export.
 * @param {string} specifier - Module path (relative to caller or absolute)
 * @param {string[]} exports - List of expected named exports to stub on failure
 * @param {object} [options] - Options
 * @param {string} [options.caller] - Caller name for logging
 * @returns {Promise<object>} Module exports or stub object
 */
export async function safeImport(specifier, exports = [], options = {}) {
  const caller = options.caller || "hook";

  if (_cache.has(specifier)) return _cache.get(specifier);

  try {
    const mod = await import(specifier);
    _cache.set(specifier, mod);
    return mod;
  } catch (err) {
    console.warn(`[${caller}] Failed to import ${specifier}: ${err.message}`);
    console.warn(`[${caller}] Continuing with no-op stubs for: ${exports.join(", ")}`);

    // Build stub object with no-op functions that log on first call
    const stubs = {};
    for (const name of exports) {
      let warned = false;
      stubs[name] = (...args) => {
        if (!warned) {
          console.warn(`[${caller}] Stub ${name}() called — ${specifier} failed to load`);
          warned = true;
        }
        return undefined;
      };
    }

    _cache.set(specifier, stubs);
    return stubs;
  }
}

/**
 * Wrap an entire handler's default export with a try/catch.
 * If the handler throws, logs the error and returns gracefully
 * instead of crashing the gateway's hook pipeline.
 * @param {function} handlerFn - The handler function to wrap
 * @param {string} hookName - Name for logging
 * @returns {function} Wrapped handler
 */
export function resilientHandler(handlerFn, hookName) {
  return async function (event, ctx) {
    try {
      return await handlerFn(event, ctx);
    } catch (err) {
      console.error(`[${hookName}] Handler crashed: ${err.message}`);
      console.error(`[${hookName}] Stack: ${err.stack?.split("\n").slice(0, 3).join(" | ")}`);
      // Return undefined — gateway continues with other hooks
      return undefined;
    }
  };
}
