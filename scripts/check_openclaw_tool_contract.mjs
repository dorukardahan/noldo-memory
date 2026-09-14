/** Reconstruct tool schemas through the installed host projector; no model/API. */
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import path from 'node:path';
const [host, baseline, candidate] = process.argv.slice(2);
const requireHost = createRequire(path.join(host, 'package.json'));
const {projectRuntimeToolInputSchema} = await import(pathToFileURL(
  requireHost.resolve('openclaw/plugin-sdk/agent-harness-runtime')));
const {normalizeOpenAIStrictCompatSchema} = await import(pathToFileURL(
  requireHost.resolve('openclaw/plugin-sdk/provider-tools')));
const {validateJsonSchemaValue} = await import(pathToFileURL(
  requireHost.resolve('openclaw/plugin-sdk/json-schema-runtime')));
const result = {};
for (const [label, root] of [['baseline', baseline], ['candidate', candidate]]) {
  const {registerTools} = await import(pathToFileURL(path.join(root, 'plugin/src/tools.js')));
  const factories = [];
  registerTools({registerTool(f) {factories.push(f);}}, {}, {});
  const store = factories.map(f => f({agentId: 'alpha'})).find(t => t.name === 'noldomem_store');
  const projected = projectRuntimeToolInputSchema(normalizeOpenAIStrictCompatSchema(store.parameters) ?? store.parameters, 'noldomem_store.inputSchema');
  assert.deepEqual(projected.violations, []);
  for (const value of [null, 0, 4102444800]) {
    const validated = validateJsonSchemaValue({schema: projected.schema,
      cacheKey: `synthetic-temporal-contract:${label}:${JSON.stringify(projected.schema)}`,
      value: {content: 'Aurora visits are on Friday at 19:30.', namespace: null,
        source: null, supersedes: 'synthetic-parent', valid_from: value}});
    assert(validated.ok, JSON.stringify(validated));
  }
  result[label] = {description: store.description, parameters: projected.schema,
    null_past_future_schema_valid: true};
}
console.log(JSON.stringify({scope: 'Installed schema reconstruction, not a recorded model request',
  model_calls: 0, api_calls: 0, ...result}, null, 2));
