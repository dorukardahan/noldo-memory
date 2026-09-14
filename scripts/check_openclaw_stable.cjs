/* Component check using the supplied stable SDK source and explicit dependencies.
 * Does not install packages, start a Gateway, or read a saved OpenClaw profile.
 * Run with an empty temporary HOME. This is not a full hook-runner check.
 */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const [host, deps, typebox] = process.argv.slice(2);
if (!host || !deps || !typebox) throw Error('Usage: node check_openclaw_stable.cjs HOST_SOURCE JITI_DEPS TYPEBOX_PACKAGE');
const {createJiti} = require(path.join(deps, 'jiti'));
const alias = {'openclaw/plugin-sdk/plugin-entry': path.join(host, 'src/plugin-sdk/plugin-entry.ts')};
const manifest = JSON.parse(fs.readFileSync(path.join(typebox, 'package.json'), 'utf8'));
for (const [key, value] of Object.entries(manifest.exports)) {
  alias[key === '.' ? 'typebox' : 'typebox/' + key.slice(2)] = path.join(typebox, value.import || value.default);
}
for (const dir of fs.readdirSync(path.join(host, 'packages'))) {
  const location = path.join(host, 'packages', dir);
  if (fs.existsSync(path.join(location, 'package.json'))) {
    alias[JSON.parse(fs.readFileSync(path.join(location, 'package.json'), 'utf8')).name] = path.join(location, 'src');
  }
}
try {
  const jiti = createJiti(path.join(host, 'package.json'), {fsCache: false, alias});
  const {definePluginEntry} = jiti(path.join(host, 'src/plugin-sdk/plugin-entry.ts'));
  const entry = jiti(path.join(__dirname, '../plugin/index.js')).default;
  assert.equal(typeof definePluginEntry, 'function');
  assert.equal(typeof entry.register, 'function');
  const {projectMessageHookMediaFacts} = jiti(path.join(host, 'src/hooks/message-hook-media.ts'));
  assert.deepEqual(projectMessageHookMediaFacts([{kind: 'image', url: 'https://example.org/dome.png', staged: true}]),
                   [{kind: 'image', url: 'https://example.org/dome.png'}]);
  console.log(JSON.stringify({stableSdkEntryLoaded: true, pluginId: entry.id, mediaProjection: true,
                             node: process.version, fullHookRunner: false}));
} catch (error) {
  console.error(error.code || error.name, error.message);
  process.exitCode = 1;
}
