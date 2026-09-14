"""Prepared file extraction must remain bound to the admitted native source."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def captured_document():
    script = r'''
import assert from 'node:assert/strict';
import {registerAutoCapture} from './plugin/src/hooks.js';
const hooks={}, rows=[];
const ctx={agentId:'alpha',sessionKey:'agent:alpha:one',sessionId:'s1',runId:'r1'};
const source={role:'user',content:'Please summarize these visit notes.',timestamp:123000,idempotencyKey:'r1:user'};
const fact={url:'media://inbound/plan.pdf',kind:'document',contentType:'application/pdf',hydrationSuppressed:true};
const stored={...source,__openclaw:{media:[fact]}};
let history={sessionKey:ctx.sessionKey,sessionId:ctx.sessionId,messages:[stored]};
const api={on(n,f){hooks[n]=f;},config:{gateway:{mode:'local',bind:'loopback',port:12345},plugins:{entries:{noldomem:{hooks:{allowConversationAccess:true}}}}}};
const cfg={captureMaxItems:3,defaultNamespace:'default'};
registerAutoCapture(api,{async store(r){rows.push(r);}},cfg,async()=>history);
const envelope=(text='The synthetic visit meets at the west pavilion.')=>
  `<file name="plan.pdf" mime="application/pdf">\n\n<<<EXTERNAL_UNTRUSTED_CONTENT id="0123456789abcdef">>>\nSource: External\n---\n${text}\n<<<END_EXTERNAL_UNTRUSTED_CONTENT id="0123456789abcdef">>>\n</file>`;
const mark=(c=ctx,prompt=envelope())=>hooks.before_prompt_build({prompt,messages:[]},c);
const event={runId:'r1',success:true,messages:[source,{role:'assistant',content:'Acknowledged.',stopReason:'stop'}]};
const run=async(e=event,c=ctx)=>{const n=rows.length;await hooks.agent_end(e,c);return rows.slice(n).filter(x=>x.source==='plugin-document-derivative');};
mark();const [first]=await run();
assert(first);assert.equal(first.text,'The synthetic visit meets at the west pavilion.');
assert.equal(first.evidence.reference,fact.url);assert.equal(first.evidence.assertion,'derived');
assert.equal(first.evidence.observed_at,123);assert.equal(first.evidence.event_id,'r1:user');
assert.equal(first.evidence.representation,'extracted_text');assert.equal(first.agent,'alpha');
assert.deepEqual(await run(),[],'completed derivative must be consumed');
for(const h of [
 {...history,sessionId:'other'}, {...history,sessionKey:'agent:beta:one'},
 {...history,messages:[stored,stored]}, {...history,messages:[{...stored,timestamp:124000}]},
 {...history,messages:[{...stored,content:'Other request'}]},
 {...history,messages:[{...stored,provenance:{kind:'inter_session'}}]},
 {...history,messages:[{...stored,display:false}]},
 {...history,messages:[{...stored,__openclaw:{media:[{...fact,url:'media://inbound/another.pdf'}]}}]},
 {...history,messages:[{...stored,__openclaw:{media:[fact,fact]}}]},
 {...history,messages:[{...stored,__openclaw:{media:[{...fact,url:'https://example.test/plan.pdf?secret=synthetic'}]}}]},
]) {const old=history;history=h;mark();assert.deepEqual(await run(),[]);history=old;}
mark();assert.deepEqual(await run(event,{...ctx,agentId:'beta',sessionKey:'agent:beta:one'}),[]);
mark();await run({...event,success:false});assert.deepEqual(await run(),[]);
for(const text of ['<file name="plan.pdf">\n[unable to extract]\n</file>',envelope().replace('END_EXTERNAL_UNTRUSTED_CONTENT id="0123456789abcdef"','END_EXTERNAL_UNTRUSTED_CONTENT id="0000000000000000"'),'x'.repeat(32001)]) {
 mark(ctx,text);assert.deepEqual(await run(),[]);
}
mark();mark(ctx,'A later plain request.');assert.deepEqual(await run(),[]);
mark();for(let i=0;i<128;i++)mark({...ctx,runId:'extra-'+i});assert.deepEqual(await run(),[]);
mark();assert.deepEqual(await run({...event,messages:[{...source,content:envelope(),__openclaw:stored.__openclaw}]}),[], 'already captured completion extraction must not gain a second representation');
mark();
api.config.plugins.entries.noldomem.hooks.allowConversationAccess=false;
assert.deepEqual(await run({...event,messages:[stored]}),[], 'a revoked grant must discard the pending derivative');
mark();assert.deepEqual(await run(),[]);
api.config.plugins.entries.noldomem.hooks.allowConversationAccess=true;
cfg.autoCaptureSource='preprocessed';mark();assert.deepEqual(await run(),[],'do not duplicate selected preprocessed capture');
console.log(JSON.stringify({passed:true,record:first}));
'''
    result = subprocess.run(['node', '--input-type=module', '--eval', script], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    receipt = json.loads(result.stdout)
    assert receipt['passed']
    return receipt['record']


def test_prepared_document_completion_identity_and_failures():
    captured_document()
