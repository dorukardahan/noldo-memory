"""Stable completion snapshot omits media retained by public chat.history."""

import json
from pathlib import Path
import subprocess

from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError
import pytest

import agent_memory.api as api
from agent_memory.evidence import Evidence
from tests.test_api import _init_api_state  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = r'''
import assert from 'node:assert/strict';
const {registerAutoCapture} = await import(process.argv[1]);
const hooks={}, rows=[], reads=[];
const cfg={captureMaxItems:3,defaultNamespace:'default'};
const api={on(n,f){hooks[n]=f;},config:{gateway:{mode:'local',bind:'loopback',port:12345},
  plugins:{entries:{noldomem:{hooks:{allowConversationAccess:true}}}}}};
const ctx={agentId:'alpha',sessionKey:'agent:alpha:input',sessionId:'session-one',runId:'run-one'};
const source={role:'user',content:'What arrival detail is on the sign?',timestamp:123000,
  idempotencyKey:'run-one:user',__openclaw:{upstreamUserText:'What arrival detail is on the sign?'}};
const stored={...source,__openclaw:{media:[{url:'media://inbound/synthetic-sign.png',contentType:'image/png'}],
  mediaImageLayout:{slots:[{kind:'inline',factIndex:0}]}}};
const reply={role:'assistant',stopReason:'stop',content:[{type:'text',text:'The sign appears to direct visitors to the green gate.'}]};
const event={runId:ctx.runId,success:true,messages:[source,reply]};
let history={sessionKey:ctx.sessionKey,sessionId:ctx.sessionId,messages:[stored,reply]};
let readError=false;
registerAutoCapture(api,{async store(r){rows.push(r);}},cfg,async (_api,_cfg,c)=>{
  reads.push({...c});if(readError)throw Error('unavailable');return history;
});
const mark=(c=ctx,imagesCount=1)=>hooks.llm_input?.({runId:c.runId,sessionId:c.sessionId,imagesCount},c);
const run=async(e=event,c=ctx)=>{const n=rows.length;await hooks.agent_end(e,c);return rows.slice(n).filter(r=>r.source==='plugin-media-response');};
mark();
const [first]=await run();
assert(first,'real completion shape with media only in chat.history was not captured');
assert.equal(first.evidence.reference,'media://inbound/synthetic-sign.png');
assert.equal(first.evidence.event_id,source.idempotencyKey);
assert.equal(first.evidence.observed_at,123);
assert.equal(first.evidence.assertion,'inferred');
assert.equal(first.evidence.delivery,'generated');
assert.equal(first.agent,'alpha');
assert.match(first.text,/appears/);
assert.equal(reads.length,1);
assert.deepEqual(await run(),[],'input hint must be consumed, not reused by another completion');
assert.equal(reads.length,1);
// Count can include history/context images. It never proves a fresh attachment.
for (const candidate of [
  {...history,sessionKey:'agent:beta:input'}, {...history,sessionId:'other-session'},
  {...history,messages:[stored,stored]}, {...history,messages:[{...stored,idempotencyKey:'older:user'}]},
  {...history,messages:[{...stored,content:'Unrelated input'}]},
  {...history,messages:[{...stored,timestamp:456000}]},
  {...history,messages:[{...stored,display:false}]},
  {...history,messages:[{...stored,provenance:{kind:'inter_session'}}]},
  {...history,messages:[source]},
  {...history,messages:[{...stored,__openclaw:{...stored.__openclaw,
    mediaImageLayout:{slots:[{kind:'inline',factIndex:0}],suppressedFactIndexes:[0]}}}]},
]) {const old=history;history=candidate;mark();assert.deepEqual(await run(),[]);history=old;}
for (const value of ['media://inbound/x.png?token=secret','media://inbound/../x.png',
  'media://other/x.png','media://inbound/x.png\n','https://example.test/x.png']) {
  const old=history;history={...history,messages:[{...stored,__openclaw:{...stored.__openclaw,
    media:[{url:value,contentType:'image/png'}]}}]};
  mark();assert.deepEqual(await run(),[]);history=old;
}
const before=reads.length;
mark();assert.deepEqual(await run(event,{...ctx,agentId:'beta',sessionKey:'agent:beta:input'}),[]);
assert.equal(reads.length,before,'other agent must not borrow the input hint');
assert.deepEqual(await run(event,{...ctx,runId:'other-run'}),[]);
for(const count of [0,-1,1.5,NaN,Infinity]){mark(ctx,count);assert.deepEqual(await run(),[]);}
mark();await run({...event,success:false});assert.deepEqual(await run(),[]);
mark();mark(ctx,0);assert.deepEqual(await run(),[],'a newer input with no images invalidates its hint');
for(const [owner,key,value] of [
  [api.config.plugins.entries.noldomem.hooks,'allowConversationAccess',false],
  [api.config.gateway,'mode','remote'],[api.config.gateway,'bind','lan'],
]) {const old=owner[key];owner[key]=value;mark();const n=reads.length;assert.deepEqual(await run(),[]);
  assert.equal(reads.length,n);owner[key]=old;}
mark();assert.deepEqual(await run({...event,messages:[{...source,__openclaw:{...source.__openclaw,
  upstreamUserText:'<relevant-memories>Forgotten earlier input</relevant-memories>'}},reply]}),[]);
readError=true;mark();assert.deepEqual(await run(),[]);readError=false;
// Missing completion events cannot retain an unbounded cache of run identities.
mark();for(let n=0;n<128;n++)mark({...ctx,runId:'extra-'+n});
assert.deepEqual(await run(),[],'oldest uncompleted hint must be evicted');
mark();assert.deepEqual(await run(),[first]);
console.log(JSON.stringify(first));
'''


def history_payload():
    result = subprocess.run(
        ['node', '--input-type=module', '--eval', SCRIPT, str(ROOT / 'plugin/src/hooks.js')],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    payload = json.loads(result.stdout)
    assert Evidence(**payload['evidence']).reference == 'media://inbound/synthetic-sign.png'
    return payload


def test_media_history_identity_and_isolation():
    history_payload()


@pytest.mark.asyncio
async def test_history_reference_persistence_forgetting_and_explicit_relearning():
    payload = history_payload()
    async with AsyncClient(transport=ASGITransport(app=api.app), base_url='http://test') as client:
        created = await client.post('/v1/store', json=payload)
        assert created.status_code == 200
        memory_id = created.json()['id']
        replay = await client.post('/v1/store', json=payload)
        assert replay.json()['id'] == memory_id and replay.json()['stored'] is False
        result = (await client.post('/v1/recall', json={'agent':'alpha','query':'green gate'})).json()['results']
        row = next(r for r in result if r['id'] == memory_id)
        assert row['evidence'] == payload['evidence']
        assert not (await client.post('/v1/recall', json={'agent':'beta','query':'green gate'})).json()['results']
        assert (await client.request('DELETE', '/v1/forget', json={'agent':'alpha','id':memory_id})).status_code == 200
        assert (await client.post('/v1/store', json=payload)).status_code == 409
        assert (await client.post('/v1/store', json={**payload,'agent':'beta'})).status_code == 200
        assert (await client.post('/v1/relearn-source', json={
            'agent':'alpha','session_id':payload['session_id'],'confirm':True,
        })).json()['cleared']
        assert (await client.post('/v1/store', json=payload)).status_code == 200


@pytest.mark.parametrize('reference', [
    'media://inbound/x.png?token=secret', 'media://inbound/x.png#fragment',
    'media://inbound/../x.png', 'media://user@inbound/x.png', 'media://other/x.png',
    'media://inbound/x.png\n', 'media://inbound/%2e%2e%2fx.png',
])
def test_managed_media_reference_rejects_access_material_and_paths(reference):
    with pytest.raises(ValidationError):
        Evidence(reference=reference)
