import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';
const host=process.argv[2];
assert(process.env.OPENCLAW_STATE_DIR.startsWith(process.env.HOME+path.sep));
assert.equal(JSON.parse(fs.readFileSync(path.join(host,'package.json'))).version,'2026.9.3');
globalThis.fetch=()=>{throw Error('External fetch forbidden');};
async function nativeExport(name) {
  const dist = path.join(host,'dist');
  for (const file of fs.readdirSync(dist).filter(f => f.endsWith('.mjs'))) {
    const source = fs.readFileSync(path.join(dist,file),'utf8');
    const exports = source.match(/export \{([^}]+)\};?\s*$/s)?.[1];
    const alias = exports?.match(new RegExp(`\\b${name} as ([\\w$]+)(?:,|\\s|$)`))?.[1];
    if (alias) return (await import(pathToFileURL(path.join(dist,file))))[alias];
  }
  throw Error(`Installed export unavailable: ${name}`);
}

function pdf(stream) {
 const content=Buffer.from(stream);
 const objects=['<< /Type /Catalog /Pages 2 0 R >>','<< /Type /Pages /Kids [3 0 R] /Count 1 >>','<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>','<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',`<< /Length ${content.length} >>\nstream\n${stream}\nendstream`];
 let text='%PDF-1.4\n';const offsets=[0];
 objects.forEach((o,i)=>{offsets.push(Buffer.byteLength(text));text+=`${i+1} 0 obj\n${o}\nendobj\n`;});
 const start=Buffer.byteLength(text);text+=`xref\n0 ${offsets.length}\n0000000000 65535 f \n`;
 offsets.slice(1).forEach(o=>{text+=`${String(o).padStart(10,'0')} 00000 n \n`;});
 return Buffer.from(text+`trailer << /Size ${offsets.length} /Root 1 0 R >>\nstartxref\n${start}\n%%EOF`);
}
const extract=await nativeExport('extractFileContentFromBuffer');
const limits=(await nativeExport('resolveInputFileLimits'))({allowedMimes:['application/pdf'],maxBytes:100000,maxChars:2000,pdf:{maxPages:1,maxPixels:1000000,minTextChars:10}});
const config={plugins:{enabled:true,allow:['document-extract'],entries:{'document-extract':{enabled:true}}}};
const expected='The Aurora observatory booking is Friday at seven.';
const text=await extract({buffer:pdf(`BT /F1 12 Tf 72 720 Td (${expected}) Tj ET`),filename:'synthetic-text.pdf',mimeType:'application/pdf',limits,config});
assert(text.text.includes(expected));
const visual=await extract({buffer:pdf('0.5 0 0.5 rg 72 600 140 100 re f'),filename:'synthetic-visual.pdf',mimeType:'application/pdf',limits,config});
assert.equal(visual.text.trim(),'');
assert(visual.images?.length===1);
assert(visual.images[0].data.length>0);
console.log(JSON.stringify({host:'2026.9.3',node:process.version,native_input_file_pdf_text_extraction:true,native_pdf_visual_fallback_images:visual.images.length,visual_has_no_text_derivative:true,raw_audio_decoded:false,ocr_performed:false,capture_or_answer_test:false,model_requests:0}));
