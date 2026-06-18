/* Validate the ACTUAL in-browser search path with the real library the page
   uses: @huggingface/transformers, SiglipTextModel int8, the same tokenizer,
   and the exact dense + BM25 + RRF fusion from web/search.js -- run against the
   shipped bundle in web/data. This is the end-to-end check the Python spike and
   bake could not do (they used onnxruntime directly, not transformers.js).

   Run: node scripts/validate_web_search.mjs   (after: npm i @huggingface/transformers)
*/
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { AutoTokenizer, SiglipTextModel, env } from "@huggingface/transformers";

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const DATA = path.join(ROOT, "..", "web", "data");
const REPO = "onnx-community/siglip2-base-patch16-224-ONNX";
const RRF_K = 60, OBJECT_LIMIT = 4, TEXT_MAX_LEN = 64;

const STOPWORDS = new Set("a an and are as at be by for from has he in is it its of on that the to was were will with this there their they or but not have".split(" "));
const tokenize = (t) => t.toLowerCase().split(/[^a-z0-9]+/).filter((w) => w.length >= 2 && !STOPWORDS.has(w));

const CHECKS = [
  ["a leather lounge chair", new Set(["armchair", "sofa", "dining chair"]), ["chair"]],
  ["wine bottles on a tray", new Set(["wine bottle", "wine glass", "tray"]), ["wine"]],
  ["a pool table", new Set(["pool table"]), ["pool table"]],
  ["a comfortable place to sleep", new Set(["bed", "pillow", "blanket"]), ["bed", "pillow"]],
  ["something to wash up in", new Set(["bathtub", "kitchen sink", "shower", "toilet"]), ["tub", "sink", "bath"]],
  ["a green plant in a pot", new Set(["houseplant", "vase", "flowers"]), ["plant", "flower"]],
];

const idx = JSON.parse(fs.readFileSync(path.join(DATA, "index.json")));
const dim = idx.dim;
const flat = new Float32Array(fs.readFileSync(path.join(DATA, "vectors.bin")).buffer);
const objVecs = [];
for (let i = 0; i < idx.n_objects; i++) objVecs.push(flat.subarray(i * dim, (i + 1) * dim));

const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const rankDesc = (arr) => Array.from(arr.keys()).sort((i, j) => arr[j] - arr[i]);

function fuse(qvec, qtoks) {
  const objs = idx.objects;
  const dense = objVecs.map((v) => dot(qvec, v));
  const bm = objs.map((o) => { let s = 0; for (const t of qtoks) s += o.bm25[t] || 0; return s; });
  const rrf = new Float64Array(objs.length);
  rankDesc(dense).forEach((i, r) => { rrf[i] += 1 / (RRF_K + r + 1); });
  rankDesc(bm).forEach((i, r) => { if (bm[i] > 0) rrf[i] += 1 / (RRF_K + r + 1); });
  return Array.from(rrf.keys()).sort((a, b) => rrf[b] - rrf[a]).slice(0, OBJECT_LIMIT)
    .map((i) => ({ cls: objs[i].cls, caption: objs[i].caption, score: dense[i] }));
}

env.allowLocalModels = false;
const tokenizer = await AutoTokenizer.from_pretrained(REPO);
const model = await SiglipTextModel.from_pretrained(REPO, { dtype: "int8" });

async function embed(text) {
  const inputs = tokenizer(text.toLowerCase(), { padding: "max_length", max_length: TEXT_MAX_LEN, truncation: true });
  const out = await model(inputs);
  const data = (out.pooler_output ?? out.last_hidden_state).data;
  const v = new Float32Array(Array.from(data).slice(0, dim));
  let n = 0; for (let i = 0; i < dim; i++) n += v[i] * v[i];
  n = Math.sqrt(n) + 1e-9; for (let i = 0; i < dim; i++) v[i] /= n;
  return v;
}

let pass = 0;
console.log("transformers.js int8 + dense/BM25/RRF over shipped bundle:");
for (const [q, accepted, kws] of CHECKS) {
  const qv = await embed(q);
  const rows = fuse(qv, new Set(tokenize(q)));
  const ok = rows.some((r) => accepted.has(r.cls) || kws.some((k) => (r.caption || "").toLowerCase().includes(k)));
  pass += ok;
  console.log(`  [${ok ? "PASS" : "FAIL"}] "${q}" -> ${rows[0].cls} (${rows[0].score.toFixed(3)})`);
  if (!ok) rows.forEach((r) => console.log(`         ${r.cls}  ${r.score.toFixed(3)}  ${(r.caption || "").slice(0, 50)}`));
}
console.log(`\n${pass}/${CHECKS.length} passed`);
process.exit(pass === CHECKS.length ? 0 : 1);
