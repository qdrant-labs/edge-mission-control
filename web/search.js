/* In-browser vector search over the robot's object memory.

   The whole point of Qdrant Edge is that search runs where the AI operates — so
   here it runs on the visitor's own device, no server, no network round-trip:

   - Chips and the auto-script queries are precomputed against the real Edge
     shard at bake time (authentic + instant, no model needed).
   - Any free-typed query is embedded on-device with the SigLIP2 text tower
     (int8 ONNX via transformers.js, lazy-loaded on first use) and fused with
     BM25 over captions using reciprocal rank fusion — the same recipe the Edge
     shard runs, reproduced here so it executes locally.

   The dense + BM25 + RRF math mirrors scripts/bake_static.py exactly and is
   validated against the shipped bundle at bake time. */

const StaticSearch = (() => {
  const DATA = "data";
  const MODEL_REPO = "onnx-community/siglip2-base-patch16-224-ONNX";
  const TRANSFORMERS_CDN = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1/+esm";
  const RRF_K = 60;
  const OBJECT_LIMIT = 4;
  const MOMENT_LIMIT = 3;
  const MMR_LAMBDA = 0.9;
  const MMR_CANDIDATES = 100;
  const WEAK_SCORE = 0.10;
  const TEXT_MAX_LEN = 64;

  // Must match tokenize() / STOPWORDS in scripts/bake_static.py exactly.
  const STOPWORDS = new Set("a an and are as at be by for from has he in is it its of on that the to was were will with this there their they or but not have".split(" "));
  const tokenize = (text) =>
    text.toLowerCase().split(/[^a-z0-9]+/).filter((t) => t.length >= 2 && !STOPWORDS.has(t));

  let index = null;          // index.json
  let canned = null;         // canned.json
  let objVecs = null;        // Float32Array views, one per object
  let frameVecs = null;      // Float32Array views, one per frame
  let dim = 768;

  let modelPromise = null;   // lazy transformers.js load
  let modelState = "idle";   // idle | loading | ready | error
  const stateListeners = [];
  const setState = (s) => { modelState = s; stateListeners.forEach((cb) => cb(s)); };

  async function load() {
    const [idx, can, vbuf] = await Promise.all([
      fetch(`${DATA}/index.json`).then((r) => r.json()),
      fetch(`${DATA}/canned.json`).then((r) => r.json()),
      fetch(`${DATA}/vectors.bin`).then((r) => r.arrayBuffer()),
    ]);
    index = idx;
    canned = can;
    dim = idx.dim;
    const flat = new Float32Array(vbuf);
    objVecs = [];
    for (let i = 0; i < idx.n_objects; i++) objVecs.push(flat.subarray(i * dim, (i + 1) * dim));
    frameVecs = [];
    for (let j = 0; j < idx.n_frames; j++) {
      const off = (idx.n_objects + j) * dim;
      frameVecs.push(flat.subarray(off, off + dim));
    }
  }

  const ready = load();

  // ----------------------------------------------------------------- model
  function loadModel() {
    if (modelPromise) return modelPromise;
    setState("loading");
    modelPromise = (async () => {
      const { AutoTokenizer, SiglipTextModel, env } = await import(TRANSFORMERS_CDN);
      env.allowLocalModels = false;
      const tokenizer = await AutoTokenizer.from_pretrained(MODEL_REPO);
      const model = await SiglipTextModel.from_pretrained(MODEL_REPO, { dtype: "int8" });
      setState("ready");
      return { tokenizer, model };
    })().catch((e) => { setState("error"); throw e; });
    return modelPromise;
  }

  async function embed(text) {
    const { tokenizer, model } = await loadModel();
    const inputs = tokenizer(text.toLowerCase(), {
      padding: "max_length", max_length: TEXT_MAX_LEN, truncation: true,
    });
    const out = await model(inputs);
    const data = (out.pooler_output ?? out.last_hidden_state).data;
    const v = new Float32Array(data.slice(0, dim));
    let n = 0;
    for (let i = 0; i < dim; i++) n += v[i] * v[i];
    n = Math.sqrt(n) + 1e-9;
    for (let i = 0; i < dim; i++) v[i] /= n;
    return v;
  }

  // ------------------------------------------------------------- math
  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  const rankDesc = (arr) => Array.from(arr.keys()).sort((i, j) => arr[j] - arr[i]);

  function fuse(qvec, qtoks, cls) {
    const objs = index.objects;
    const dense = new Float64Array(objs.length);
    const bm = new Float64Array(objs.length);
    for (let i = 0; i < objs.length; i++) {
      if (cls && objs[i].cls !== cls) { dense[i] = -2; bm[i] = -1; continue; }
      dense[i] = dot(qvec, objVecs[i]);
      let s = 0;
      for (const t of qtoks) s += objs[i].bm25[t] || 0;
      bm[i] = s;
    }
    const rrf = new Float64Array(objs.length);
    rankDesc(dense).forEach((i, rank) => { if (dense[i] > -2) rrf[i] += 1 / (RRF_K + rank + 1); });
    rankDesc(bm).forEach((i, rank) => { if (bm[i] > 0) rrf[i] += 1 / (RRF_K + rank + 1); });
    return Array.from(rrf.keys())
      .filter((i) => rrf[i] > 0)
      .sort((a, b) => rrf[b] - rrf[a])
      .slice(0, OBJECT_LIMIT)
      .map((i) => {
        const o = objs[i];
        const score = Math.round(dense[i] * 1000) / 1000;
        return {
          obj: o.obj, cls: o.cls, caption: o.caption || null, score,
          thumb: o.thumb, t_first: o.t_first, t_last: o.t_last, box: o.box,
          xy: o.xy, sightings: o.sightings, weak: dense[i] < WEAK_SCORE,
        };
      });
  }

  function mmr(qvec) {
    const sims = frameVecs.map((v) => dot(qvec, v));
    const cand = rankDesc(sims).slice(0, MMR_CANDIDATES);
    const picked = [];
    while (picked.length < MOMENT_LIMIT && cand.length) {
      let best = -1, bestScore = -Infinity;
      for (let c = 0; c < cand.length; c++) {
        const i = cand[c];
        let maxSim = 0;
        for (const p of picked) maxSim = Math.max(maxSim, dot(frameVecs[i], frameVecs[p]));
        const mmrScore = MMR_LAMBDA * sims[i] - (1 - MMR_LAMBDA) * maxSim;
        if (mmrScore > bestScore) { bestScore = mmrScore; best = c; }
      }
      const i = cand.splice(best, 1)[0];
      picked.push(i);
    }
    return picked.map((i) => {
      const f = index.frames[i];
      return { score: Math.round(sims[i] * 1000) / 1000, thumb: f.thumb,
               video_ts: f.t, xy: f.xy };
    });
  }

  // ------------------------------------------------------------- public
  function lookupCanned(text, cls) {
    if (cls) return null;
    const c = canned[(text || "").trim().toLowerCase()];
    if (!c) return null;
    return { ...c, _canned: true };
  }

  async function search(text, cls) {
    await ready;
    const hit = lookupCanned(text, cls);
    if (hit) return hit;
    const qvec = await embed(text);
    const qtoks = new Set(tokenize(text));
    const t0 = performance.now();
    const objects = fuse(qvec, qtoks, cls || null);
    const moments = mmr(qvec);
    const latency_us = Math.round((performance.now() - t0) * 1000);
    return { type: "query_result", text, cls: cls || null, latency_us, objects, moments };
  }

  return {
    ready,
    search,
    hasCanned: (text, cls) => !!lookupCanned(text, cls),
    onModelState: (cb) => { stateListeners.push(cb); cb(modelState); },
    get modelState() { return modelState; },
    warmModel: loadModel,
  };
})();

window.StaticSearch = StaticSearch;
