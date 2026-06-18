# Qdrant Edge Mission Control — static web build

A fully static, zero-backend version of the demo for the open web. Everything
runs on the visitor's device:

- **Auto-play** replays the real recorded pipeline output (detections, object
  discovery, captions, inventory, memory map, metrics) synced to the footage.
- **Search** runs *in the browser*: the chips and the scripted queries are
  precomputed against the real Edge shard (authentic + instant); any free-typed
  query is embedded on-device with the SigLIP2 text tower (int8 ONNX via
  transformers.js, lazy-loaded on first use) and fused with BM25 over captions
  using reciprocal rank fusion — the same recipe the Edge shard runs.

No server, no GPU, no network round-trip — which is exactly the point of Edge.

## How it works

One mode for everyone. On load the robot's patrol auto-plays; contextual search
suggestions surface as pills (max 3 at a time) exactly when the robot has just
seen the relevant objects. Tap a pill, click an inventory facet, or type your
own query — all run locally in the browser. A persistent badge in the top bar
marks this as a demo and notes that the latencies shown are representative, not
native on-device numbers.

## Deploy to Vercel

The bundle is pre-baked and committed, so there is **no build step**. Point a
Vercel project at this directory:

```bash
cd web
vercel deploy --prod
```

or in the dashboard set **Root Directory = `web`**, Framework Preset = *Other*,
no build command. `vercel.json` sets long-lived caching for the heavy static
assets (`mission.mp4`, `thumbs/`, `vectors.bin`).

The SigLIP2 text model is fetched at runtime from the Hugging Face CDN (only
when a visitor types a free-text query), so no model files are hosted here.

## Regenerating the bundle

Re-run the bake from the repo root (needs the local model stack — see the main
README). It runs the real pipeline once and writes everything under `web/`:

```bash
uv run python scripts/bake_static.py
```

Outputs: `data/events.json` (replay timeline), `data/index.json` +
`data/vectors.bin` (in-browser search index), `data/canned.json` (precomputed
chip/auto results), `thumbs/` (object + frame crops), `mission.mp4`
(web-optimized footage). The bake self-validates retrieval before finishing.

To re-check the actual in-browser search path (transformers.js + int8 model)
against the shipped bundle:

```bash
cd scripts && npm install && node validate_web_search.mjs
```

## Files

- `index.html` / `style.css` / `brand/` — UI (shared with the live demo).
- `app.js` — rendering (reused from the live demo, transport swapped out).
- `replay.js` — drives the recorded timeline against the video clock; schedules
  the contextual search pills.
- `search.js` — in-browser canned + free-text vector search.
- `data/`, `thumbs/`, `mission.mp4` — the baked bundle.
