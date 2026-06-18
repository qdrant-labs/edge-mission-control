"""Bake the live demo into a fully static bundle for Vercel.

Runs the real ingest stack once (detector, tracker, registry, captioner, Edge
shard) over mission.mp4 -- applying the auto-script's taught concepts at their
scripted times -- and records everything the browser needs to replay the run
and search the memory with NO server:

  web/data/events.json   timestamped event timeline (drives the replay)
  web/data/index.json    object/frame payloads + BM25 doc weights (free-text search)
  web/data/vectors.bin   packed float32 vision vectors (objects then frames)
  web/data/canned.json   precomputed results for the chips + auto queries (real shard)
  web/data/meta.json     manifest (counts, duration, vocab)
  web/thumbs/*.jpg        object + frame thumbnails (referenced by filename)
  web/mission.mp4         web-optimized footage

The dense free-text path is validated in scripts/spike_browser_search.py; the
fused (dense + BM25 + RRF) path is re-checked at the end of this bake against
the exported files so the shipped bundle is known-good before deploy.

Run: uv run python scripts/bake_static.py
"""

import base64
import json
import math
import re
import shutil
import struct
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

import qdrant_edge as q
from qdrant_edge import FieldCondition, Filter, MatchValue

from app.captioner import Captioner
from app.constants import (
    DETECTOR_VOCAB,
    INGEST_FPS,
    JPEG_QUALITY,
    MISSION_VIDEO,
    OBJECT_SEARCH_LIMIT,
    PROJECT_ROOT,
    RRF_K,
    THUMB_WIDTH,
    THUMBS_DIR,
    VECTOR_NAME,
    WEAK_OBJECT_SCORE,
)
from app.detector import ObjectDetector
from app.director import TIMELINE
from app.edge_store import EdgeStore
from app.encoder import get_encoder
from app.projection import MemoryMapProjector
from app.registry import ObjectRegistry
from app.session import BOOT_LINES

WEB = PROJECT_ROOT / "web"
DATA = WEB / "data"
WEB_THUMBS = WEB / "thumbs"
SHARD_DIR = PROJECT_ROOT / "edge-data" / "shard"

CAPTION_REPLAY_DELAY = 1.6  # seconds after discovery the caption "fills in"

# English stopwords for the BM25 reimplementation (must match web/search.js).
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were",
    "will", "with", "this", "there", "their", "they", "or", "but", "not", "have",
}


def tokenize(text):
    """Lowercase, split on non-alphanumerics, drop stopwords + len<2. No stemming
    -- kept deliberately simple so Python (bake) and JS (runtime) match exactly."""
    toks = re.split(r"[^a-z0-9]+", text.lower())
    return [t for t in toks if len(t) >= 2 and t not in STOPWORDS]


# --------------------------------------------------------------------------- run
def run_ingest():
    encoder = get_encoder()
    encoder.warm()
    detector = ObjectDetector()
    detector.warm()
    detector.reset()
    captioner = Captioner()
    captioner.warm()

    if SHARD_DIR.exists():
        shutil.rmtree(SHARD_DIR)
    store = EdgeStore(SHARD_DIR)
    store.initialize()
    if THUMBS_DIR.exists():
        shutil.rmtree(THUMBS_DIR)
    THUMBS_DIR.mkdir(parents=True, exist_ok=True)

    projector = MemoryMapProjector()
    projector.load()

    events = []  # (replay_ts, event_dict)
    registry = ObjectRegistry(encoder, store, projector, captioner, lambda e: events.append((e.get("t", 0.0), e)))
    captioner.on_caption = registry.attach_caption
    captioner.start()

    teach_actions = sorted((at, a["text"]) for at, a in TIMELINE if a["type"] == "teach")
    teach_i = 0

    cap = cv2.VideoCapture(str(MISSION_VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, round(fps / INGEST_FPS))
    last_inventory = -1.0
    idx = 0
    t0 = time.time()
    print("ingesting (offline, unpaced)...")
    while True:
        if not cap.grab():
            break
        ts = idx / fps
        if idx % interval == 0:
            # Apply the auto-script's taught concepts at their scripted time so
            # the recording shows open-vocabulary detection working.
            while teach_i < len(teach_actions) and ts >= teach_actions[teach_i][0]:
                term = teach_actions[teach_i][1]
                if detector.teach(term):
                    events.append((ts, {"type": "label_added", "text": term}))
                teach_i += 1

            ok, frame = cap.retrieve()
            if not ok:
                idx += 1
                continue
            detections, detect_ms = detector.track(frame)

            h, w = frame.shape[:2]
            thumb = cv2.resize(frame, (THUMB_WIDTH, int(h * THUMB_WIDTH / w)),
                               interpolation=cv2.INTER_AREA)
            ok, jpeg = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            thumb_name = f"t{int(ts * 1000):07d}.jpg"
            (THUMBS_DIR / thumb_name).write_bytes(jpeg.tobytes())

            image = Image.fromarray(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))
            te = time.perf_counter_ns()
            emb = encoder.encode_image(image)
            embed_ms = (time.perf_counter_ns() - te) / 1e6
            x, y = projector.project(emb)
            _, upsert_us = store.upsert_frame(str(uuid.uuid4()), emb,
                                              {"t": round(ts, 2), "thumb": thumb_name,
                                               "x": round(x, 4), "y": round(y, 4)})
            crop_ms = registry.observe(detections, frame, ts)

            events.append((ts, {
                "type": "frame_ingested",
                "thumb": thumb_name,
                "video_ts": round(ts, 2),
                "detect_ms": round(detect_ms, 1),
                "embed_ms": round(embed_ms + crop_ms, 1),
                "upsert_us": round(upsert_us, 1),
                "count": store.count,
                "objects": registry.object_count,
                "bytes": store.bytes_on_disk(),
                "xy": [round(x, 4), round(y, 4)],
                "boxes": registry.overlay(detections),
            }))

            if ts - last_inventory >= 1.0:
                last_inventory = ts
                inv = registry.inventory()
                events.append((ts, inv))
        idx += 1
    cap.release()
    duration = idx / fps

    print(f"  ingest done: {store.count} frames, {registry.object_count} objects "
          f"({duration:.0f}s mission in {time.time() - t0:.0f}s)")

    print("draining captioner...")
    deadline = time.time() + 180
    while captioner.pending() > 0 and time.time() < deadline:
        time.sleep(0.5)
    time.sleep(1.0)
    captioner.stop()

    events.append((duration, {"type": "mission_complete",
                              "count": store.count, "objects": registry.object_count}))
    return store, events, duration, encoder


# ----------------------------------------------------------------------- exports
def strip_and_time_events(events):
    """Assign replay timestamps and drop any inline base64 (thumbs are files)."""
    discovery_ts = {e["obj"]: t for t, e in events if e["type"] == "object_discovered"}
    out = []
    for t, e in events:
        e = dict(e)
        if e["type"] == "object_discovered":
            e["thumb"] = e.pop("thumb_name", e.get("thumb"))
            t = e["t"]
        elif e["type"] == "object_enriched":
            t = min(discovery_ts.get(e["obj"], 0.0) + CAPTION_REPLAY_DELAY, 1e9)
        out.append({"t": round(float(t), 3), "ev": e})
    out.sort(key=lambda r: r["t"])
    return out


def export_search_index(store):
    """Object + frame vectors/payloads + BM25 doc weights for in-browser search."""
    def scroll(kind):
        res = store.shard.scroll(q.ScrollRequest(
            limit=100000,
            filter=Filter(must=[FieldCondition(key="kind", match=MatchValue(kind))]),
            with_payload=True, with_vector=[VECTOR_NAME]))
        return res[0] if isinstance(res, tuple) else res

    objs, frames = [], []
    obj_vecs, frame_vecs = [], []
    for p in scroll("object"):
        v = np.asarray(p.vector[VECTOR_NAME], dtype=np.float32)
        v /= (np.linalg.norm(v) + 1e-9)
        obj_vecs.append(v)
        pl = p.payload
        objs.append({
            "obj": pl.get("obj"), "cls": pl.get("cls"), "caption": pl.get("caption", ""),
            "thumb": pl.get("thumb"), "t_first": pl.get("t_first"), "t_last": pl.get("t_last"),
            "sightings": pl.get("sightings"), "box": pl.get("box"),
            "xy": [pl.get("x"), pl.get("y")],
        })
    for p in scroll("frame"):
        v = np.asarray(p.vector[VECTOR_NAME], dtype=np.float32)
        v /= (np.linalg.norm(v) + 1e-9)
        frame_vecs.append(v)
        pl = p.payload
        frames.append({"thumb": pl.get("thumb"), "t": pl.get("t"),
                       "xy": [pl.get("x"), pl.get("y")]})

    # BM25 over object captions: document = "{cls}. {caption}" (matches edge_store).
    docs = [tokenize(f"{o['cls']}. {o['caption']}") for o in objs]
    N = len(docs)
    df = Counter(tok for d in set_iter(docs) for tok in d)
    avgdl = (sum(len(d) for d in docs) / N) if N else 0.0
    k, b = 1.2, 0.75
    bm25 = []
    for d in docs:
        tf = Counter(d)
        dl = len(d)
        weights = {}
        for tok, f in tf.items():
            idf = math.log(1 + (N - df[tok] + 0.5) / (df[tok] + 0.5))
            weights[tok] = idf * (f * (k + 1)) / (f + k * (1 - b + b * dl / max(avgdl, 1e-9)))
        bm25.append(weights)

    dim = len(obj_vecs[0]) if obj_vecs else 768
    buf = bytearray()
    for v in obj_vecs + frame_vecs:
        buf += struct.pack(f"<{dim}f", *v.tolist())
    (DATA / "vectors.bin").write_bytes(buf)

    index = {
        "dim": dim, "n_objects": len(objs), "n_frames": len(frames),
        "bm25": {"k": k, "b": b, "avgdl": avgdl, "N": N},
        "objects": [{**o, "bm25": bm25[i]} for i, o in enumerate(objs)],
        "frames": frames,
    }
    (DATA / "index.json").write_text(json.dumps(index))
    return objs, obj_vecs, frames, frame_vecs


def set_iter(docs):
    """Per-document unique tokens, for document-frequency counting."""
    return (set(d) for d in docs)


def run_query_card(store, encoder, text, cls=None):
    """Mirror session.run_query but emit thumb filenames (no base64)."""
    qvec = encoder.encode_text(text)
    qnorm = qvec / (np.linalg.norm(qvec) + 1e-9)
    objects, obj_us = store.search_objects(qvec, text, cls=cls)
    moments, mom_us = store.search_frames(qvec)
    cards = []
    for r in objects:
        vec = r.vector.get(VECTOR_NAME) if isinstance(r.vector, dict) else None
        score = None
        if vec is not None:
            v = np.asarray(vec, dtype=np.float32)
            score = float(qnorm @ (v / (np.linalg.norm(v) + 1e-9)))
        p = r.payload
        cards.append({
            "obj": p.get("obj"), "cls": p.get("cls"), "caption": p.get("caption"),
            "score": round(score, 3) if score is not None else None,
            "thumb": p.get("thumb"), "t_first": p.get("t_first"), "t_last": p.get("t_last"),
            "box": p.get("box"), "xy": [p.get("x"), p.get("y")], "sightings": p.get("sightings"),
            "weak": score is not None and score < WEAK_OBJECT_SCORE,
        })
    mcards = [{"score": round(r.score, 3), "thumb": r.payload.get("thumb"),
               "video_ts": r.payload.get("t"), "xy": [r.payload.get("x"), r.payload.get("y")]}
              for r in moments]
    return {"type": "query_result", "text": text, "cls": cls,
            "latency_us": round(obj_us + mom_us, 1), "objects": cards, "moments": mcards}


def export_canned(store, encoder):
    """Precompute results for the chips + auto-script queries against the real
    shard. These are authentic, instant, and need no model in the browser."""
    chips = ["a leather lounge chair", "bar stools", "a pool table",
             "wine bottles on a tray", "a bed with pillows", "a white bathtub"]
    auto = [a["text"] for _, a in TIMELINE if a["type"] == "query"]
    canned = {}
    for text in dict.fromkeys(chips + auto):
        canned[text.lower()] = run_query_card(store, encoder, text)
    (DATA / "canned.json").write_text(json.dumps(canned))
    return canned, encoder


def copy_thumbs():
    if WEB_THUMBS.exists():
        shutil.rmtree(WEB_THUMBS)
    shutil.copytree(THUMBS_DIR, WEB_THUMBS)
    return len(list(WEB_THUMBS.glob("*.jpg")))


def transcode_video():
    out = WEB / "mission.mp4"
    cmd = [
        "ffmpeg", "-y", "-i", str(MISSION_VIDEO),
        "-vf", "scale=1280:-2", "-c:v", "libx264", "-crf", "27",
        "-preset", "slow", "-an", "-movflags", "+faststart", "-pix_fmt", "yuv420p",
        str(out),
    ]
    print("transcoding video for web...")
    subprocess.run(cmd, check=True, capture_output=True)
    return out.stat().st_size


# --------------------------------------------------------------- self-validation
def validate(index_objs, obj_vecs, encoder):
    """Re-run the EXACT shipped path (dense cosine + JS-identical BM25 + RRF) over
    the exported index. If this fails, the bundle is not safe to ship."""
    CHECKS = [
        ("a leather lounge chair", {"armchair", "sofa", "dining chair"}, ("chair",)),
        ("wine bottles on a tray", {"wine bottle", "wine glass", "tray"}, ("wine",)),
        ("a pool table", {"pool table"}, ("pool table",)),
        ("bar stools", {"bar stool", "stool", "kitchen island"}, ("stool",)),
        ("a bed with pillows", {"bed", "pillow", "blanket"}, ("bed",)),
        ("a white bathtub", {"bathtub"}, ("bathtub", "tub")),
        ("a houseplant", {"houseplant"}, ("plant",)),
    ]
    idx = json.loads((DATA / "index.json").read_text())
    objs = idx["objects"]
    mat = np.stack(obj_vecs)
    ok_all = True
    print("\nfused (dense + BM25 + RRF) check over exported bundle:")
    for query, accepted, keywords in CHECKS:
        qv = encoder.encode_text(query)
        qv = qv / (np.linalg.norm(qv) + 1e-9)
        dense = (mat @ qv)
        dense_rank = np.argsort(-dense)
        # RRF over dense ranking + BM25 ranking, mirroring web/search.js.
        qtoks = set(tokenize(query))
        bm = np.array([sum(objs[i]["bm25"].get(t, 0.0) for t in qtoks) for i in range(len(objs))])
        bm_rank = np.argsort(-bm)
        rrf = np.zeros(len(objs))
        for rank, i in enumerate(dense_rank):
            rrf[i] += 1.0 / (RRF_K + rank + 1)
        for rank, i in enumerate(bm_rank):
            if bm[i] > 0:
                rrf[i] += 1.0 / (RRF_K + rank + 1)
        top = np.argsort(-rrf)[:OBJECT_SEARCH_LIMIT]
        rows = [(objs[i]["cls"], objs[i]["caption"]) for i in top]
        ok = any(cls in accepted or any(k in (cap or "").lower() for k in keywords)
                 for cls, cap in rows)
        ok_all &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] '{query}' -> {rows[0][0]}")
    return ok_all


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    store, events, duration, encoder = run_ingest()

    timeline = strip_and_time_events(events)
    (DATA / "events.json").write_text(json.dumps({
        "duration": round(duration, 2),
        "boot": BOOT_LINES,
        "vocab": len(DETECTOR_VOCAB),
        "events": timeline,
    }))
    print(f"  events.json: {len(timeline)} events")

    objs, obj_vecs, frames, frame_vecs = export_search_index(store)
    print(f"  index.json:  {len(objs)} objects, {len(frames)} frames")

    canned, encoder = export_canned(store, encoder)
    print(f"  canned.json: {len(canned)} queries")

    n_thumbs = copy_thumbs()
    print(f"  thumbs:      {n_thumbs} files")

    vbytes = transcode_video()
    print(f"  mission.mp4: {vbytes / 1e6:.1f} MB")

    (DATA / "meta.json").write_text(json.dumps({
        "duration": round(duration, 2), "objects": len(objs), "frames": len(frames),
        "vocab": len(DETECTOR_VOCAB), "thumbs": n_thumbs,
    }))

    ok = validate(objs, obj_vecs, encoder)
    store.close()
    print(f"\n{'BUNDLE OK -- safe to ship' if ok else 'VALIDATION FAILED -- do not ship'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
