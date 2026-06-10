"""Offline end-to-end check of the object memory pipeline.

Runs the real ingest stack (detector, tracker, registry, captioner, Edge
shard) over mission.mp4 with no pacing and no server, then fires the scripted
object queries and verifies each one's top hits. Use --seconds to limit the
pass while iterating; the full run is what `make prepare` uses.

Run: uv run python scripts/test_objects.py [--seconds 30]
"""

import argparse
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.captioner import Captioner
from app.constants import INGEST_FPS, MISSION_VIDEO, THUMBS_DIR, PROJECT_ROOT
from app.detector import ObjectDetector
from app.edge_store import EdgeStore
from app.encoder import get_encoder
from app.projection import MemoryMapProjector
from app.registry import ObjectRegistry

# (query, accepted classes, accepted caption keywords, expected time range)
# A hit passes if an object in the time range matches by class OR caption.
CANDIDATES = [
    ("a leather lounge chair", {"armchair", "sofa", "dining chair"}, ("chair",), (0, 33)),
    ("wine bottles on a tray", {"wine bottle", "wine glass", "tray"}, ("wine",), (49, 80)),
    ("a pool table", {"pool table"}, ("pool table",), (49, 73)),
    ("bar stools at a kitchen counter", {"bar stool", "stool", "kitchen island"}, ("stool",), (33, 80)),
    ("a bed with pillows", {"bed", "pillow", "blanket"}, ("bed",), (94, 110)),
    ("a white bathtub", {"bathtub"}, ("bathtub", "tub"), (94, 126)),
    ("a potted plant", {"houseplant"}, ("plant",), (0, 153)),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=None,
                        help="only process the first N seconds of the mission")
    args = parser.parse_args()

    encoder = get_encoder()
    encoder.warm()
    detector = ObjectDetector()
    detector.warm()
    captioner = Captioner()
    captioner.warm()

    store = EdgeStore(PROJECT_ROOT / "edge-data" / "shard-test")
    store.initialize()
    THUMBS_DIR.mkdir(parents=True, exist_ok=True)

    projector = MemoryMapProjector()
    projector.load()

    events = []
    registry = ObjectRegistry(encoder, store, projector, captioner, events.append)
    captioner.on_caption = registry.attach_caption
    captioner.start()

    cap = cv2.VideoCapture(str(MISSION_VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, round(fps / INGEST_FPS))

    t_start = time.time()
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        ts = idx / fps
        if args.seconds and ts > args.seconds:
            break
        if idx % interval == 0:
            ok, frame = cap.retrieve()
            if ok:
                detections, _ = detector.track(frame)
                thumb = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
                emb = encoder.encode_image(
                    Image.fromarray(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)))
                x, y = projector.project(emb)
                store.upsert_frame(str(uuid.uuid4()), emb,
                                   {"t": round(ts, 2), "thumb": "", "x": x, "y": y})
                registry.observe(detections, frame, ts)
        idx += 1
    cap.release()

    # Let the caption queue drain so BM25 sparse vectors are in place.
    deadline = time.time() + 120
    while captioner.pending() > 0 and time.time() < deadline:
        time.sleep(0.5)
    time.sleep(1.0)
    captioner.stop()

    wall = time.time() - t_start
    video_len = min(args.seconds or 1e9, idx / fps)
    discovered = [e for e in events if e["type"] == "object_discovered"]
    enriched = [e for e in events if e["type"] == "object_enriched"]
    print(f"\n{video_len:.0f}s of mission in {wall:.0f}s wall "
          f"({video_len / wall:.2f}x realtime)")
    print(f"objects: {len(discovered)} discovered, {len(enriched)} captioned, "
          f"{store.count} frame memories")
    print("inventory:", store.class_facets())

    print("\nsample captions:")
    caps = {e["obj"]: e["caption"] for e in enriched}
    for obj_id, cap_text in list(caps.items())[:8]:
        cls = next((r.cls for r in registry.objects.values() if r.obj_id == obj_id), "?")
        print(f"  {obj_id} [{cls}]: {cap_text}")

    all_ok = True
    print("\nobject retrieval:")
    for query, accepted, keywords, (lo, hi) in CANDIDATES:
        if args.seconds and lo >= args.seconds:
            continue
        qvec = encoder.encode_text(query)
        results, micros = store.search_objects(qvec, query, limit=3)
        rows = [(r.payload["cls"], r.payload["obj"], r.payload["t_first"],
                 r.payload.get("caption", "")) for r in results]
        ok = any(
            (cls in accepted or any(k in cap_text.lower() for k in keywords))
            and lo <= t <= hi
            for cls, _, t, cap_text in rows
        )
        all_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] '{query}' ({micros / 1000:.2f} ms)")
        for cls, obj, t, cap_text in rows:
            print(f"        {obj} {cls} @{t:.0f}s · {cap_text[:70]}")

    store.close()
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
