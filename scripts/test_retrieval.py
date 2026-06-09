"""Verify that the scripted demo queries retrieve frames from the right clip.

Embeds mission.mp4 frames at the live ingest rate, runs each candidate query,
and reports where the top hits land on the mission timeline. Pass criteria:
top hits fall inside the expected clip's time range, and the clip has already
been seen when the query fires in the director timeline.

Run: uv run python scripts/test_retrieval.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.constants import INGEST_FPS, MISSION_VIDEO
from app.encoder import get_encoder

# (query, fired_at_seconds, expected time range in mission.mp4)
# Home patrol boundaries: living 0-21, dining 21-33, kitchen 33-49,
# game room 49-73, hallway 73-94, bedroom 94-110, bathroom 110-125.5,
# patio 125.5-138, backyard 138-153.
CANDIDATES = [
    ("a fireplace", 23.0, (0, 21)),
    ("a kitchen island with bar stools", 45.0, (33, 49)),
    ("a pool table", 72.5, (49, 73)),
    ("a bed with pillows", 100.0, (94, 110)),
    ("a white bathtub", 119.0, (110, 125.5)),
    ("a green lawn in a backyard", 999.0, (130, 153.2)),  # spare
]


def main():
    encoder = get_encoder()
    encoder.load()

    cap = cv2.VideoCapture(str(MISSION_VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, round(fps / INGEST_FPS))

    embeddings, timestamps = [], []
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % interval == 0:
            ok, frame = cap.retrieve()
            if ok:
                thumb = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
                image = Image.fromarray(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))
                embeddings.append(encoder.encode_image(image))
                timestamps.append(idx / fps)
        idx += 1
    cap.release()

    X = np.array(embeddings)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    ts = np.array(timestamps)
    print(f"{len(X)} frames embedded, mission length {ts[-1]:.0f}s\n")

    all_ok = True
    for query, fired_at, (lo, hi) in CANDIDATES:
        q = encoder.encode_text(query)
        q = q / np.linalg.norm(q)
        scores = X @ q

        # Only frames ingested before the query fires exist in the shard.
        visible = ts <= fired_at if fired_at < 900 else np.ones_like(ts, dtype=bool)
        vis_scores = np.where(visible, scores, -1)
        top = np.argsort(vis_scores)[::-1][:4]

        in_range = [lo <= ts[i] <= hi for i in top]
        ok = all(in_range[:2])  # top 2 must be in the expected clip
        all_ok &= ok
        status = "PASS" if ok else "FAIL"
        hits = ", ".join(f"{ts[i]:5.1f}s ({vis_scores[i]:.3f})" for i in top)
        print(f"[{status}] '{query}' fired@{fired_at:.0f}s expected {lo}-{hi}s")
        print(f"        top hits: {hits}\n")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
