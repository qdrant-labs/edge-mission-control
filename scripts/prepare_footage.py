"""Stitch the mission footage and fit the 2D projection basis.

1. Concatenate source clips (normalized to 1080p30) into footage/mission.mp4.
2. Sample frames at the demo's ingest rate, embed them with the same CLIP
   model used live, and fit a PCA basis for the on-screen memory map.
3. Save mean/components/range to footage/projection.npz.

Run: uv run python scripts/prepare_footage.py
"""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.constants import FOOTAGE_DIR, INGEST_FPS, MISSION_VIDEO, PROJECTION_FILE
from app.encoder import get_encoder

# Home patrol: one house, one creator (Kindel Media, Pexels License).
# Entries are (filename, trim_start_s, trim_end_s); None = clip bound.
# Mission boundaries: living 0-21, dining 21-33, kitchen 33-49, game room
# 49-73, hallway 73-94, bedroom 94-110, bathroom 110-125, patio doors
# 125-138, backyard 138-153.
CLIPS = [
    ("7578546-hd_1920_1080_30fps.mp4", 0, None),  # entry, staircase, living room, fireplace
    ("7578552-hd_1920_1080_30fps.mp4", 0, None),  # dining table, chandelier
    ("7578540-hd_1920_1080_30fps.mp4", 0, None),  # kitchen island, stove
    ("7578549-hd_1920_1080_30fps.mp4", 0, None),  # game room: pool table, bar
    ("7578547-hd_1920_1080_30fps.mp4", 0, None),  # hallway, projection screen, stairs
    ("7578550-hd_1920_1080_30fps.mp4", 0, None),  # bedroom into ensuite
    ("7578551-hd_1920_1080_30fps.mp4", 0, None),  # bathroom: bathtub, shower
    ("7578542-hd_1920_1080_30fps.mp4", 0, None),  # sliders out to the patio
    ("7578543-hd_1920_1080_30fps.mp4", 0, None),  # backyard lawn
]


def stitch():
    inputs = []
    filters = []
    for i, (name, start, end) in enumerate(CLIPS):
        path = FOOTAGE_DIR / name
        if not path.exists():
            sys.exit(f"Missing clip: {path}")
        inputs += ["-i", str(path)]
        trim = f"trim=start={start}" + (f":end={end}" if end is not None else "")
        filters.append(
            f"[{i}:v]{trim},setpts=PTS-STARTPTS,"
            f"scale=1920:1080:force_original_aspect_ratio=increase,"
            f"crop=1920:1080,fps=30,setsar=1,format=yuv420p[v{i}]"
        )
    chain = "".join(f"[v{i}]" for i in range(len(CLIPS)))
    graph = ";".join(filters) + f";{chain}concat=n={len(CLIPS)}:v=1:a=0[out]"

    cmd = [
        "ffmpeg", "-y", *inputs,
        "-filter_complex", graph,
        "-map", "[out]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "19",
        "-movflags", "+faststart",
        str(MISSION_VIDEO),
    ]
    print("Stitching mission.mp4 ...")
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Wrote {MISSION_VIDEO}")


def fit_projection():
    print("Sampling and embedding frames ...")
    encoder = get_encoder()
    encoder.load()

    cap = cv2.VideoCapture(str(MISSION_VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, round(fps / INGEST_FPS))

    embeddings = []
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
        idx += 1
    cap.release()

    X = np.array(embeddings)
    print(f"Embedded {len(X)} frames")

    mean = X.mean(axis=0)
    centered = X - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T  # (512, 2)
    projected = centered @ components
    lo = np.percentile(projected, 2, axis=0)
    hi = np.percentile(projected, 98, axis=0)

    np.savez(PROJECTION_FILE, mean=mean, components=components, lo=lo, hi=hi)
    print(f"Wrote {PROJECTION_FILE}")


if __name__ == "__main__":
    stitch()
    fit_projection()
