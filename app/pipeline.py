import base64
import logging
import threading
import time

import cv2
import numpy as np
from PIL import Image

from .constants import (
    INGEST_FPS,
    JPEG_QUALITY,
    LABEL_THRESHOLD,
    LABEL_TOP_K,
    MISSION_VIDEO,
    THUMB_WIDTH,
    THUMBS_DIR,
)

logger = logging.getLogger(__name__)


class IngestPipeline:
    """Real-time loop: capture frame -> embed on-device -> upsert to edge shard.

    Paced against the wall clock so it stays aligned with the video element
    playing the same file in the browser.
    """

    def __init__(self, encoder, store, sync, projector, emit, labels=None):
        self.encoder = encoder
        self.store = store
        self.sync = sync
        self.projector = projector
        self.emit = emit  # callback(event_dict), thread-safe
        # (names, matrix) replaced atomically so concepts can be taught live.
        self.labels = labels or ([], None)
        # The most recently taught concept always shows its live score on the
        # HUD, even when the generic labels outrank it.
        self.pinned = None
        self.is_running = False
        self.thread = None

    def _recognize(self, embedding: np.ndarray) -> list:
        """Zero-shot labels for the HUD: cosine against the open vocabulary."""
        names, matrix = self.labels
        if matrix is None:
            return []
        vec = embedding / (np.linalg.norm(embedding) + 1e-9)
        scores = matrix @ vec
        top = np.argsort(scores)[::-1][:LABEL_TOP_K]
        out = [
            [names[i], round(float(scores[i]), 3)]
            for i in top
            if scores[i] >= LABEL_THRESHOLD
        ]
        pinned = self.pinned
        if pinned in names and pinned not in [n for n, *_ in out]:
            i = names.index(pinned)
            out.append([pinned, round(float(scores[i]), 3), "pinned"])
        return out

    def start(self):
        THUMBS_DIR.mkdir(parents=True, exist_ok=True)
        self.is_running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_running = False

    def _run(self):
        cap = cv2.VideoCapture(str(MISSION_VIDEO))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, round(video_fps / INGEST_FPS))

        start = time.monotonic()
        frame_idx = 0
        completed = False

        while self.is_running:
            ok = cap.grab()
            if not ok:
                completed = True
                break

            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            ok, frame = cap.retrieve()
            video_ts = frame_idx / video_fps
            frame_idx += 1
            if not ok:
                continue
            # Stay aligned with the browser's video playback.
            lag = (start + video_ts) - time.monotonic()
            if lag > 0:
                time.sleep(lag)

            self._ingest(frame, video_ts)

        cap.release()
        logger.info("Pipeline finished: %d vectors", self.store.count)
        if completed:
            # Natural end of the mission (not an external stop).
            self.emit({"type": "mission_complete", "count": self.store.count})

    def _ingest(self, frame, video_ts: float):
        h, w = frame.shape[:2]
        thumb_h = int(h * THUMB_WIDTH / w)
        thumb = cv2.resize(frame, (THUMB_WIDTH, thumb_h), interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            return
        jpeg_bytes = jpeg.tobytes()

        thumb_name = f"t{int(video_ts * 1000):07d}.jpg"
        (THUMBS_DIR / thumb_name).write_bytes(jpeg_bytes)

        image = Image.fromarray(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))

        t0 = time.perf_counter_ns()
        embedding = self.encoder.encode_image(image)
        embed_ms = (time.perf_counter_ns() - t0) / 1_000_000

        x, y = self.projector.project(embedding)
        payload = {
            "t": round(video_ts, 2),
            "thumb": thumb_name,
            "x": round(x, 4),
            "y": round(y, 4),
        }
        point_id, upsert_us = self.store.upsert(embedding, payload)
        self.sync.enqueue(point_id, embedding.tolist(), payload)

        self.emit(
            {
                "type": "frame_ingested",
                "thumb": base64.b64encode(jpeg_bytes).decode(),
                "video_ts": round(video_ts, 2),
                "embed_ms": round(embed_ms, 1),
                "upsert_us": round(upsert_us, 1),
                "count": self.store.count,
                "bytes": self.store.bytes_on_disk(),
                "xy": [round(x, 4), round(y, 4)],
                "labels": self._recognize(embedding),
            }
        )
