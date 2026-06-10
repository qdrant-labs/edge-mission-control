"""Real-time loop: capture -> detect+track -> embed -> upsert, all on-device.

Every tick (INGEST_FPS per second of mission time) the pipeline:
  1. runs open-vocabulary detection + tracking on the frame,
  2. embeds the whole frame as a scene memory,
  3. lets the registry confirm new object identities (crop embeddings),
  4. emits one event with the overlay boxes and loop timings.

Paced against the wall clock so it stays aligned with the video element
playing the same file in the browser.
"""

import base64
import logging
import threading
import time
import uuid

import cv2
from PIL import Image

from .constants import (
    INGEST_FPS,
    JPEG_QUALITY,
    MISSION_VIDEO,
    THUMB_WIDTH,
    THUMBS_DIR,
)

logger = logging.getLogger(__name__)


class IngestPipeline:
    def __init__(self, encoder, detector, registry, store, projector, emit):
        self.encoder = encoder
        self.detector = detector
        self.registry = registry
        self.store = store
        self.projector = projector
        self.emit = emit  # callback(event_dict), thread-safe
        self.is_running = False
        self.thread = None
        self._last_inventory = 0.0

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
        interval = max(1, round(video_fps / INGEST_FPS))

        # Adaptive pacing: the browser plays the same file in real time, so
        # the loop targets whatever frame is on screen *now*. If a tick runs
        # long, the next tick skips ahead instead of drifting behind the feed.
        start = time.monotonic()
        pos = 0          # frames consumed from the decoder so far
        completed = False

        while self.is_running:
            elapsed = time.monotonic() - start
            due = int(elapsed * video_fps)
            next_slot = max(pos, due)
            if next_slot % interval:
                next_slot = (next_slot // interval + 1) * interval

            delay = (start + next_slot / video_fps) - time.monotonic()
            if delay > 0:
                time.sleep(delay)

            while pos <= next_slot:
                if not cap.grab():
                    completed = True
                    break
                pos += 1
            if completed:
                break

            ok, frame = cap.retrieve()
            if ok:
                self._ingest(frame, next_slot / video_fps)

        cap.release()
        logger.info("Pipeline finished: %d frame memories, %d objects",
                    self.store.count, self.registry.object_count)
        if completed:
            # Natural end of the mission (not an external stop).
            self.emit({
                "type": "mission_complete",
                "count": self.store.count,
                "objects": self.registry.object_count,
            })

    def _ingest(self, frame, video_ts: float):
        # 1. Detect + track every object in view.
        detections, detect_ms = self.detector.track(frame)

        # 2. The whole frame is a scene memory (kind=frame).
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
        point_id = str(uuid.uuid4())
        _, upsert_us = self.store.upsert_frame(point_id, embedding, payload)

        # 3. Object identities: confirm tracks, embed crops, caption async.
        crop_embed_ms = self.registry.observe(detections, frame, video_ts)

        self.emit({
            "type": "frame_ingested",
            "thumb": base64.b64encode(jpeg_bytes).decode(),
            "video_ts": round(video_ts, 2),
            "detect_ms": round(detect_ms, 1),
            "embed_ms": round(embed_ms + crop_embed_ms, 1),
            "upsert_us": round(upsert_us, 1),
            "count": self.store.count,
            "objects": self.registry.object_count,
            "bytes": self.store.bytes_on_disk(),
            "xy": [round(x, 4), round(y, 4)],
            "boxes": self.registry.overlay(detections),
        })

        # 4. Inventory facets, throttled to one per second.
        if video_ts - self._last_inventory >= 1.0:
            self._last_inventory = video_ts
            self.emit(self.registry.inventory())
