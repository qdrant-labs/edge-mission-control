"""Object identities: turns raw track ids into a persistent object memory.

A track must be sighted CONFIRM_SIGHTINGS times before it becomes an object.
On confirmation the registry picks the best crop seen so far (confidence ×
size × sharpness), embeds it with the cross-modal encoder, and either merges
it into a known object of the same class (cosine re-id, handles camera cuts)
or mints a new object: one point in the Edge shard, one caption job, one
"object discovered" event.
"""

import base64
import logging
import threading
import time
import uuid

import cv2
import numpy as np
from PIL import Image

from .constants import (
    CONFIRM_SIGHTINGS,
    CROP_PAD,
    CROP_THUMB_WIDTH,
    JPEG_QUALITY,
    MAX_CROP_EMBEDS_PER_TICK,
    REID_THRESHOLD,
    THUMBS_DIR,
)

logger = logging.getLogger(__name__)


class _Track:
    __slots__ = ("cls", "sightings", "best_score", "best_crop", "obj_id", "last_seen")

    def __init__(self, cls):
        self.cls = cls
        self.sightings = 0
        self.best_score = 0.0
        self.best_crop = None  # PIL image, padded
        self.obj_id = None
        self.last_seen = 0.0


class ObjectRecord:
    __slots__ = ("obj_id", "point_id", "label", "cls", "embedding", "t_first",
                 "t_last", "sightings", "caption", "thumb_name", "box", "xy",
                 "dirty")

    def __init__(self, obj_id, point_id, label, cls):
        self.obj_id = obj_id
        self.point_id = point_id
        self.label = label
        self.cls = cls
        self.embedding = None
        self.t_first = 0.0
        self.t_last = 0.0
        self.sightings = 0
        self.caption = None
        self.thumb_name = None
        self.box = None
        self.xy = (0.5, 0.5)
        self.dirty = False

    def payload(self) -> dict:
        p = {
            "kind": "object",
            "obj": self.label,
            "cls": self.cls,
            "t_first": round(self.t_first, 2),
            "t_last": round(self.t_last, 2),
            "t": round(self.t_first, 2),
            "sightings": self.sightings,
            "thumb": self.thumb_name,
            "box": [round(v, 4) for v in self.box],
            "x": round(self.xy[0], 4),
            "y": round(self.xy[1], 4),
        }
        if self.caption:
            p["caption"] = self.caption
        return p


def crop_quality(frame_bgr, box, conf) -> float:
    """Bigger, sharper, more confident crops make better object portraits."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return 0.0
    region = frame_bgr[y1:y2, x1:x2]
    small = cv2.resize(region, (96, 96), interpolation=cv2.INTER_AREA)
    sharp = cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var()
    area = (box[2] - box[0]) * (box[3] - box[1])
    return conf * (area ** 0.5) * min(sharp, 1200.0)


def padded_crop(frame_bgr, box) -> Image.Image:
    """Crop with a context margin: captions and embeddings ground better."""
    h, w = frame_bgr.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    x1 = max(0, int((box[0] - bw * CROP_PAD) * w))
    y1 = max(0, int((box[1] - bh * CROP_PAD) * h))
    x2 = min(w, int((box[2] + bw * CROP_PAD) * w))
    y2 = min(h, int((box[3] + bh * CROP_PAD) * h))
    return Image.fromarray(cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))


class ObjectRegistry:
    """Lives on the ingest thread; attach_caption arrives from the captioner."""

    def __init__(self, encoder, store, projector, captioner, emit):
        self.encoder = encoder
        self.store = store
        self.projector = projector
        self.captioner = captioner
        self.emit = emit
        self.tracks: dict[int, _Track] = {}
        self.objects: dict[str, ObjectRecord] = {}
        self._by_class: dict[str, list[str]] = {}
        self._lock = threading.Lock()
        self._counter = 0
        self._last_flush = 0.0

    @property
    def object_count(self) -> int:
        return len(self.objects)

    def observe(self, detections, frame_bgr, video_ts: float) -> float:
        """One detection tick. Returns crop-embedding milliseconds spent."""
        embed_budget = MAX_CROP_EMBEDS_PER_TICK
        embed_ms = 0.0

        for det in detections:
            track = self.tracks.get(det.track_id)
            if track is None:
                track = self.tracks[det.track_id] = _Track(det.cls)
            track.sightings += 1
            track.last_seen = video_ts

            score = crop_quality(frame_bgr, det.box, det.conf)
            if score > track.best_score:
                track.best_score = score
                track.best_crop = padded_crop(frame_bgr, det.box)

            if track.obj_id is not None:
                rec = self.objects[track.obj_id]
                rec.t_last = video_ts
                rec.sightings += 1
                rec.box = det.box
                rec.dirty = True
            elif track.sightings >= CONFIRM_SIGHTINGS and embed_budget > 0 \
                    and track.best_crop is not None:
                embed_budget -= 1
                t0 = time.perf_counter()
                self._confirm(track, det, video_ts)
                embed_ms += (time.perf_counter() - t0) * 1000

        self._expire_tracks(video_ts)
        self._flush_dirty(video_ts)
        return embed_ms

    def _confirm(self, track: _Track, det, video_ts: float):
        embedding = self.encoder.encode_image(track.best_crop)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-9)

        # Re-id: the same physical object seen again (new track / camera cut).
        with self._lock:
            for obj_id in self._by_class.get(track.cls, []):
                rec = self.objects[obj_id]
                if rec.embedding is not None and float(rec.embedding @ embedding) >= REID_THRESHOLD:
                    track.obj_id = obj_id
                    rec.t_last = video_ts
                    rec.sightings += track.sightings
                    rec.dirty = True
                    return

            self._counter += 1
            label = f"OBJ-{self._counter:03d}"
            rec = ObjectRecord(label, str(uuid.uuid4()), label, track.cls)
            rec.embedding = embedding
            rec.t_first = rec.t_last = video_ts
            rec.sightings = track.sightings
            rec.box = det.box
            rec.xy = self.projector.project(embedding)
            rec.thumb_name = f"o{self._counter:04d}.jpg"
            track.obj_id = label
            self.objects[label] = rec
            self._by_class.setdefault(track.cls, []).append(label)

        thumb_b64 = self._write_thumb(track.best_crop, rec.thumb_name)
        _, upsert_us = self.store.upsert_object(rec.point_id, embedding, rec.payload())
        self.captioner.submit(rec.obj_id, track.best_crop)

        self.emit({
            "type": "object_discovered",
            "obj": rec.label,
            "cls": rec.cls,
            "thumb": thumb_b64,
            "t": round(video_ts, 2),
            "xy": [round(rec.xy[0], 4), round(rec.xy[1], 4)],
            "upsert_us": round(upsert_us, 1),
            "total": len(self.objects),
        })

    def attach_caption(self, obj_id: str, caption: str):
        """Caption arrived from the enrichment worker: store it and index it."""
        with self._lock:
            rec = self.objects.get(obj_id)
            if rec is None:
                return
            rec.caption = caption
        self.store.set_caption(rec.point_id, caption, f"{rec.cls}. {caption}")
        self.emit({"type": "object_enriched", "obj": obj_id, "caption": caption})

    def overlay(self, detections) -> list:
        """Compact per-frame overlay: tracked boxes with object labels."""
        out = []
        for det in detections:
            track = self.tracks.get(det.track_id)
            obj_id = track.obj_id if track else None
            out.append({
                "tid": det.track_id,
                "obj": obj_id,
                "cls": det.cls,
                "conf": round(det.conf, 2),
                "box": [round(v, 4) for v in det.box],
            })
        return out

    def inventory(self) -> dict:
        counts = self.store.class_facets()
        return {"type": "inventory", "total": len(self.objects), "classes": counts}

    def _expire_tracks(self, video_ts: float):
        stale = [tid for tid, tr in self.tracks.items() if video_ts - tr.last_seen > 8.0]
        for tid in stale:
            del self.tracks[tid]

    def _flush_dirty(self, video_ts: float):
        """Batch t_last/sightings updates: cheap set_payload every ~2 s."""
        if video_ts - self._last_flush < 2.0:
            return
        self._last_flush = video_ts
        with self._lock:
            dirty = [r for r in self.objects.values() if r.dirty]
            for rec in dirty:
                rec.dirty = False
        for rec in dirty:
            self.store.update_payload(rec.point_id, {
                "t_last": round(rec.t_last, 2),
                "sightings": rec.sightings,
                "box": [round(v, 4) for v in rec.box],
            })

    def _write_thumb(self, crop: Image.Image, name: str) -> str:
        thumb = crop.copy()
        if thumb.width > CROP_THUMB_WIDTH:
            thumb = thumb.resize(
                (CROP_THUMB_WIDTH, max(1, int(thumb.height * CROP_THUMB_WIDTH / thumb.width))))
        arr = cv2.cvtColor(np.asarray(thumb), cv2.COLOR_RGB2BGR)
        ok, jpeg = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            return ""
        (THUMBS_DIR / name).write_bytes(jpeg.tobytes())
        return base64.b64encode(jpeg.tobytes()).decode()
