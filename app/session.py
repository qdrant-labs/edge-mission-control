"""A live demo session: the Edge shard, detector, captioner, ingest pipeline,
and the shared query path used by both interactive commands and the auto
director. Everything runs in this one process; nothing leaves the device."""

import asyncio
import base64
import logging

import numpy as np

from .constants import (
    SHARD_DIR,
    THUMBS_DIR,
    WEAK_OBJECT_SCORE,
)
from .edge_store import EdgeStore
from .pipeline import IngestPipeline
from .projection import MemoryMapProjector
from .registry import ObjectRegistry

logger = logging.getLogger(__name__)

BOOT_LINES = [
    "qdrant edge // in-process vector search engine",
    "shard path: ./edge-data/shard  (a folder, not a server)",
    "dense: siglip2-base · 768d · cosine   sparse: bm25 over captions",
    "detector: yoloe-11l · open vocabulary · on-device",
    "captioner: florence-2-base · enriching every object it meets",
    "0 objects remembered · patrol start",
]


class DemoSession:
    def __init__(self, encoder, detector, captioner, emit):
        self.encoder = encoder
        self.detector = detector
        self.captioner = captioner
        self.emit = emit
        self.store = None
        self.pipeline = None
        self.registry = None
        self.projector = None
        self.ready = False

    async def start(self, boot_delay: float):
        """Reset all state, play the boot sequence, start ingesting."""
        self.ready = False

        self.store = EdgeStore(SHARD_DIR)
        self.projector = MemoryMapProjector()

        self.emit({"type": "phase", "name": "boot"})

        await asyncio.to_thread(self.store.initialize)
        await asyncio.to_thread(self.detector.reset)
        self.projector.load()

        self.registry = ObjectRegistry(
            self.encoder, self.store, self.projector, self.captioner, self.emit,
        )
        self.captioner.on_caption = self.registry.attach_caption
        if self.captioner.thread is None:
            self.captioner.start()

        for line in BOOT_LINES:
            self.emit({"type": "boot_line", "text": line})
            await asyncio.sleep(boot_delay)
        await asyncio.sleep(0.4)

        self.pipeline = IngestPipeline(
            self.encoder, self.detector, self.registry,
            self.store, self.projector, self.emit,
        )
        self.emit({"type": "video_start", "vocab": len(self.detector.vocab)})
        self.pipeline.start()
        self.ready = True

    async def run_query(self, text: str, cls: str | None = None):
        """Embed the query, hybrid-search the shard, emit results. Real work."""
        if not self.ready or self.store.count == 0:
            self.emit({"type": "query_result", "text": text, "latency_us": 0,
                       "objects": [], "moments": []})
            return

        def search():
            qvec = self.encoder.encode_text(text)
            qnorm = qvec / (np.linalg.norm(qvec) + 1e-9)
            objects, obj_us = self.store.search_objects(qvec, text, cls=cls)
            moments, mom_us = self.store.search_frames(qvec)
            return qnorm, objects, moments, obj_us + mom_us

        qnorm, objects, moments, micros = await asyncio.to_thread(search)

        object_cards = []
        for r in objects:
            score = None
            vec = r.vector.get("vision") if isinstance(r.vector, dict) else None
            if vec is not None:
                v = np.asarray(vec, dtype=np.float32)
                score = float(qnorm @ (v / (np.linalg.norm(v) + 1e-9)))
            p = r.payload
            object_cards.append({
                "obj": p.get("obj"),
                "cls": p.get("cls"),
                "caption": p.get("caption"),
                "score": round(score, 3) if score is not None else None,
                "thumb": self._thumb_b64(p.get("thumb")),
                "t_first": p.get("t_first"),
                "t_last": p.get("t_last"),
                "box": p.get("box"),
                "xy": [p.get("x"), p.get("y")],
                "sightings": p.get("sightings"),
                "weak": score is not None and score < WEAK_OBJECT_SCORE,
            })

        moment_cards = [
            {
                "score": round(r.score, 3),
                "thumb": self._thumb_b64(r.payload.get("thumb")),
                "video_ts": r.payload.get("t"),
                "xy": [r.payload.get("x"), r.payload.get("y")],
            }
            for r in moments
        ]

        self.emit({
            "type": "query_result",
            "text": text,
            "cls": cls,
            "latency_us": round(micros, 1),
            "objects": object_cards,
            "moments": moment_cards,
        })

    @staticmethod
    def _thumb_b64(name) -> str:
        if not name:
            return ""
        path = THUMBS_DIR / name
        if not path.exists():
            return ""
        return base64.b64encode(path.read_bytes()).decode()

    async def warm_query(self):
        if self.ready:
            def warm():
                qvec = self.encoder.encode_text("warm up")
                self.store.search_objects(qvec, "warm up")
                self.store.search_frames(qvec)
            await asyncio.to_thread(warm)

    async def teach(self, text: str):
        """Teach the detector a new concept: one text embedding, applied live."""
        if not self.ready:
            return
        added = self.detector.teach(text)
        if not added:
            return
        self.emit({"type": "label_added", "text": text})

    def shutdown(self):
        self.ready = False
        if self.pipeline:
            self.pipeline.stop()
        if self.captioner:
            self.captioner.on_caption = None
        if self.store:
            try:
                self.store.close()
            except Exception:
                pass
