"""A live demo session: the Edge shard, ingest pipeline, sync worker, and the
shared query path used by both interactive commands and the auto director."""

import asyncio
import base64
import logging

import numpy as np

from .cloud_sync import CloudSync
from .constants import HUD_LABELS, LABEL_PROMPT, SHARD_DIR, THUMBS_DIR
from .edge_store import EdgeStore
from .pipeline import IngestPipeline
from .projection import MemoryMapProjector

logger = logging.getLogger(__name__)

BOOT_LINES = [
    "qdrant edge // in-process vector search engine",
    "shard path: ./edge-data/shard  (a folder, not a server)",
    "vectors: siglip2-base · 768d · cosine",
    "embedding models loaded on-device",
    "uplink: CONNECTED · cloud collection ready",
    "0 vectors in memory · patrol start",
]


class DemoSession:
    def __init__(self, encoder, emit):
        self.encoder = encoder
        self.emit = emit
        self.store = None
        self.sync = None
        self.pipeline = None
        self.projector = None
        self.ready = False

    async def start(self, boot_delay: float):
        """Reset all state, play the boot sequence, start ingesting."""
        self.ready = False

        self.store = EdgeStore(SHARD_DIR)
        self.sync = CloudSync(
            on_state=lambda q, c, up: self.emit(
                {"type": "sync", "queue": q, "cloud_count": c, "link_up": up}
            )
        )
        self.projector = MemoryMapProjector()

        self.emit({"type": "phase", "name": "boot"})

        def build_labels():
            vecs = np.array(
                [self.encoder.encode_text(LABEL_PROMPT.format(l)) for l in HUD_LABELS]
            )
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
            return vecs

        labels_task = asyncio.create_task(asyncio.to_thread(build_labels))

        await asyncio.to_thread(self.store.initialize)
        await asyncio.to_thread(self.sync.initialize)
        self.projector.load()

        for line in BOOT_LINES:
            self.emit({"type": "boot_line", "text": line})
            await asyncio.sleep(boot_delay)
        await asyncio.sleep(0.4)

        label_matrix = await labels_task

        self.pipeline = IngestPipeline(
            self.encoder, self.store, self.sync, self.projector, self.emit,
            labels=(list(HUD_LABELS), label_matrix),
        )
        self.emit({"type": "video_start"})
        self.pipeline.start()
        self.ready = True

    async def run_query(self, text: str):
        """Embed the query, search the shard, emit results. Real work, timed."""
        if not self.ready or self.store.count == 0:
            self.emit({"type": "query_result", "text": text, "latency_us": 0,
                       "offline": False, "results": []})
            return

        def search():
            query_vec = self.encoder.encode_text(text)
            return self.store.search(query_vec)

        results, micros = await asyncio.to_thread(search)

        payload_results = []
        for r in results:
            thumb_b64 = ""
            thumb_path = THUMBS_DIR / r.payload["thumb"]
            if thumb_path.exists():
                thumb_b64 = base64.b64encode(thumb_path.read_bytes()).decode()
            payload_results.append(
                {
                    "score": round(r.score, 3),
                    "thumb": thumb_b64,
                    "video_ts": r.payload["t"],
                    "xy": [r.payload["x"], r.payload["y"]],
                }
            )

        self.emit(
            {
                "type": "query_result",
                "text": text,
                "latency_us": round(micros, 1),
                "offline": not self.sync.link_up,
                "results": payload_results,
            }
        )

    async def warm_query(self):
        if self.ready:
            await asyncio.to_thread(
                lambda: self.store.search(self.encoder.encode_text("warm up"))
            )

    async def add_label(self, text: str):
        """Teach the recognizer a new concept: one text embedding, applied live."""
        if not self.ready or self.pipeline is None:
            return
        names, matrix = self.pipeline.labels
        if text in names or len(names) >= 30:
            return

        def embed():
            vec = self.encoder.encode_text(LABEL_PROMPT.format(text))
            return vec / (np.linalg.norm(vec) + 1e-9)

        vec = await asyncio.to_thread(embed)
        self.pipeline.labels = (names + [text], np.vstack([matrix, vec]))
        self.pipeline.pinned = text
        self.emit({"type": "label_added", "text": text})
        self.emit({"type": "caption",
                   "text": f"Now watching for “{text}”. No retraining, one embedding."})

    def set_link(self, up: bool):
        if self.sync is None:
            return
        self.sync.set_link(up)
        self.emit(
            {
                "type": "caption",
                "text": (
                    "Uplink restored. The shard streams its evidence to the cluster."
                    if up
                    else "Uplink lost. Watch the loop: nothing slows down, nothing is lost."
                ),
            }
        )

    def shutdown(self):
        self.ready = False
        if self.pipeline:
            self.pipeline.stop()
        if self.sync:
            self.sync.stop()
        if self.store:
            try:
                self.store.close()
            except Exception:
                pass
