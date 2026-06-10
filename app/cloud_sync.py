import logging
import queue
import threading
import time

from qdrant_client import QdrantClient
from qdrant_client import models as rest_models

from .constants import (
    CLOUD_COLLECTION,
    CLOUD_URL,
    SPARSE_VECTOR_NAME,
    SYNC_BATCH_SIZE,
    SYNC_INTERVAL,
    VECTOR_DIMENSION,
    VECTOR_NAME,
)

logger = logging.getLogger(__name__)


class CloudSync:
    """Background sync from the edge shard to a Qdrant cluster.

    The link_up flag is the demo's kill switch: when False, points queue up
    on-device exactly as they would during a real connectivity loss.
    """

    def __init__(self, on_state):
        self.client = QdrantClient(url=CLOUD_URL, check_compatibility=False)
        self.upload_queue: queue.Queue = queue.Queue()
        self.link_up = True
        self.cloud_count = 0
        # Object points re-sync when their caption lands; count unique ids so
        # the cloud tally matches the collection, not the upload volume.
        self._synced_ids: set = set()
        self.is_running = False
        self.on_state = on_state  # callback(queue_size, cloud_count, link_up)
        self.thread = None

    def initialize(self):
        if self.client.collection_exists(CLOUD_COLLECTION):
            self.client.delete_collection(CLOUD_COLLECTION)
        self.client.create_collection(
            collection_name=CLOUD_COLLECTION,
            vectors_config={
                VECTOR_NAME: rest_models.VectorParams(
                    size=VECTOR_DIMENSION,
                    distance=rest_models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: rest_models.SparseVectorParams()
            },
        )
        self.cloud_count = 0
        self.is_running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def enqueue(self, point_id: str, vector: list, payload: dict):
        self.upload_queue.put(
            rest_models.PointStruct(
                id=point_id, vector={VECTOR_NAME: vector}, payload=payload
            )
        )
        self._emit()

    def set_link(self, up: bool):
        self.link_up = up
        self._emit()

    def _emit(self):
        self.on_state(self.upload_queue.qsize(), self.cloud_count, self.link_up)

    def _worker(self):
        while self.is_running:
            if not self.link_up:
                time.sleep(0.2)
                continue

            batch = []
            while len(batch) < SYNC_BATCH_SIZE:
                try:
                    batch.append(self.upload_queue.get_nowait())
                except queue.Empty:
                    break

            if not batch:
                time.sleep(SYNC_INTERVAL)
                continue

            try:
                self.client.upsert(collection_name=CLOUD_COLLECTION, points=batch)
                self._synced_ids.update(p.id for p in batch)
                self.cloud_count = len(self._synced_ids)
            except Exception as e:
                logger.warning("Sync failed: %s", e)
                for p in batch:
                    self.upload_queue.put(p)
                time.sleep(1.0)
            self._emit()
            time.sleep(SYNC_INTERVAL)

    def stop(self):
        self.is_running = False
