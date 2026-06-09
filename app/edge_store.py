import os
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
from qdrant_edge import (
    Distance,
    EdgeConfig,
    EdgeShard,
    EdgeVectorParams,
    Mmr,
    Point,
    QueryRequest,
    UpdateOperation,
)

from .constants import (
    MMR_LAMBDA,
    MMR_MAX_CANDIDATES,
    SEARCH_LIMIT,
    VECTOR_DIMENSION,
    VECTOR_NAME,
)


class EdgeStore:
    """A Qdrant Edge shard living in a folder on this device. No server process."""

    def __init__(self, shard_dir: Path):
        self.shard_dir = shard_dir
        self.shard = None
        self.count = 0

    def initialize(self):
        if self.shard_dir.exists():
            shutil.rmtree(self.shard_dir)
        self.shard_dir.mkdir(parents=True, exist_ok=True)

        self.shard = EdgeShard.create(
            str(self.shard_dir),
            EdgeConfig(
                vectors={
                    VECTOR_NAME: EdgeVectorParams(
                        size=VECTOR_DIMENSION, distance=Distance.Cosine
                    )
                }
            ),
        )
        self.count = 0

    def upsert(self, embedding: np.ndarray, payload: dict) -> tuple[str, float]:
        """Store one vector. Returns (point_id, upsert_micros)."""
        point_id = str(uuid.uuid4())
        point = Point(
            id=point_id,
            vector={VECTOR_NAME: embedding.tolist()},
            payload=payload,
        )
        t0 = time.perf_counter_ns()
        self.shard.update(UpdateOperation.upsert_points([point]))
        micros = (time.perf_counter_ns() - t0) / 1_000
        self.count += 1
        return point_id, micros

    def search(self, query_embedding: np.ndarray, limit: int = SEARCH_LIMIT):
        """Relevance-weighted MMR search. Returns (results, search_micros)."""
        request = QueryRequest(
            query=Mmr(
                query_embedding.tolist(),
                MMR_LAMBDA,
                MMR_MAX_CANDIDATES,
                using=VECTOR_NAME,
            ),
            limit=limit,
            with_payload=True,
        )
        t0 = time.perf_counter_ns()
        results = self.shard.query(request)
        micros = (time.perf_counter_ns() - t0) / 1_000
        return results, micros

    def close(self):
        if self.shard is not None:
            self.shard.close()

    def bytes_on_disk(self) -> int:
        # Segment files are sparse; st_blocks gives bytes actually allocated.
        total = 0
        for root, _, files in os.walk(self.shard_dir):
            for f in files:
                try:
                    total += os.stat(os.path.join(root, f)).st_blocks * 512
                except OSError:
                    pass
        return total
