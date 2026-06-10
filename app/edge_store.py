"""A Qdrant Edge shard living in a folder on this device. No server process.

Two vector spaces in one shard:
  - dense  "vision":  SigLIP2 embeddings of frames and object crops
  - sparse "caption": on-device BM25 over Florence-2 object captions

Object search is a hybrid query — dense and sparse prefetches fused with
reciprocal rank fusion — executed entirely in-process.
"""

import os
import shutil
import threading
import time
from pathlib import Path

import numpy as np
from qdrant_edge import (
    Bm25,
    Bm25Config,
    Distance,
    EdgeConfig,
    EdgeShard,
    EdgeSparseVectorParams,
    EdgeVectorParams,
    FacetRequest,
    FieldCondition,
    Filter,
    Fusion,
    MatchValue,
    Mmr,
    PayloadSchemaType,
    Point,
    PointVectors,
    Prefetch,
    Query,
    QueryRequest,
    UpdateOperation,
)

from .constants import (
    MMR_LAMBDA,
    MMR_MAX_CANDIDATES,
    MOMENT_SEARCH_LIMIT,
    OBJECT_SEARCH_LIMIT,
    RRF_K,
    SPARSE_VECTOR_NAME,
    VECTOR_DIMENSION,
    VECTOR_NAME,
)


def _kind_filter(kind: str, cls: str | None = None) -> Filter:
    must = [FieldCondition(key="kind", match=MatchValue(kind))]
    if cls:
        must.append(FieldCondition(key="cls", match=MatchValue(cls)))
    return Filter(must=must)


class EdgeStore:
    def __init__(self, shard_dir: Path):
        self.shard_dir = shard_dir
        self.shard = None
        self.bm25 = Bm25(Bm25Config())
        self.count = 0          # frame memories
        self.object_count = 0   # object memories
        self._write_lock = threading.Lock()
        self._disk_cache = (0.0, 0)  # (measured_at, bytes)

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
                },
                sparse_vectors={SPARSE_VECTOR_NAME: EdgeSparseVectorParams()},
            ),
        )
        with self._write_lock:
            self.shard.update(UpdateOperation.create_field_index("kind", PayloadSchemaType.Keyword))
            self.shard.update(UpdateOperation.create_field_index("cls", PayloadSchemaType.Keyword))
        self.count = 0
        self.object_count = 0

    # ------------------------------------------------------------------ write
    def upsert_frame(self, point_id: str, embedding: np.ndarray, payload: dict):
        payload = {"kind": "frame", **payload}
        micros = self._upsert(point_id, {VECTOR_NAME: embedding.tolist()}, payload)
        self.count += 1
        return point_id, micros

    def upsert_object(self, point_id: str, embedding: np.ndarray, payload: dict):
        micros = self._upsert(point_id, {VECTOR_NAME: embedding.tolist()}, payload)
        self.object_count += 1
        return point_id, micros

    def _upsert(self, point_id, vectors, payload) -> float:
        point = Point(id=point_id, vector=vectors, payload=payload)
        t0 = time.perf_counter_ns()
        with self._write_lock:
            self.shard.update(UpdateOperation.upsert_points([point]))
        return (time.perf_counter_ns() - t0) / 1_000

    def set_caption(self, point_id: str, caption: str, bm25_doc: str):
        """Attach the enrichment result: payload text + sparse BM25 vector."""
        sparse = self.bm25.embed_document(bm25_doc)
        with self._write_lock:
            self.shard.update(UpdateOperation.set_payload([point_id], {"caption": caption}))
            self.shard.update(
                UpdateOperation.update_vectors([PointVectors(point_id, {SPARSE_VECTOR_NAME: sparse})])
            )

    def update_payload(self, point_id: str, patch: dict):
        with self._write_lock:
            self.shard.update(UpdateOperation.set_payload([point_id], patch))

    # ------------------------------------------------------------------- read
    def search_objects(self, query_embedding: np.ndarray, text: str,
                       cls: str | None = None, limit: int = OBJECT_SEARCH_LIMIT):
        """Hybrid dense+BM25 search over object memories, fused on-device.

        Returns (results, micros). Each result keeps its dense vector so the
        caller can attach a true cosine relevance score to the fused ranking.
        """
        dense = query_embedding.tolist()
        sparse = self.bm25.embed_query(text)
        flt = _kind_filter("object", cls)
        request = QueryRequest(
            prefetches=[
                Prefetch(query=Query.Nearest(dense, using=VECTOR_NAME),
                         filter=flt, limit=limit * 6),
                Prefetch(query=Query.Nearest(sparse, using=SPARSE_VECTOR_NAME),
                         filter=flt, limit=limit * 6),
            ],
            query=Fusion.Rrf(RRF_K),
            limit=limit,
            with_payload=True,
            with_vector=[VECTOR_NAME],
        )
        t0 = time.perf_counter_ns()
        results = self.shard.query(request)
        micros = (time.perf_counter_ns() - t0) / 1_000
        return results, micros

    def search_frames(self, query_embedding: np.ndarray, limit: int = MOMENT_SEARCH_LIMIT):
        """Relevance-weighted MMR over frame memories: diverse moments."""
        request = QueryRequest(
            query=Mmr(query_embedding.tolist(), MMR_LAMBDA, MMR_MAX_CANDIDATES,
                      using=VECTOR_NAME),
            filter=_kind_filter("frame"),
            limit=limit,
            with_payload=True,
        )
        t0 = time.perf_counter_ns()
        results = self.shard.query(request)
        micros = (time.perf_counter_ns() - t0) / 1_000
        return results, micros

    def class_facets(self, limit: int = 24) -> list:
        facets = self.shard.facet(FacetRequest(key="cls", limit=limit,
                                               filter=_kind_filter("object")))
        return [[h.value, h.count] for h in facets.hits]

    def close(self):
        if self.shard is not None:
            self.shard.close()

    def bytes_on_disk(self) -> int:
        # Segment files are sparse; st_blocks gives bytes actually allocated.
        # Walking the shard dir every ingest tick adds up: cache for a second.
        measured_at, cached = self._disk_cache
        now = time.monotonic()
        if now - measured_at < 1.0:
            return cached
        total = 0
        for root, _, files in os.walk(self.shard_dir):
            for f in files:
                try:
                    total += os.stat(os.path.join(root, f)).st_blocks * 512
                except OSError:
                    pass
        self._disk_cache = (now, total)
        return total
