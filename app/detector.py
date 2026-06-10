"""Open-vocabulary object detection + tracking on-device.

YOLOE-11L-seg in text-prompt mode: the class vocabulary is plain English,
embedded once with MobileCLIP at startup. Tracking (BoT-SORT) gives every
detection a persistent track id so the registry can build object identities.
The vocabulary can be extended live in ~300 ms — that is the "teach a
concept" feature.
"""

import logging
import os
import threading
import time

os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # no pip calls at runtime

from .constants import (
    DETECT_CONF,
    DETECT_IMGSZ,
    DETECT_MAX_DET,
    DETECTOR_VOCAB,
    DETECTOR_WEIGHTS,
    MAX_VOCAB,
    MIN_BOX_AREA,
    PROJECT_ROOT,
)

logger = logging.getLogger(__name__)


class Detection:
    __slots__ = ("track_id", "cls", "conf", "box")

    def __init__(self, track_id: int, cls: str, conf: float, box: tuple):
        self.track_id = track_id
        self.cls = cls
        self.conf = conf
        self.box = box  # (x1, y1, x2, y2) normalized to [0, 1]


class ObjectDetector:
    """YOLOE wrapper: load once, track frames, grow the vocabulary live."""

    def __init__(self):
        self.model = None
        self.device = None
        self.vocab = list(DETECTOR_VOCAB)
        self._pending_terms = []
        self._lock = threading.Lock()

    def load(self):
        if self.model is not None:
            return
        import torch
        from ultralytics import YOLO

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info("Loading %s on %s", DETECTOR_WEIGHTS, self.device)
        # Weights download next to the repo root on first run (gitignored).
        weights = PROJECT_ROOT / DETECTOR_WEIGHTS
        self.model = YOLO(str(weights) if weights.exists() else DETECTOR_WEIGHTS)
        self._apply_vocab(self.vocab)

    def warm(self):
        self.load()
        import numpy as np

        dummy = np.zeros((360, 640, 3), dtype=np.uint8)
        self.model.predict(dummy, device=self.device, verbose=False, imgsz=DETECT_IMGSZ)

    def _apply_vocab(self, vocab):
        # MobileCLIP's TorchScript text tower cannot load on MPS (float64
        # weights), so embed the vocabulary with the model on CPU, then move
        # everything (including the new class embeddings) back to the GPU.
        import torch

        nn_model = self.model.model
        on_gpu = next(nn_model.parameters()).device.type != "cpu"
        if on_gpu:
            nn_model.to("cpu")
        with torch.inference_mode():
            self.model.set_classes(vocab, self.model.get_text_pe(vocab))
        if on_gpu:
            nn_model.to(self.device)
        self.vocab = vocab

    def reset(self):
        """Fresh tracker state for a new session (track ids keep counting)."""
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            tracker.reset()

    def teach(self, term: str) -> bool:
        """Queue a new concept; applied between frames on the tracking thread."""
        term = term.lower().strip()
        with self._lock:
            if term in self.vocab or term in self._pending_terms:
                return False
            if len(self.vocab) + len(self._pending_terms) >= MAX_VOCAB:
                return False
            self._pending_terms.append(term)
        return True

    def track(self, frame_bgr) -> tuple[list[Detection], float]:
        """Detect + track one frame. Returns (detections, detect_ms)."""
        with self._lock:
            pending, self._pending_terms = self._pending_terms, []
        if pending:
            t0 = time.perf_counter()
            self._apply_vocab(self.vocab + pending)
            logger.info("Vocabulary extended with %s in %.0f ms",
                        pending, (time.perf_counter() - t0) * 1000)

        h, w = frame_bgr.shape[:2]
        t0 = time.perf_counter_ns()
        result = self.model.track(
            frame_bgr,
            device=self.device,
            conf=DETECT_CONF,
            imgsz=DETECT_IMGSZ,
            max_det=DETECT_MAX_DET,
            persist=True,
            verbose=False,
        )[0]
        detect_ms = (time.perf_counter_ns() - t0) / 1e6

        detections = []
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            for tid, c, cf, xyxy in zip(boxes.id, boxes.cls, boxes.conf, boxes.xyxy):
                x1, y1, x2, y2 = (float(v) for v in xyxy)
                box = (x1 / w, y1 / h, x2 / w, y2 / h)
                if (box[2] - box[0]) * (box[3] - box[1]) < MIN_BOX_AREA:
                    continue
                detections.append(
                    Detection(int(tid), result.names[int(c)], float(cf), box)
                )
        return detections, detect_ms
