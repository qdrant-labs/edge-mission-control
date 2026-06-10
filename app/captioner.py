"""Async caption enrichment: Florence-2-base describing each discovered object.

Runs on its own thread so the detection loop never waits for language. Every
confirmed object's best crop gets a short grounded caption (~0.2 s on Apple
Silicon). The result lands back in the Edge shard as payload plus an
on-device BM25 sparse vector, which is what makes hybrid text search work.
"""

import logging
import queue
import threading

from .constants import CAPTION_MAX_TOKENS, CAPTIONER_DEVICE, CAPTIONER_MODEL

logger = logging.getLogger(__name__)


class Captioner:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        # LIFO: the most recently discovered object gets captioned first, so
        # fresh discoveries become caption-searchable while the trail of older
        # ones drains in the gaps between rooms.
        self.jobs: queue.LifoQueue = queue.LifoQueue()
        self.on_caption = None  # set by the session: callback(obj_id, caption)
        self.is_running = False
        self.thread = None

    def load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoProcessor, Florence2ForConditionalGeneration

        self.device = CAPTIONER_DEVICE if torch.backends.mps.is_available() else "cpu"
        logger.info("Loading %s on %s", CAPTIONER_MODEL, self.device)
        self.processor = AutoProcessor.from_pretrained(CAPTIONER_MODEL)
        self.model = Florence2ForConditionalGeneration.from_pretrained(
            CAPTIONER_MODEL, dtype=torch.float32
        ).to(self.device)

    def warm(self):
        self.load()
        from PIL import Image

        self.caption(Image.new("RGB", (224, 224), (40, 40, 40)))

    def caption(self, image) -> str:
        task = "<CAPTION>"
        if image.width < 224:
            image = image.resize((224, max(1, int(224 * image.height / image.width))))
        inputs = self.processor(text=task, images=image, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs, max_new_tokens=CAPTION_MAX_TOKENS, num_beams=1, do_sample=False
        )
        decoded = self.processor.batch_decode(out, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            decoded, task=task, image_size=image.size
        )
        return parsed[task].strip().rstrip(".")

    def submit(self, obj_id: str, crop):
        self.jobs.put((obj_id, crop))

    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_running = False
        self.jobs.put(None)

    def pending(self) -> int:
        return self.jobs.qsize()

    def _worker(self):
        while self.is_running:
            job = self.jobs.get()
            if job is None:
                continue
            obj_id, crop = job
            try:
                text = self.caption(crop)
            except Exception:
                logger.exception("Caption failed for %s", obj_id)
                continue
            if self.on_caption is not None:
                try:
                    self.on_caption(obj_id, text)
                except Exception:
                    logger.exception("Caption callback failed for %s", obj_id)
