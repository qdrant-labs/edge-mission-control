import logging

import numpy as np
from fastembed import ImageEmbedding, TextEmbedding

from .constants import ENCODER_BACKEND, TEXT_MODEL_NAME, VISION_MODEL_NAME

logger = logging.getLogger(__name__)


def get_encoder():
    """The demo's encoder, selected by ENCODER_BACKEND in constants.py."""
    if ENCODER_BACKEND == "siglip2":
        from .siglip_encoder import SiglipEncoder

        return SiglipEncoder()
    return CrossModalEncoder()


class CrossModalEncoder:
    """CLIP text + image encoders running fully on-device via fastembed/ONNX."""

    def __init__(self):
        self.image_model = None
        self.text_model = None

    def load(self):
        if self.image_model is None:
            logger.info("Loading vision model %s", VISION_MODEL_NAME)
            self.image_model = ImageEmbedding(model_name=VISION_MODEL_NAME)
        if self.text_model is None:
            logger.info("Loading text model %s", TEXT_MODEL_NAME)
            self.text_model = TextEmbedding(model_name=TEXT_MODEL_NAME)

    def warm(self):
        """Run one dummy inference per model so first real call has no JIT cost."""
        from PIL import Image

        self.load()
        dummy = Image.new("RGB", (224, 224))
        list(self.image_model.embed([dummy]))
        list(self.text_model.embed(["warm up"]))

    def encode_image(self, image) -> np.ndarray:
        return next(iter(self.image_model.embed([image])))

    def encode_text(self, text: str) -> np.ndarray:
        return next(iter(self.text_model.embed([text])))
