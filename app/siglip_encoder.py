"""SigLIP2-base cross-modal encoder running on ONNX Runtime.

Same interface as encoder.CrossModalEncoder, substantially better zero-shot
retrieval than 2021-era CLIP. Uses the official onnx-community export; no
torch, no new dependencies (onnxruntime, tokenizers, and huggingface_hub all
ship with fastembed).
"""

import json
import logging

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)

REPO = "onnx-community/siglip2-base-patch16-224-ONNX"
TEXT_MAX_LEN = 64


class SiglipEncoder:
    """CLIP-compatible dual encoder: encode_image / encode_text -> 768-d."""

    def __init__(self):
        self.vision = None
        self.text = None
        self.tokenizer = None
        self.mean = None
        self.std = None

    def load(self):
        if self.vision is not None:
            return
        logger.info("Loading SigLIP2 ONNX towers from %s", REPO)
        vision_path = hf_hub_download(REPO, "onnx/vision_model.onnx")
        text_path = hf_hub_download(REPO, "onnx/text_model.onnx")
        tok_path = hf_hub_download(REPO, "tokenizer.json")
        pp_path = hf_hub_download(REPO, "preprocessor_config.json")

        self.vision = ort.InferenceSession(vision_path)
        self.text = ort.InferenceSession(text_path)

        pp = json.load(open(pp_path))
        self.mean = np.array(pp.get("image_mean", [0.5] * 3), dtype=np.float32)
        self.std = np.array(pp.get("image_std", [0.5] * 3), dtype=np.float32)

        self.tokenizer = Tokenizer.from_file(tok_path)
        pad_id = self.tokenizer.token_to_id("<pad>") or 0
        self.tokenizer.enable_padding(length=TEXT_MAX_LEN, pad_id=pad_id)
        self.tokenizer.enable_truncation(max_length=TEXT_MAX_LEN)

    def warm(self):
        self.load()
        self.encode_image(Image.new("RGB", (224, 224)))
        self.encode_text("warm up")

    def encode_image(self, image) -> np.ndarray:
        if isinstance(image, str):
            image = Image.open(image)
        a = np.asarray(image.convert("RGB").resize((224, 224), Image.BICUBIC),
                       dtype=np.float32) / 255.0
        a = ((a - self.mean) / self.std).transpose(2, 0, 1)[None]
        out = self.vision.run(None, {self.vision.get_inputs()[0].name: a})
        return self._pooled(out)

    def encode_text(self, text: str) -> np.ndarray:
        # SigLIP was trained on lowercased text.
        ids = np.array([self.tokenizer.encode(text.lower()).ids], dtype=np.int64)
        out = self.text.run(None, {self.text.get_inputs()[0].name: ids})
        return self._pooled(out)

    @staticmethod
    def _pooled(out) -> np.ndarray:
        # Exports return (last_hidden_state, pooler_output); take the pooled vector.
        v = out[1][0] if len(out) > 1 and out[1].ndim == 2 else out[0][0]
        return v.astype(np.float32)
