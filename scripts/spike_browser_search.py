"""Validation spike: does an in-browser SigLIP2 text encoder return the right
objects? We load each quantized text ONNX the browser would use, embed the
scripted queries, and rank object crops by pure numpy cosine (mirroring the JS
path -- NOT shard.query) against the prebuilt shard's object vectors.

PASS == the quantized model returns the same right objects as fp32.
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

import qdrant_edge as q
from qdrant_edge import FieldCondition, Filter, MatchValue
from app.constants import SHARD_DIR, VECTOR_NAME

REPO = "onnx-community/siglip2-base-patch16-224-ONNX"
TEXT_MAX_LEN = 64

# (query, accepted classes, accepted caption keywords)  -- from test_objects.py
CANDIDATES = [
    ("a leather lounge chair", {"armchair", "sofa", "dining chair"}, ("chair",)),
    ("wine bottles on a tray", {"wine bottle", "wine glass", "tray"}, ("wine",)),
    ("a pool table", {"pool table"}, ("pool table",)),
    ("bar stools at a kitchen counter", {"bar stool", "stool", "kitchen island"}, ("stool",)),
    ("a bed with pillows", {"bed", "pillow", "blanket"}, ("bed",)),
    ("a white bathtub", {"bathtub"}, ("bathtub", "tub")),
    ("a potted plant", {"houseplant"}, ("plant",)),
]


def load_text_session(variant):
    fname = "onnx/text_model.onnx" if variant == "fp32" else f"onnx/text_model_{variant}.onnx"
    path = hf_hub_download(REPO, fname)
    # int8/uint8 exports sometimes carry external data; hf_hub handles siblings lazily.
    return ort.InferenceSession(path)


def make_tokenizer():
    tok_path = hf_hub_download(REPO, "tokenizer.json")
    tok = Tokenizer.from_file(tok_path)
    pad_id = tok.token_to_id("<pad>") or 0
    tok.enable_padding(length=TEXT_MAX_LEN, pad_id=pad_id)
    tok.enable_truncation(max_length=TEXT_MAX_LEN)
    return tok


def encode_text(sess, tok, text):
    ids = np.array([tok.encode(text.lower()).ids], dtype=np.int64)
    out = sess.run(None, {sess.get_inputs()[0].name: ids})
    v = out[1][0] if len(out) > 1 and out[1].ndim == 2 else out[0][0]
    v = v.astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def main():
    # Pull object vectors + payloads straight out of the shard (this is exactly
    # the data the static site will ship as JSON).
    shard = q.EdgeShard.load(str(SHARD_DIR))
    res = shard.scroll(q.ScrollRequest(
        limit=10000,
        filter=Filter(must=[FieldCondition(key="kind", match=MatchValue("object"))]),
        with_payload=True,
        with_vector=[VECTOR_NAME],
    ))
    points = res[0] if isinstance(res, tuple) else res
    objs = []
    for p in points:
        vec = p.vector.get(VECTOR_NAME) if isinstance(p.vector, dict) else None
        if vec is None:
            continue
        v = np.asarray(vec, dtype=np.float32)
        objs.append({
            "cls": p.payload.get("cls"),
            "caption": p.payload.get("caption") or "",
            "t": p.payload.get("t_first"),
            "v": v / (np.linalg.norm(v) + 1e-9),
        })
    shard.close()
    if not objs:
        print("NO OBJECTS in shard -- run the pipeline first (make prepare / bake).")
        sys.exit(2)
    mat = np.stack([o["v"] for o in objs])
    print(f"objects in shard: {len(objs)}")

    tok = make_tokenizer()
    variants = ["fp32", "fp16", "int8", "quantized"]
    summary = {}
    for variant in variants:
        try:
            sess = load_text_session(variant)
        except Exception as e:
            print(f"\n=== {variant}: COULD NOT LOAD ({repr(e)[:80]}) ===")
            continue
        print(f"\n=== text_model {variant} ===")
        passed = 0
        for query, accepted, keywords in CANDIDATES:
            qv = encode_text(sess, tok, query)
            sims = mat @ qv
            top = np.argsort(-sims)[:3]
            rows = [(objs[i]["cls"], objs[i]["caption"], float(sims[i])) for i in top]
            ok = any(
                cls in accepted or any(k in cap.lower() for k in keywords)
                for cls, cap, _ in rows
            )
            passed += ok
            flag = "PASS" if ok else "FAIL"
            print(f"  [{flag}] '{query}'  top={rows[0][0]} {sims[top[0]]:.3f}")
            if not ok:
                for cls, cap, s in rows:
                    print(f"         {cls:14s} {s:.3f}  {cap[:48]}")
        summary[variant] = f"{passed}/{len(CANDIDATES)}"
    print("\nSUMMARY (dense-only, numpy cosine == the JS path):", json.dumps(summary))


if __name__ == "__main__":
    main()
