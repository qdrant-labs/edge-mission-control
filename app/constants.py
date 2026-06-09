from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()

FOOTAGE_DIR = PROJECT_ROOT / "footage"
MISSION_VIDEO = FOOTAGE_DIR / "mission.mp4"
PROJECTION_FILE = FOOTAGE_DIR / "projection.npz"

DATA_DIR = PROJECT_ROOT / "edge-data"
SHARD_DIR = DATA_DIR / "shard"
THUMBS_DIR = DATA_DIR / "thumbs"

# "siglip2" (better retrieval, 768d) or "clip" (smaller/faster, 512d).
# Thresholds below are calibrated per backend; change them together.
ENCODER_BACKEND = "siglip2"
VISION_MODEL_NAME = "Qdrant/clip-ViT-B-32-vision"
TEXT_MODEL_NAME = "Qdrant/clip-ViT-B-32-text"
VECTOR_DIMENSION = 768 if ENCODER_BACKEND == "siglip2" else 512
VECTOR_NAME = "vision"

INGEST_FPS = 3.0          # frames embedded + stored per second of mission time

# Zero-shot scene recognition shown live on the HUD: every ingested frame is
# scored against these labels using the same CLIP embeddings, fully on-device.
# Tuned against the mission footage by scripts/test_retrieval.py.
HUD_LABELS = [
    "a sofa in a living room",
    "a fireplace",
    "a staircase",
    "a dining table and chairs",
    "a chandelier",
    "a kitchen island",
    "a stove with a range hood",
    "a pool table",
    "a bar counter with stools",
    "a projection screen",
    "a bed with a headboard",
    "a bathtub",
    "a bathroom vanity and mirror",
    "outdoor patio furniture",
    "a green lawn",
]
LABEL_PROMPT = "a photo of {}"
# SigLIP2 cosine bands are low and narrow: real hits 0.10-0.20, absent <0.08.
LABEL_THRESHOLD = 0.08
LABEL_TOP_K = 2
THUMB_WIDTH = 320
JPEG_QUALITY = 72

SEARCH_LIMIT = 4
MMR_LAMBDA = 0.9  # 1.0 = pure relevance, 0.0 = pure diversity
MMR_MAX_CANDIDATES = 100

CLOUD_URL = "http://localhost:6333"
CLOUD_COLLECTION = "edge_mission_demo"
SYNC_BATCH_SIZE = 24
SYNC_INTERVAL = 0.35      # seconds between sync batches while link is up
