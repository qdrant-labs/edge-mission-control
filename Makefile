CLOUD_PORT ?= 6333

.PHONY: setup footage prepare run test clean

setup:
	@command -v uv >/dev/null || (echo "uv not installed." && exit 1)
	@command -v ffmpeg >/dev/null || (echo "ffmpeg not installed." && exit 1)
	uv sync
	@echo "Downloading embedding models..."
	@uv run python -c "from app.encoder import get_encoder; get_encoder().warm()"
	@echo "Setup complete. Next: make footage prepare"

footage:
	@mkdir -p footage
	@cd footage && for id in 7578546 7578552 7578540 7578549 7578547 7578550 7578551 7578542 7578543; do \
		curl -sL -O "https://videos.pexels.com/video-files/$$id/$$id-hd_1920_1080_30fps.mp4"; \
	done
	@echo "Clips downloaded (Pexels License, free for commercial use)."

prepare:
	uv run python scripts/prepare_footage.py
	uv run python scripts/test_retrieval.py

run:
	@docker ps --format '{{.Ports}}' | grep -q $(CLOUD_PORT) || \
		echo "Warning: no Qdrant container on port $(CLOUD_PORT). Run: docker run -d -p 6333:6333 qdrant/qdrant"
	@rm -rf edge-data
	uv run uvicorn app.main:app --port 8000

test:
	uv run python scripts/smoke_test.py 160

clean:
	rm -rf edge-data .venv footage/mission.mp4
