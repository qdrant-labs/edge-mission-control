"""Auto mode (?auto): a scripted timeline that makes the demo play itself for
screen recording. Interactive mode does not use this; the user is the director.

Timestamps are seconds relative to video start. Everything triggered here is
real work: real detections, real embeddings, real shard queries.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)

# NOTE: query strings are validated against the mission footage by
# scripts/test_objects.py. The presenter narrates; the demo only acts.
TIMELINE = [
    (0.0, {"type": "scene", "title": "ACT 1 · OBJECT MEMORY"}),
    (20.0, {"type": "warm"}),
    (23.0, {"type": "query", "text": "a leather lounge chair"}),
    (45.0, {"type": "query", "text": "bar stools"}),
    (52.0, {"type": "scene", "title": "ACT 2 · OPEN VOCABULARY"}),
    (72.5, {"type": "query", "text": "a pool table"}),
    (90.0, {"type": "teach", "text": "a freestanding bathtub"}),
    (100.0, {"type": "query", "text": "a bed with pillows"}),
    (110.0, {"type": "scene", "title": "ACT 3 · INSTANT RECALL"}),
    (119.0, {"type": "query", "text": "a white bathtub"}),
    (133.0, {"type": "query", "text": "wine bottles on a tray"}),
    (144.0, {"type": "closing"}),
]

TYPEWRITER_SECONDS = 1.6  # how long the frontend takes to "type" a query


class Director:
    def __init__(self, session, emit):
        self.session = session
        self.emit = emit

    async def run_timeline(self):
        # Anchor to an absolute clock: action handlers (queries take ~2s)
        # must not push later events off the video timeline.
        start = asyncio.get_running_loop().time()
        for at, action in TIMELINE:
            now = asyncio.get_running_loop().time()
            await asyncio.sleep(max(0.0, start + at - now))
            if action["type"] == "query":
                self.emit({"type": "query_typed", "text": action["text"]})
                # Warm the search path while the typewriter animation plays so
                # the timed search shows steady-state latency.
                await self.session.warm_query()
                await asyncio.sleep(TYPEWRITER_SECONDS)
                await self.session.run_query(action["text"])
            elif action["type"] == "teach":
                await self.session.teach(action["text"])
            elif action["type"] == "warm":
                await self.session.warm_query()
            else:
                self.emit(action)
        logger.info("Timeline complete")
