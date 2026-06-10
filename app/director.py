"""Auto mode (?auto): a scripted timeline that makes the demo play itself for
screen recording. Interactive mode does not use this; the user is the director.

Timestamps are seconds relative to video start. Everything triggered here is
real work: real detections, real embeddings, real shard queries.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)

# NOTE: query strings are validated against the mission footage by
# scripts/test_objects.py. Keep captions in sync with what retrieval returns.
TIMELINE = [
    (0.0, {"type": "scene", "title": "ACT 1 · OBJECT MEMORY"}),
    (0.5, {"type": "caption",
           "text": "Every object this robot sees becomes its own searchable memory, on the device."}),
    (9.0, {"type": "caption",
           "text": "Detect, track, embed, caption, store: one process, one device. No server, no cloud."}),
    (20.0, {"type": "warm"}),
    (21.0, {"type": "caption", "text": "Search the objects the robot has met so far."}),
    (23.0, {"type": "query", "text": "a leather lounge chair"}),
    (30.0, {"type": "caption",
            "text": "It found the object, not just the frame. Vectors plus captions, fused on-device."}),
    (45.0, {"type": "query", "text": "bar stools"}),
    (52.0, {"type": "scene", "title": "ACT 2 · OPEN VOCABULARY"}),
    (53.0, {"type": "caption",
            "text": "There is no network in this loop. Airplane mode would change nothing."}),
    (72.5, {"type": "query", "text": "a pool table"}),
    (80.0, {"type": "caption",
            "text": "Sub-millisecond answers from a shard that is a folder, not a server."}),
    (88.0, {"type": "caption",
            "text": "The detector has no fixed class list. Teach it something new, mid-mission."}),
    (90.0, {"type": "teach", "text": "a freestanding bathtub"}),
    (100.0, {"type": "query", "text": "a bed with pillows"}),
    (110.0, {"type": "scene", "title": "ACT 3 · INSTANT RECALL"}),
    (114.0, {"type": "caption",
             "text": "The concept it learned 24 seconds ago? Watch the feed."}),
    (119.0, {"type": "query", "text": "a white bathtub"}),
    (126.0, {"type": "caption",
             "text": "Recognized on sight, found on demand. One text embedding did that."}),
    (133.0, {"type": "query", "text": "wine bottles on a tray"}),
    (137.0, {"type": "caption",
             "text": "Seen once, an hour of patrol ago or a minute: every object stays findable."}),
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
