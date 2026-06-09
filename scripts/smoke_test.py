"""Headless end-to-end check: trigger the demo over WebSocket and verify the
event stream covers every subsystem (boot, ingest, query, sync, captions).

Assumes the server is already running on localhost:8000.
Run: uv run python scripts/smoke_test.py [seconds]
"""

import asyncio
import json
import sys
from collections import Counter

import websockets

DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 45.0


async def main():
    counts = Counter()
    issues = []
    queries = []

    async with websockets.connect("ws://localhost:8000/ws", max_size=2**23) as ws:
        await ws.send(json.dumps({"cmd": "start", "mode": "auto"}))
        loop = asyncio.get_event_loop()
        deadline = loop.time() + DURATION

        while loop.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=deadline - loop.time())
            except asyncio.TimeoutError:
                break
            ev = json.loads(raw)
            counts[ev["type"]] += 1

            if ev["type"] == "frame_ingested":
                if not ev["thumb"]:
                    issues.append("frame without thumbnail")
                if not (0 <= ev["xy"][0] <= 1.1 and 0 <= ev["xy"][1] <= 1.1):
                    issues.append(f"projection out of range: {ev['xy']}")
            if ev["type"] == "query_result":
                queries.append(ev)
                if len(ev["results"]) == 0:
                    issues.append(f"query '{ev['text']}' returned no results")
                if any(not r["thumb"] for r in ev["results"]):
                    issues.append(f"query '{ev['text']}' has missing thumbs")

    print("Event counts:", dict(counts))
    for q in queries:
        tops = ", ".join(f"T+{r['video_ts']:.0f}s ({r['score']})" for r in q["results"])
        print(f"query '{q['text']}': {q['latency_us']}us offline={q['offline']} -> {tops}")
    if issues:
        print("ISSUES:")
        for i in issues:
            print(" -", i)
        sys.exit(1)
    required = {"boot_line", "video_start", "frame_ingested", "caption", "sync", "scene"}
    missing = required - set(counts)
    if missing:
        print("MISSING EVENT TYPES:", missing)
        sys.exit(1)
    print("Smoke test OK")


asyncio.run(main())
