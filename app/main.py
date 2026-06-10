import asyncio
import json
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .captioner import Captioner
from .constants import MISSION_VIDEO, PROJECT_ROOT
from .detector import ObjectDetector
from .director import Director
from .encoder import get_encoder
from .session import DemoSession

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


class Hub:
    """Broadcasts demo events to connected browsers, callable from any thread."""

    def __init__(self):
        self.sockets: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    async def send_all(self, event: dict):
        message = json.dumps(event)
        dead = []
        for ws in self.sockets:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.sockets.discard(ws)

    def emit(self, event: dict):
        if self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(self.send_all(event), self.loop)


hub = Hub()
encoder = get_encoder()
detector = ObjectDetector()
captioner = Captioner()
session: DemoSession | None = None
demo_lock = asyncio.Lock()
demo_running = False


@app.on_event("startup")
async def startup():
    hub.loop = asyncio.get_running_loop()
    logger.info("Warming models (encoder, detector, captioner)...")
    await asyncio.to_thread(encoder.warm)
    await asyncio.to_thread(detector.warm)
    await asyncio.to_thread(captioner.warm)
    logger.info("Models warm. Ready to run.")


async def run_demo(mode: str):
    global demo_running, session
    async with demo_lock:
        if demo_running:
            return
        demo_running = True

    try:
        if session is not None:
            session.shutdown()
        session = DemoSession(encoder, detector, captioner, hub.emit)

        if mode == "auto":
            await session.start(boot_delay=1.1)
            director = Director(session, hub.emit)
            await director.run_timeline()
            # Let the tail of the video and the final sync play out.
            await asyncio.sleep(8)
            session.shutdown()
        else:
            await session.start(boot_delay=0.55)
            hub.emit({"type": "caption",
                      "text": "Search anything you remember. Teach a concept. Hover the memory map."})
            # Silent warm-up so the user's first search shows steady-state latency.
            await asyncio.sleep(2.5)
            await session.warm_query()
            # Stay live until the mission video ends; the pipeline announces
            # mission_complete itself. Memory remains searchable afterwards.
            await asyncio.to_thread(session.pipeline.thread.join)
            hub.emit({"type": "caption",
                      "text": "Mission complete. The memory stays searchable, entirely on-device."})
    finally:
        demo_running = False


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    hub.sockets.add(ws)
    logger.info("Browser connected (%d active)", len(hub.sockets))
    await ws.send_text(json.dumps({"type": "ready", "running": demo_running}))
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            cmd = msg.get("cmd")
            if cmd == "start":
                if demo_running:
                    logger.info("Start ignored: demo already running")
                else:
                    logger.info("Start command received (mode=%s)", msg.get("mode", "interactive"))
                    asyncio.create_task(run_demo(msg.get("mode", "interactive")))
            elif cmd == "query":
                text = (msg.get("text") or "").strip()[:120]
                cls = (msg.get("cls") or "").strip()[:60] or None
                if text and session is not None:
                    logger.info("User query: %r (cls=%r)", text, cls)
                    asyncio.create_task(session.run_query(text, cls=cls))
            elif cmd == "label":
                text = (msg.get("text") or "").strip()[:60]
                if text and session is not None:
                    logger.info("Teach concept: %r", text)
                    asyncio.create_task(session.teach(text))
    except WebSocketDisconnect:
        hub.sockets.discard(ws)
        logger.info("Browser disconnected (%d active)", len(hub.sockets))


@app.get("/")
async def index():
    return FileResponse(PROJECT_ROOT / "static" / "index.html")


@app.get("/footage/mission.mp4")
async def mission_video():
    return FileResponse(MISSION_VIDEO, media_type="video/mp4")


app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "static"), name="static")
