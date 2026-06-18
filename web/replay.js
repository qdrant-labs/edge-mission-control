/* Replay driver: turns the baked event timeline into the same live experience,
   synced to the <video> clock. No WebSocket, no server.

   One mode for everyone: the robot's patrol plays automatically (detection,
   object discovery, captions, inventory, memory map, live metrics — all the
   real recorded output of the on-device pipeline), and contextual search
   suggestions surface as pills exactly when the robot has just seen the thing.
   Tap a pill or type your own query; both run locally (see search.js). */

const Replay = (() => {
  const BOOT_DELAY = 0.5;     // seconds between boot lines
  const MAX_PILLS = 3;        // never crowd the search space

  // Search suggestions that appear while the relevant objects are on screen.
  // Windows are tuned so at most ~2-3 overlap; the tick caps it at MAX_PILLS.
  const PILLS = [
    { text: "a leather lounge chair", from: 11, to: 44 },
    { text: "bar stools", from: 40, to: 70 },
    { text: "a pool table", from: 57, to: 90 },
    { text: "wine bottles on a tray", from: 69, to: 100 },
    { text: "a bed with pillows", from: 101, to: 136 },
    { text: "a white bathtub", from: 118, to: 154 },
  ];

  let data = null;
  let events = [];
  let evIdx = 0;
  let raf = 0;
  let running = false;
  let done = false;

  const ready = fetch("data/events.json").then((r) => r.json()).then((d) => { data = d; });
  const feed = () => document.getElementById("feed");

  function reset() {
    events = data.events.slice().sort((a, b) => a.t - b.t);
    evIdx = 0;
    done = false;
  }

  function activePills(t) {
    // Once the patrol ends, offer the full set as a menu (no longer crowding a
    // live moment). During the patrol, show only what's relevant, capped.
    if (done) return PILLS.map((p) => p.text);
    return PILLS.filter((p) => t >= p.from && t < p.to).map((p) => p.text).slice(-MAX_PILLS);
  }

  function tick() {
    if (!running) return;
    const t = feed().currentTime;
    while (evIdx < events.length && events[evIdx].t <= t) {
      window.handle(events[evIdx].ev);
      evIdx++;
    }
    if (!done && data.duration && t >= data.duration - 0.4) done = true;
    window.renderPills(activePills(t));
    raf = requestAnimationFrame(tick);
  }

  async function bootSequence() {
    document.getElementById("title-overlay").classList.add("hidden");
    const boot = document.getElementById("boot-overlay");
    boot.classList.remove("hidden");
    document.getElementById("boot-terminal").innerHTML = "";
    for (const line of data.boot) {
      window.handle({ type: "boot_line", text: line });
      await new Promise((r) => setTimeout(r, BOOT_DELAY * 1000));
    }
    await new Promise((r) => setTimeout(r, 300));
    boot.classList.add("hidden");
  }

  function play() {
    const v = feed();
    v.currentTime = 0;
    window.handle({ type: "video_start", vocab: data.vocab });
    running = true;
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(tick);
    return v.play();
  }

  async function start() {
    await ready;
    if (running) return;
    window.resetUI();
    reset();
    await bootSequence();
    try { await play(); } catch (e) { /* autoplay may be blocked; resumes on click */ }
  }

  function restart() {
    running = false;
    cancelAnimationFrame(raf);
    window.resetUI();
    reset();
    play().catch(() => {});
  }

  return { ready, start, restart, get duration() { return data?.duration; } };
})();

window.Replay = Replay;
