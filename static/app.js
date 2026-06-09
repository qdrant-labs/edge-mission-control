/* Qdrant Edge mission control: renders real events streamed from the backend.
   Interactive by default (the user searches, cuts the link, explores the map).
   Append ?auto to the URL for the self-playing scripted version. */

const $ = (id) => document.getElementById(id);
const AUTO = location.search.includes("auto");
if (AUTO) document.body.classList.add("auto");

const state = {
  points: [],        // {x, y, born, hit, thumb, ts}
  hits: [],
  lastQueryUs: null,
  edgeCount: 0,
  embedAvg: null,
  upsertAvg: null,
  started: false,
  linkUp: true,
};

/* ---------- websocket (auto-reconnects across server restarts) ---------- */
let ws = null;
function setHint(text) {
  const el = document.querySelector(".title-hint");
  if (el) el.textContent = text;
}
function connect() {
  setHint("connecting…");
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onmessage = (e) => handle(JSON.parse(e.data));
  ws.onclose = () => {
    state.started = false;
    setHint("connecting…");
    setTimeout(connect, 800);
  };
}
connect();

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}

function requestStart() {
  if (state.started) return;
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    setHint("not connected · retrying…");
    return;
  }
  state.started = true;
  send({ cmd: "start", mode: AUTO ? "auto" : "interactive" });
  setHint("starting…");
}

document.addEventListener("keydown", (e) => {
  if (e.code !== "Space") return;
  // Space starts the demo from the title card; in inputs it types a space.
  if (!state.started && document.activeElement.tagName !== "INPUT") {
    e.preventDefault();
    requestStart();
  }
});
$("title-overlay").addEventListener("click", requestStart);

function handle(ev) {
  switch (ev.type) {
    case "ready":
      setHint(ev.running
        ? "a run is already in progress · restart the server for a clean take"
        : "press space or click to begin");
      break;
    case "phase":
      if (ev.name === "boot") {
        resetUI();
        $("title-overlay").classList.add("hidden");
        $("boot-overlay").classList.remove("hidden");
      }
      break;
    case "boot_line": bootLine(ev.text); break;
    case "video_start": startVideo(); break;
    case "frame_ingested": onFrame(ev); break;
    case "query_typed": typeQuery(ev.text); break;
    case "query_result": showResults(ev); break;
    case "sync": onSync(ev); break;
    case "caption": showCaption(ev.text); break;
    case "scene": showScene(ev.title); break;
    case "mission_complete": onMissionComplete(ev); break;
    case "label_added": $("label-input").value = ""; break;
    case "closing": showClosing(); break;
  }
}

/* ---------- reset (a new run must not inherit the previous one) ---------- */
function resetUI() {
  state.points.length = 0;
  state.hits = [];
  state.lastQueryUs = null;
  state.edgeCount = 0;
  state.embedAvg = null;
  state.upsertAvg = null;
  state.linkUp = true;
  lastLabels = [];
  lastLabelTime = 0;

  const feed = $("feed");
  feed.pause();
  feed.currentTime = 0;

  $("boot-terminal").innerHTML = "";
  $("query-input").value = "";
  $("label-input").value = "";
  updateChips(0);
  $("results").innerHTML = "";
  $("search-badge").textContent = "";
  $("search-badge").className = "";
  $("caption-text").classList.remove("show");
  $("caption-text").textContent = "";
  $("scene-badge").classList.remove("show");
  $("scene-badge").textContent = "";
  $("closing-overlay").classList.add("hidden");
  $("zoom-overlay").classList.add("hidden");
  $("map-tip").classList.add("hidden");
  $("replay-btn").classList.add("hidden");
  $("hud-label-lines").innerHTML = `<span class="lbl-none">scanning…</span>`;

  $("tick-embed").textContent = "—";
  $("tick-upsert").textContent = "—";
  $("tick-count").textContent = "#0";
  $("m-vectors").textContent = "0";
  $("m-disk").innerHTML = `0.0<small> MB</small>`;
  $("m-embed").textContent = "—";
  $("m-upsert").textContent = "—";
  $("map-count").textContent = "0 vectors";
  $("s-queue").textContent = "0";
  $("s-cloud").textContent = "0";
  $("s-edge").textContent = "0";
  $("queue-fill").style.width = "0%";
  $("queue-fill").className = "";
  $("cloud-fill").style.width = "0%";
  $("cloud-fill").className = "";
  $("link-pill").textContent = "CONNECTED";
  $("link-pill").className = "pill ok";
  $("link-toggle").textContent = "CUT LINK";
}

/* ---------- boot ---------- */
function bootLine(text) {
  const div = document.createElement("div");
  div.className = "line";
  div.textContent = text;
  $("boot-terminal").appendChild(div);
}

function startVideo() {
  $("boot-overlay").classList.add("hidden");
  $("feed").play();
  if (!AUTO) $("query-input").focus();
}

/* ---------- mission clock ---------- */
setInterval(() => {
  const t = $("feed").currentTime;
  const m = String(Math.floor(t / 60)).padStart(2, "0");
  const s = String(Math.floor(t % 60)).padStart(2, "0");
  $("mission-clock").textContent = `T+${m}:${s}`;
}, 250);

/* ---------- ingest ---------- */
function onFrame(ev) {
  state.edgeCount = ev.count;
  state.embedAvg = state.embedAvg === null ? ev.embed_ms : state.embedAvg * 0.8 + ev.embed_ms * 0.2;
  state.upsertAvg = state.upsertAvg === null ? ev.upsert_us : state.upsertAvg * 0.8 + ev.upsert_us * 0.2;

  state.points.push({
    x: ev.xy[0], y: ev.xy[1],
    born: performance.now(), hit: 0,
    thumb: ev.thumb, ts: ev.video_ts,
  });

  $("tick-embed").textContent = `${ev.embed_ms.toFixed(0)}ms`;
  $("tick-upsert").textContent = `${(ev.upsert_us / 1000).toFixed(1)}ms`;
  $("tick-count").textContent = `#${ev.count}`;
  $("m-vectors").textContent = ev.count;
  $("m-disk").innerHTML = `${(ev.bytes / 1e6).toFixed(1)}<small> MB</small>`;
  $("m-embed").innerHTML = `${state.embedAvg.toFixed(0)}<small> ms</small>`;
  $("m-upsert").innerHTML = `${(state.upsertAvg / 1000).toFixed(1)}<small> ms</small>`;
  $("map-count").textContent = `${ev.count} vectors`;
  $("s-edge").textContent = ev.count;

  const flash = $("capture-flash");
  flash.classList.remove("flash");
  void flash.offsetWidth;
  flash.classList.add("flash");

  renderLabels(ev.labels || []);
  updateChips(ev.video_ts);
}

/* Chips unlock once the robot has actually seen their subject. */
function updateChips(ts) {
  document.querySelectorAll(".chip").forEach((chip) => {
    const after = parseFloat(chip.dataset.after || "0");
    const locked = ts < after;
    chip.disabled = locked;
    chip.title = locked ? "the robot has not seen this yet" : "";
  });
}

/* Sticky label display: hold the last confident recognition briefly so the
   HUD reads steadily instead of flickering frame to frame. */
let lastLabels = [];
let lastLabelTime = 0;
function renderLabels(labels) {
  const now = performance.now();
  if (labels.length) {
    lastLabels = labels;
    lastLabelTime = now;
  } else if (now - lastLabelTime > 2500) {
    lastLabels = [];
  }
  const el = $("hud-label-lines");
  if (!lastLabels.length) {
    el.innerHTML = `<span class="lbl-none">scanning…</span>`;
    return;
  }
  el.innerHTML = lastLabels
    .map(([name, score, pinned]) =>
      pinned
        ? `<span class="lbl-pinned">◎ ${name}<span class="lbl-score">${score.toFixed(2)}</span></span>`
        : `▣ ${name}<span class="lbl-score">${score.toFixed(2)}</span>`)
    .join("<br>");
}

/* ---------- search (interactive) ---------- */
const input = $("query-input");
input.addEventListener("keydown", (e) => {
  if (e.key !== "Enter") return;
  const text = input.value.trim();
  if (!text) return;
  runUserQuery(text);
});
document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    input.value = chip.textContent;
    runUserQuery(chip.textContent);
  });
});

function runUserQuery(text) {
  $("results").innerHTML = "";
  const badge = $("search-badge");
  badge.textContent = "searching…";
  badge.className = "";
  send({ cmd: "query", text });
}

/* ---------- open vocabulary: teach the recognizer a concept ---------- */
$("label-input").addEventListener("keydown", (e) => {
  if (e.key !== "Enter") return;
  const text = $("label-input").value.trim();
  if (text) send({ cmd: "label", text });
});

/* ---------- search (auto mode typewriter) ---------- */
let typeTimer = null;
function typeQuery(text) {
  clearInterval(typeTimer);
  $("results").innerHTML = "";
  $("search-badge").textContent = "";
  input.value = "";
  let i = 0;
  typeTimer = setInterval(() => {
    input.value = text.slice(0, ++i);
    if (i >= text.length) clearInterval(typeTimer);
  }, Math.min(70, 1300 / text.length));
}

const WEAK_SCORE = 0.085;  // SigLIP2 band: real hits 0.10-0.20, absent <0.082

function showResults(ev) {
  if (ev.results.length === 0) {
    $("search-badge").textContent = "no memories yet";
    return;
  }
  state.lastQueryUs = ev.latency_us;
  const ms = ev.latency_us / 1000;
  const topScore = Math.max(...ev.results.map((r) => r.score));
  const weak = topScore < WEAK_SCORE;
  const badge = $("search-badge");
  badge.textContent = weak
    ? `weak match · maybe not seen yet · ${ms.toFixed(2)} ms`
    : ev.offline
      ? `${ms.toFixed(2)} ms · on-device · OFFLINE`
      : `${ms.toFixed(2)} ms · on-device`;
  badge.className = weak || ev.offline ? "off" : "";

  const wrap = $("results");
  wrap.innerHTML = "";
  ev.results.forEach((r, i) => {
    const div = document.createElement("div");
    div.className = weak ? "result weak" : "result";
    div.style.animationDelay = `${i * 90}ms`;
    div.innerHTML = `<img src="data:image/jpeg;base64,${r.thumb}"><div class="score">${r.score.toFixed(3)}</div>`;
    div.addEventListener("click", () => showZoom(r, ev.text));
    wrap.appendChild(div);
  });

  // highlight hits on the memory map
  const now = performance.now();
  state.hits = [];
  ev.results.forEach((r) => {
    let best = -1, bestD = 1e9;
    state.points.forEach((p, idx) => {
      const d = (p.x - r.xy[0]) ** 2 + (p.y - r.xy[1]) ** 2;
      if (d < bestD) { bestD = d; best = idx; }
    });
    if (best >= 0) { state.points[best].hit = now; state.hits.push(best); }
  });
}

function fmt(t) {
  const m = String(Math.floor(t / 60)).padStart(2, "0");
  const s = String(Math.floor(t % 60)).padStart(2, "0");
  return `${m}:${s}`;
}

/* ---------- result zoom ---------- */
function showZoom(r, query) {
  $("zoom-img").src = `data:image/jpeg;base64,${r.thumb}`;
  $("zoom-meta").innerHTML =
    `<b>“${query}”</b> · score ${r.score.toFixed(3)} · remembered at T+${fmt(r.video_ts)}`;
  $("zoom-overlay").classList.remove("hidden");
}
$("zoom-overlay").addEventListener("click", () => $("zoom-overlay").classList.add("hidden"));

/* ---------- uplink ---------- */
$("link-toggle").addEventListener("click", () => {
  send({ cmd: "link", up: !state.linkUp });
});

function onSync(ev) {
  state.linkUp = ev.link_up;
  $("s-queue").textContent = ev.queue;
  $("s-cloud").textContent = ev.cloud_count;
  const qf = $("queue-fill");
  qf.style.width = `${Math.min(100, ev.queue * 1.2)}%`;
  qf.className = ev.queue > 40 ? "danger" : "";
  const cf = $("cloud-fill");
  const pct = state.edgeCount ? (ev.cloud_count / state.edgeCount) * 100 : 0;
  cf.style.width = `${pct}%`;
  cf.className = pct >= 99.5 && state.edgeCount > 0 ? "done" : "";

  const pill = $("link-pill");
  if (ev.link_up) {
    pill.textContent = "CONNECTED";
    pill.className = "pill ok";
    $("link-toggle").textContent = "CUT LINK";
  } else {
    pill.textContent = "LINK LOST";
    pill.className = "pill lost";
    $("link-toggle").textContent = "RESTORE LINK";
  }
}

/* ---------- narrative ---------- */
let captionTimer = null;
function showCaption(text) {
  const el = $("caption-text");
  clearTimeout(captionTimer);
  el.classList.remove("show");
  captionTimer = setTimeout(() => {
    el.textContent = text;
    el.classList.add("show");
  }, 350);
}

function showScene(title) {
  const el = $("scene-badge");
  el.textContent = title;
  el.classList.add("show");
}

function onMissionComplete(ev) {
  $("replay-btn").classList.remove("hidden");
}
$("replay-btn").addEventListener("click", () => {
  // Replay is an explicit fresh run; bypass the start latch.
  send({ cmd: "start", mode: AUTO ? "auto" : "interactive" });
});

function showClosing() {
  $("closing-overlay").classList.remove("hidden");
}

/* ---------- memory map canvas + hover ---------- */
const map = $("map");
const mctx = map.getContext("2d");
const MAP_PAD = 18;
let hoverIdx = -1;

map.addEventListener("mousemove", (e) => {
  const rect = map.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  const w = rect.width, h = rect.height;
  let best = -1, bestD = 18 * 18;
  state.points.forEach((p, idx) => {
    const x = MAP_PAD + p.x * (w - MAP_PAD * 2);
    const y = MAP_PAD + p.y * (h - MAP_PAD * 2);
    const d = (x - mx) ** 2 + (y - my) ** 2;
    if (d < bestD) { bestD = d; best = idx; }
  });
  hoverIdx = best;
  const tip = $("map-tip");
  if (best >= 0) {
    const p = state.points[best];
    tip.querySelector("img").src = `data:image/jpeg;base64,${p.thumb}`;
    tip.querySelector("span").textContent = `memory #${best + 1} · T+${fmt(p.ts)}`;
    tip.style.left = `${Math.min(mx + 14, w - 190)}px`;
    tip.style.top = `${Math.min(my + 14, h - 140)}px`;
    tip.classList.remove("hidden");
  } else {
    tip.classList.add("hidden");
  }
});
map.addEventListener("mouseleave", () => {
  hoverIdx = -1;
  $("map-tip").classList.add("hidden");
});

function drawMap(now) {
  const w = map.clientWidth, h = map.clientHeight;
  if (map.width !== w * 2) { map.width = w * 2; map.height = h * 2; }
  const c = mctx;
  c.setTransform(2, 0, 0, 2, 0, 0);
  c.clearRect(0, 0, w, h);

  // grid
  c.strokeStyle = "rgba(108, 140, 255, 0.10)";
  c.lineWidth = 1;
  for (let gx = 0; gx <= w; gx += 36) {
    c.beginPath(); c.moveTo(gx, 0); c.lineTo(gx, h); c.stroke();
  }
  for (let gy = 0; gy <= h; gy += 36) {
    c.beginPath(); c.moveTo(0, gy); c.lineTo(w, gy); c.stroke();
  }

  for (let i = 0; i < state.points.length; i++) {
    const p = state.points[i];
    const x = MAP_PAD + p.x * (w - MAP_PAD * 2);
    const y = MAP_PAD + p.y * (h - MAP_PAD * 2);
    const age = (now - p.born) / 1000;
    const birth = Math.min(1, age / 0.6);
    const hitAge = p.hit ? (now - p.hit) / 1000 : 99;

    if (age < 0.6) {
      c.beginPath();
      c.arc(x, y, 11 * (1 - birth) + 2, 0, 7);
      c.strokeStyle = `rgba(52, 240, 176, ${0.85 * (1 - birth)})`;
      c.stroke();
    }

    const isHit = hitAge < 6;
    const isHover = i === hoverIdx;
    c.beginPath();
    c.arc(x, y, isHover ? 4.6 : isHit ? 3.8 : 2.6, 0, 7);
    c.fillStyle = isHover
      ? "rgba(52, 240, 176, 1)"
      : isHit
        ? `rgba(239, 45, 94, ${Math.max(0.65, 1 - hitAge / 8)})`
        : `rgba(138, 164, 255, ${0.45 + 0.4 * birth})`;
    c.fill();

    if (isHit && hitAge < 1.6) {
      c.beginPath();
      c.arc(x, y, 5 + hitAge * 15, 0, 7);
      c.strokeStyle = `rgba(239, 45, 94, ${0.9 * (1 - hitAge / 1.6)})`;
      c.lineWidth = 2;
      c.stroke();
    }
  }
}

/* ---------- latency bar canvas (log scale) ---------- */
const lat = $("latbar");
const lctx = lat.getContext("2d");
const LOG_MIN = Math.log10(0.05);   // 0.05 ms
const LOG_MAX = Math.log10(500);    // 500 ms

function lx(ms, w) {
  return ((Math.log10(ms) - LOG_MIN) / (LOG_MAX - LOG_MIN)) * w;
}

function drawLat() {
  const w = lat.clientWidth, h = lat.clientHeight;
  if (lat.width !== w * 2) { lat.width = w * 2; lat.height = h * 2; }
  const c = lctx;
  c.setTransform(2, 0, 0, 2, 0, 0);
  c.clearRect(0, 0, w, h);

  const barY = 4, barH = 20, tickY = barY + barH + 15;

  c.font = "12px monospace";
  c.fillStyle = "rgba(176, 180, 210, 0.85)";
  for (const t of [0.1, 1, 10, 100]) {
    const x = lx(t, w);
    c.fillRect(x, barY, 1.5, barH + 5);
    c.fillText(`${t}ms`, x + 4, tickY);
  }

  const cx1 = lx(80, w), cx2 = lx(200, w);
  c.fillStyle = "rgba(255, 190, 102, 0.22)";
  c.fillRect(cx1, barY, cx2 - cx1, barH);
  c.strokeStyle = "rgba(255, 190, 102, 0.7)";
  c.strokeRect(cx1, barY, cx2 - cx1, barH);

  const ms = state.lastQueryUs !== null ? state.lastQueryUs / 1000 : null;
  if (ms !== null) {
    const x = lx(Math.max(0.051, ms), w);
    c.fillStyle = "#34f0b0";
    c.fillRect(x - 2, barY - 3, 4, barH + 6);
  }

  const legY = h - 6;
  c.font = "bold 13px monospace";
  c.fillStyle = "#34f0b0";
  c.fillRect(0, legY - 11, 12, 12);
  c.fillText(ms !== null ? `local search ${ms.toFixed(2)} ms` : "local search", 18, legY);
  const cloudX = w / 2 + 6;
  c.fillStyle = "rgba(255, 190, 102, 0.5)";
  c.fillRect(cloudX, legY - 11, 12, 12);
  c.strokeStyle = "rgba(255, 190, 102, 0.9)";
  c.strokeRect(cloudX, legY - 11, 12, 12);
  c.fillStyle = "rgba(255, 200, 120, 1)";
  c.fillText("cloud round trip (typical)", cloudX + 18, legY);
}

function loop(now) {
  drawMap(now);
  drawLat();
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);
