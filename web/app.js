/* Qdrant Edge mission control: renders real events streamed from the backend.
   Interactive by default (the user searches, cuts the link, teaches concepts).
   Append ?auto to the URL for the self-playing scripted version. */

const $ = (id) => document.getElementById(id);

const state = {
  points: [],        // memory map dots: {x, y, born, hit, thumb, ts, kind, cls, obj}
  objPoints: {},     // obj id -> index into points
  tracks: new Map(), // overlay boxes: tid -> {cur, target, cls, obj, conf, seen}
  inventory: {},     // obj id -> {el, cls, caption, thumb, t}
  lastQueryUs: null,
  edgeCount: 0,
  objectCount: 0,
  detectAvg: null,
  embedAvg: null,
  started: false,
};

/* ---------- transport: replay the baked timeline, no server ---------- */
const thumbURL = (name) => (name ? `thumbs/${name}` : "");

// Kiosk mode (?auto): the patrol loops on its own instead of stopping at the
// end and waiting for a manual replay.
const AUTO = new URLSearchParams(location.search).has("auto");
const LOOP_PAUSE_MS = 4000;  // hold the finished state before running again

function setHint(text) {
  const el = document.querySelector(".title-hint");
  if (el) el.textContent = text;
}

function requestStart() {
  if (state.started) return;
  state.started = true;
  setHint("starting…");
  Replay.start();
}

document.addEventListener("keydown", (e) => {
  if (e.code !== "Space") return;
  if (!state.started && document.activeElement.tagName !== "INPUT") {
    e.preventDefault();
    requestStart();
  }
});
$("title-overlay").addEventListener("click", requestStart);

// Self-playing: once the timeline is loaded, begin on its own so visitors
// immediately see it work. Muted video autoplays in modern browsers; if the
// browser blocks it, the title card simply waits for a press/click.
Replay.ready.then(() => {
  setHint("press space or click to begin");
  setTimeout(requestStart, 1400);
});

function handle(ev) {
  switch (ev.type) {
    case "boot_line": bootLine(ev.text); break;
    case "video_start": startVideo(ev); break;
    case "frame_ingested": onFrame(ev); break;
    case "object_discovered": onObject(ev); break;
    case "object_enriched": onEnriched(ev); break;
    case "inventory": onInventory(ev); break;
    case "query_result": showResults(ev); break;
    case "mission_complete": onMissionComplete(ev); break;
    case "label_added": onLabelAdded(ev.text); break;
  }
}

/* ---------- reset (a new run must not inherit the previous one) ---------- */
function resetUI() {
  state.points.length = 0;
  state.objPoints = {};
  state.tracks.clear();
  state.inventory = {};
  state.lastQueryUs = null;
  state.edgeCount = 0;
  state.objectCount = 0;
  state.detectAvg = null;
  state.embedAvg = null;

  const feed = $("feed");
  feed.pause();
  feed.currentTime = 0;

  $("boot-terminal").innerHTML = "";
  $("query-input").value = "";
  $("label-input").value = "";
  $("chips").innerHTML = "";
  $("results").innerHTML = "";
  $("search-badge").textContent = "";
  $("search-badge").className = "";
  $("inv-rail").innerHTML = "";
  $("inv-facets").innerHTML = "";
  $("inv-count").textContent = "0 unique objects";
  $("watch-lines").innerHTML = "";
  $("vocab-count").textContent = "";
  $("zoom-overlay").classList.add("hidden");
  $("map-tip").classList.add("hidden");
  $("replay-btn").classList.add("hidden");

  $("tick-detect").textContent = "—";
  $("tick-embed").textContent = "—";
  $("tick-upsert").textContent = "—";
  $("tick-count").textContent = "#0";
  $("m-vectors").textContent = "0";
  $("m-objects").textContent = "0";
  $("m-disk").innerHTML = `0.0<small> MB</small>`;
  $("m-detect").textContent = "—";
  $("map-count").textContent = "0 vectors";
}

/* ---------- boot ---------- */
function bootLine(text) {
  const div = document.createElement("div");
  div.className = "line";
  div.textContent = text;
  $("boot-terminal").appendChild(div);
}

function startVideo(ev) {
  $("boot-overlay").classList.add("hidden");
  if (ev && ev.vocab) $("vocab-count").textContent = `· ${ev.vocab} concepts`;
  $("feed").play();
  $("query-input").focus();
}

/* ---------- mission clock ---------- */
setInterval(() => {
  const t = $("feed").currentTime;
  $("mission-clock").textContent = `T+${fmt(t)}`;
}, 250);

/* ---------- ingest ---------- */
function onFrame(ev) {
  state.edgeCount = ev.count;
  state.objectCount = ev.objects;
  state.detectAvg = state.detectAvg === null ? ev.detect_ms : state.detectAvg * 0.8 + ev.detect_ms * 0.2;
  state.embedAvg = state.embedAvg === null ? ev.embed_ms : state.embedAvg * 0.8 + ev.embed_ms * 0.2;

  state.points.push({
    x: ev.xy[0], y: ev.xy[1],
    born: performance.now(), hit: 0,
    thumb: ev.thumb, ts: ev.video_ts, kind: "frame",
  });

  updateBoxes(ev.boxes || []);

  const total = ev.count + ev.objects;
  $("tick-detect").textContent = `${ev.detect_ms.toFixed(0)}ms`;
  $("tick-embed").textContent = `${ev.embed_ms.toFixed(0)}ms`;
  $("tick-upsert").textContent = `${(ev.upsert_us / 1000).toFixed(1)}ms`;
  $("tick-count").textContent = `#${total}`;
  $("m-vectors").textContent = total;
  $("m-objects").textContent = ev.objects;
  $("m-disk").innerHTML = `${(ev.bytes / 1e6).toFixed(1)}<small> MB</small>`;
  $("m-detect").innerHTML = `${state.detectAvg.toFixed(0)}<small> ms</small>`;
  $("map-count").textContent = `${total} vectors`;
}

/* ---------- live detection overlay ---------- */
const overlay = $("overlay");
const octx = overlay.getContext("2d");

function updateBoxes(boxes) {
  const now = performance.now();
  const seen = new Set();
  for (const b of boxes) {
    seen.add(b.tid);
    const t = state.tracks.get(b.tid);
    if (t) {
      t.target = b.box;
      t.cls = b.cls;
      t.obj = b.obj;
      t.conf = b.conf;
      t.seen = now;
    } else {
      state.tracks.set(b.tid, {
        cur: [...b.box], target: b.box,
        cls: b.cls, obj: b.obj, conf: b.conf,
        born: now, seen: now,
      });
    }
  }
  for (const [tid, t] of state.tracks) {
    if (!seen.has(tid) && now - t.seen > 1100) state.tracks.delete(tid);
  }
}

/* Video uses object-fit: cover — map normalized coords to the visible crop. */
function contentRect(videoEl, w, h) {
  const vw = videoEl.videoWidth || 1920, vh = videoEl.videoHeight || 1080;
  const scale = Math.max(w / vw, h / vh);
  const cw = vw * scale, ch = vh * scale;
  return { x: (w - cw) / 2, y: (h - ch) / 2, w: cw, h: ch };
}

function drawOverlay(now, dt) {
  const wrap = $("feed-wrap");
  const w = wrap.clientWidth, h = wrap.clientHeight;
  // Re-allocate on EITHER dimension change: a stale-height bitmap gets
  // stretched by CSS and smears strokes into bands.
  if (overlay.width !== w * 2 || overlay.height !== h * 2) {
    overlay.width = w * 2; overlay.height = h * 2;
  }
  const c = octx;
  c.setTransform(2, 0, 0, 2, 0, 0);
  c.clearRect(0, 0, w, h);
  if (!state.tracks.size) return;

  const r = contentRect($("feed"), w, h);
  const k = 1 - Math.exp(-dt * 9);  // smooth pursuit between detection ticks

  c.font = "600 13.5px " + getComputedStyle(document.body).fontFamily;
  for (const t of state.tracks.values()) {
    for (let i = 0; i < 4; i++) t.cur[i] += (t.target[i] - t.cur[i]) * k;
    const age = (now - t.seen) / 1000;
    if (age > 1.1) continue;
    const fade = age < 0.7 ? 1 : 1 - (age - 0.7) / 0.4;
    const birth = Math.min(1, (now - t.born) / 250);

    const x = r.x + t.cur[0] * r.w, y = r.y + t.cur[1] * r.h;
    const bw = (t.cur[2] - t.cur[0]) * r.w, bh = (t.cur[3] - t.cur[1]) * r.h;
    const confirmed = !!t.obj;
    const alpha = (confirmed ? 0.95 : 0.45) * fade * birth;

    c.strokeStyle = confirmed ? `rgba(52, 240, 176, ${alpha})` : `rgba(138, 164, 255, ${alpha})`;
    c.lineWidth = confirmed ? 1.6 : 1;
    c.strokeRect(x, y, bw, bh);

    if (confirmed && bw > 46) {
      const label = t.cls;
      const tw = c.measureText(label).width + 11;
      c.fillStyle = `rgba(5, 5, 12, ${0.82 * fade})`;
      c.fillRect(x - 0.8, y - 20, tw, 19);
      c.fillStyle = `rgba(52, 240, 176, ${alpha})`;
      c.fillText(label, x + 4, y - 6);
    }
  }
}

/* ---------- object inventory ---------- */
function onObject(ev) {
  state.objectCount = ev.total;
  $("inv-count").textContent = `${ev.total} unique object${ev.total === 1 ? "" : "s"}`;

  state.points.push({
    x: ev.xy[0], y: ev.xy[1],
    born: performance.now(), hit: 0,
    thumb: ev.thumb, ts: ev.t, kind: "obj", cls: ev.cls, obj: ev.obj,
  });
  state.objPoints[ev.obj] = state.points.length - 1;

  const card = document.createElement("div");
  card.className = "inv-item";
  card.innerHTML =
    `<img src="${thumbURL(ev.thumb)}"><span class="inv-cls">${ev.cls}</span>`;
  card.addEventListener("click", () => {
    const info = state.inventory[ev.obj];
    showZoomRaw(ev.thumb,
      `<b>${ev.obj} · ${ev.cls}</b> · first seen T+${fmt(ev.t)}` +
      (info && info.caption ? `<br>“${info.caption}”` : ""));
  });
  state.inventory[ev.obj] = { el: card, cls: ev.cls, caption: null, thumb: ev.thumb, t: ev.t };

  const rail = $("inv-rail");
  rail.prepend(card);
  while (rail.children.length > 90) rail.lastChild.remove();
}

function onEnriched(ev) {
  const info = state.inventory[ev.obj];
  if (info) {
    info.caption = ev.caption;
    info.el.title = `“${ev.caption}”`;
    info.el.classList.add("captioned");
  }
}

function onInventory(ev) {
  const wrap = $("inv-facets");
  wrap.innerHTML = "";
  for (const [cls, count] of ev.classes.slice(0, 9)) {
    const b = document.createElement("button");
    b.className = "facet";
    b.innerHTML = `${cls} <b>${count}</b>`;
    b.addEventListener("click", () => {
      $("query-input").value = cls;
      runUserQuery(cls, cls);
    });
    wrap.appendChild(b);
  }
}

/* ---------- open vocabulary: teach the detector a concept ---------- */
$("label-input").addEventListener("keydown", (e) => {
  if (e.key !== "Enter") return;
  const text = $("label-input").value.trim();
  if (text) handle({ type: "label_added", text });
});

function onLabelAdded(text) {
  $("label-input").value = "";
  const el = $("watch-lines");
  const div = document.createElement("div");
  div.className = "watch-line";
  div.textContent = `◎ ${text}`;
  el.prepend(div);
  while (el.children.length > 3) el.lastChild.remove();
}

/* ---------- search ---------- */
const input = $("query-input");
input.addEventListener("keydown", (e) => {
  if (e.key !== "Enter") return;
  const text = input.value.trim();
  if (!text) return;
  runUserQuery(text);
});
// Begin downloading the on-device model as soon as the visitor shows intent to
// type, so a free-text query doesn't pay the full load cold.
input.addEventListener("focus", () => { StaticSearch.warmModel().catch(() => {}); }, { once: true });
/* Contextual suggestions: the replay driver decides which (max 3) are relevant
   right now; diff them into the chip row. Tap one to run that search locally. */
function renderPills(texts) {
  const wrap = $("chips");
  const want = new Set(texts);
  for (const el of [...wrap.children]) {
    if (!want.has(el.dataset.q)) el.remove();
  }
  const have = new Set([...wrap.children].map((el) => el.dataset.q));
  let fresh = null;
  for (const text of texts) {
    if (have.has(text)) continue;
    const b = document.createElement("button");
    b.className = "chip appear";
    b.dataset.q = text;
    b.textContent = text;
    b.addEventListener("click", () => { input.value = text; runUserQuery(text); });
    wrap.appendChild(b);
    fresh = text;
  }
  // Kiosk mode: when a new suggestion surfaces (the robot has just seen the
  // thing), run that search on its own so the demo drives itself.
  if (AUTO && fresh) {
    input.value = fresh;
    runUserQuery(fresh);
  }
}
window.renderPills = renderPills;

function runUserQuery(text, cls) {
  $("results").innerHTML = "";
  const badge = $("search-badge");
  badge.className = "";
  const instant = StaticSearch.hasCanned(text, cls) || StaticSearch.modelState === "ready";
  badge.textContent = instant ? "searching…" : "loading on-device model…";
  StaticSearch.search(text, cls || null)
    .then(showResults)
    .catch((err) => {
      console.error(err);
      badge.textContent = "search unavailable";
      badge.className = "off";
    });
}

function showResults(ev) {
  const objects = ev.objects || [];
  const moments = ev.moments || [];
  if (!objects.length && !moments.length) {
    $("search-badge").textContent = "no memories yet";
    return;
  }
  state.lastQueryUs = ev.latency_us;
  const ms = ev.latency_us / 1000;
  const weak = objects.length === 0 || objects.every((o) => o.weak);
  const badge = $("search-badge");
  badge.textContent = weak
    ? `weak match · maybe not seen yet · ${ms.toFixed(2)} ms`
    : `${ms.toFixed(2)} ms · hybrid · on-device`;
  badge.className = weak ? "off" : "";

  const wrap = $("results");
  wrap.innerHTML = "";

  objects.forEach((o, i) => {
    const div = document.createElement("div");
    div.className = o.weak ? "obj-card weak" : "obj-card";
    div.style.animationDelay = `${i * 90}ms`;
    const cap = o.caption
      ? `“${o.caption}”`
      : `<span class="capping">captioning…</span>`;
    div.innerHTML =
      `<img src="${thumbURL(o.thumb)}">
       <div class="obj-body">
         <div class="obj-head"><span class="obj-cls">${o.cls}</span>
           <span class="obj-score">${o.score !== null ? o.score.toFixed(3) : ""}</span></div>
         <div class="obj-cap">${cap}</div>
         <div class="obj-meta">seen T+${fmt(o.t_first)} · ${o.sightings || 1}× sightings</div>
       </div>`;
    div.addEventListener("click", () => showZoomRaw(o.thumb,
      `<b>“${ev.text}”</b> · ${o.cls} · score ${o.score !== null ? o.score.toFixed(3) : "—"}` +
      (o.caption ? `<br>“${o.caption}”` : "") +
      `<br>first seen T+${fmt(o.t_first)} · last T+${fmt(o.t_last)}`));
    wrap.appendChild(div);

    const idx = state.objPoints[o.obj];
    if (idx !== undefined) { state.points[idx].hit = performance.now(); }
  });

  if (moments.length) {
    const head = document.createElement("div");
    head.className = "moments-head";
    head.textContent = "MOMENTS · full-frame matches";
    wrap.appendChild(head);
    const row = document.createElement("div");
    row.className = "moments-row";
    moments.forEach((r) => {
      const d = document.createElement("div");
      d.className = "result";
      d.innerHTML = `<img src="${thumbURL(r.thumb)}"><div class="score">${r.score.toFixed(3)}</div>`;
      d.addEventListener("click", () => showZoomRaw(r.thumb,
        `<b>“${ev.text}”</b> · score ${r.score.toFixed(3)} · remembered at T+${fmt(r.video_ts)}`));
      row.appendChild(d);

      let best = -1, bestD = 1e9;
      state.points.forEach((p, idx) => {
        if (p.kind !== "frame") return;
        const dd = (p.x - r.xy[0]) ** 2 + (p.y - r.xy[1]) ** 2;
        if (dd < bestD) { bestD = dd; best = idx; }
      });
      if (best >= 0) state.points[best].hit = performance.now();
    });
    wrap.appendChild(row);
  }
}

function fmt(t) {
  t = Math.max(0, t || 0);
  const m = String(Math.floor(t / 60)).padStart(2, "0");
  const s = String(Math.floor(t % 60)).padStart(2, "0");
  return `${m}:${s}`;
}

/* ---------- result zoom ---------- */
function showZoomRaw(thumbName, metaHtml) {
  $("zoom-img").src = thumbURL(thumbName);
  $("zoom-meta").innerHTML = metaHtml;
  $("zoom-overlay").classList.remove("hidden");
}
$("zoom-overlay").addEventListener("click", () => $("zoom-overlay").classList.add("hidden"));

function onMissionComplete(ev) {
  if (AUTO) {
    setTimeout(() => Replay.restart(), LOOP_PAUSE_MS);  // loop for kiosk display
    return;
  }
  $("replay-btn").classList.remove("hidden");
}
$("replay-btn").addEventListener("click", () => Replay.restart());

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
    tip.querySelector("img").src = thumbURL(p.thumb);
    tip.querySelector("span").textContent = p.kind === "obj"
      ? `${p.obj} · ${p.cls} · T+${fmt(p.ts)}`
      : `memory #${best + 1} · T+${fmt(p.ts)}`;
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
  if (map.width !== w * 2 || map.height !== h * 2) { map.width = w * 2; map.height = h * 2; }
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
    const isObj = p.kind === "obj";

    if (age < 0.6) {
      c.beginPath();
      c.arc(x, y, 11 * (1 - birth) + 2, 0, 7);
      c.strokeStyle = `rgba(52, 240, 176, ${0.85 * (1 - birth)})`;
      c.stroke();
    }

    const isHit = hitAge < 6;
    const isHover = i === hoverIdx;
    c.beginPath();
    c.arc(x, y, isHover ? 4.6 : isHit ? 3.8 : isObj ? 3.1 : 2.6, 0, 7);
    c.fillStyle = isHover
      ? "rgba(52, 240, 176, 1)"
      : isHit
        ? `rgba(239, 45, 94, ${Math.max(0.65, 1 - hitAge / 8)})`
        : isObj
          ? `rgba(255, 190, 102, ${0.55 + 0.35 * birth})`
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
  if (lat.width !== w * 2 || lat.height !== h * 2) { lat.width = w * 2; lat.height = h * 2; }
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

let lastTick = performance.now();
function loop(now) {
  const dt = Math.min(0.1, (now - lastTick) / 1000);
  lastTick = now;
  drawOverlay(now, dt);
  drawMap(now);
  drawLat();
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

/* The replay driver (replay.js) dispatches recorded events through handle()
   and resets between runs; expose both. */
window.handle = handle;
window.resetUI = resetUI;
