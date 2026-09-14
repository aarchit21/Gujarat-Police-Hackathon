// Operator token for this console. `let`, not `const`: every ${TOKEN} below is
// interpolated at render time, so changing it in the header field and
// refreshing is enough to re-auth the whole console. It used to be a hard
// constant, which meant any host with a real ADMIN_TOKEN got a 401 on every
// call -- including evidence crops -- with no way to fix it from the UI.
const DEV_TOKEN_KEY = "gp-dev-operator-token";
let TOKEN = sessionStorage.getItem(DEV_TOKEN_KEY) || "p0-operator";

// Leaflet and the OSM basemap are the only parts of this console that need the
// network. Leaflet itself is vendored under /static/vendor/, but if it still
// fails to load (missing file, corrupted download, blocked by policy) every
// L.* call below would throw at module scope and the ENTIRE console would be a
// blank page -- no tabs, no alerts, no watchlist, no search. The stub keeps the
// operations UI alive without a basemap and says so on screen rather than
// leaving an empty grey box that looks like "no cameras".
const MAP_AVAILABLE = typeof L !== "undefined" && L !== null && typeof L.map === "function";

function leafletStub() {
  const node = {};
  const self = () => node;
  Object.assign(node, {
    setView: self, addTo: self, addLayer: self, removeLayer: self,
    clearLayers: self, bindPopup: self, openPopup: self, on: self,
    setStyle: self, fitBounds: self, invalidateSize: self, remove: self,
  });
  return node;
}

if (!MAP_AVAILABLE) {
  window.L = {
    map: leafletStub, tileLayer: leafletStub, layerGroup: leafletStub,
    circleMarker: leafletStub, polyline: leafletStub, marker: leafletStub,
  };
}

const map = L.map("map").setView([22.3, 71.2], 7);
// Basemap tiles are the one genuinely remote asset left. Losing them leaves the
// markers and routes drawn on a blank canvas, which still carries the geometry.
const tiles = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: "&copy; OpenStreetMap",
}).addTo(map);

const markers = L.layerGroup().addTo(map);
const links = L.layerGroup().addTo(map);
const routes = L.layerGroup().addTo(map);

function showMapNotice(text) {
  const host = document.getElementById("map");
  if (!host || document.getElementById("mapOfflineNote")) return;
  const el = document.createElement("div");
  el.id = "mapOfflineNote";
  el.className = "map-offline-note";
  el.textContent = text;
  host.appendChild(el);
}

if (!MAP_AVAILABLE) {
  showMapNotice(
    "Basemap library unavailable — map disabled. All camera, alert, watchlist " +
    "and vehicle-search functions below continue to work. Coordinates are still " +
    "exported via CSV/GeoJSON."
  );
} else {
  // One failed tile is normal (edge of coverage); a burst means no basemap.
  let tileErrors = 0;
  tiles.on("tileerror", () => {
    tileErrors += 1;
    if (tileErrors === 6) {
      showMapNotice(
        "Basemap tiles unreachable (offline) — camera markers and routes are " +
        "still plotted at their true coordinates on a blank canvas."
      );
    }
  });
}
let previewCam = null;
let hlsPlayer = null;
let snapshotTimer = null;
let didFitCameras = false;
let lastCameras = [];

function headers(json) {
  const h = { Authorization: `Bearer ${TOKEN}` };
  if (json) h["Content-Type"] = "application/json";
  return h;
}

async function j(url, opts = {}) {
  const res = await fetch(url, {
    ...opts,
    headers: { ...headers(Boolean(opts.body)), ...(opts.headers || {}) },
  });
  if (!res.ok) throw new Error(await res.text());
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) return res.json();
  return res.text();
}

function statusClass(s) {
  if (s === "connected") return "status-connected";
  return "status-blocked";
}

function coordNote(c, html = true) {
  let text = "";
  if (c.coords_are_inferred) text = "Approximate city from camera name — not surveyed GPS.";
  else if (c.coords_are_placeholder) text = "placeholder map position (catalogue omitted lat/lng)";
  if (!text) return "";
  return html ? `${text}<br>` : text;
}

function resolutionText(c) {
  if (c.width && c.height) return `${c.width}×${c.height}`;
  return "—";
}

function govFeedText(h) {
  return h.government_feed_label || h.government_feed_status || "—";
}

function ollamaVisionText(h) {
  const v = h.ollama_vision || {};
  if (v.label) return v.label;
  if (!v.enabled) return "Ollama off";
  const where = v.cloud ? "Ollama Cloud" : "Ollama local";
  const live = v.live || v.reachable ? "live" : "not live";
  const model = v.resolved_model || v.configured_model || "";
  return `${where} · ${live}${model ? " · " + model : ""}`;
}

function formatWhen(iso) {
  if (!iso) return "";
  let text = String(iso);
  if (/IST$/.test(text)) return text;
  if (!/Z$|[+-]\d\d:\d\d$/.test(text)) text += "Z";
  const d = new Date(text);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString("en-IN", { timeZone: "Asia/Kolkata", hour12: false }) + " IST";
}

function evidenceUrl(path) {
  if (!path) return "";
  return `/api/evidence?rel=${encodeURIComponent(path)}&token=${encodeURIComponent(TOKEN)}`;
}

async function loadCoverage() {
  const h = await j("/api/health");
  const db = h.database || {};
  document.getElementById("hostMeta").textContent =
    `DB ${db.type || "?"} · ${ollamaVisionText(h)} · gov ${govFeedText(h)}`;
  document.getElementById("coverageBanner").textContent =
    `${ollamaVisionText(h)}. ${h.honest_coverage}. Catalogue ${h.government_catalogue_count} (not a hardcoded 50). Own-feed ${h.own_feed_count}. Gov feed ${govFeedText(h)} — decode success is not a running worker. Last sync ${h.catalogue_synced_at || "never"}.`;
  document.getElementById("statsBar").innerHTML = [
    ["Onboarded", h.onboarded_count],
    ["Own-feed", h.own_feed_count],
    ["Gov catalogue", h.government_catalogue_count],
    ["Connected", h.connected_count],
    ["Analytics", h.analytics_active_count],
    ["Blocked/deferred", h.blocked_count],
    ["Queued", h.queued_count],
    ["Gov feed", govFeedText(h)],
    ["Ollama vision", ollamaVisionText(h)],
    ["Enhancement", (h.vision_enhancement && h.vision_enhancement.enabled) ? h.vision_enhancement.method : "off"],
    ["CPU ANPR", (h.cpu_anpr && h.cpu_anpr.models_ready) ? "FastALPR ready" : ((h.cpu_anpr && h.cpu_anpr.tesseract_available) ? "Tesseract fallback" : "not ready")],
    ["Plate recognition", (h.plate_recognition && h.plate_recognition.enabled) ? "ON" : "OFF"],
    ["ANPR attempts", (h.recognition && h.recognition.attempt_count) || 0],
    ["YOLO", (h.yolo_detector && h.yolo_detector.label) || "—"],
    ["Measured fps", h.measured_safe_fps || (h.capacity && h.capacity.measured_safe_fps) || "—"],
    ["Recommended fps", h.recommended_target_fps || (h.capacity && h.capacity.recommended_target_fps) || "—"],
    ["Database", db.type],
    ["PostGIS", db.postgis ? "yes" : "no"],
    ["Open captures", h.open_capture_count],
    ["Previews", h.preview_active_count],
    ["Catalogue live", h.catalogue_live_count],
    ["Decode ok", h.decode_ok_count],
    ["HTTP", h.catalogue_last_http_status || "—"],
    ["Hunt visited", (h.hunt && `${h.hunt.visited_count || 0}/${h.hunt.total || 0}`) || "—"],
    ["Hunt vehicles", (h.hunt && h.hunt.vehicles_seen) ?? "—"],
    ["Review alerts", h.alerts_requiring_review],
    ["Snap-to-road", (h.map_match && h.map_match.provider) ? `OSM ${h.map_match.provider}` : "OSRM (free OSM)"],
  ]
    .map(([k, v]) => `<span><b>${k}</b> ${v}</span>`)
    .join("");
  const demo = document.getElementById("demoStrip");
  if (demo) {
    demo.textContent = `Demo · analytics ${h.analytics_active_count || 0} running · last sighting ${h.last_sighting_plate || "—"} @ ${h.last_sighting_camera || "—"} ${formatWhen(h.last_sighting_at)} · ${ollamaVisionText(h)}`;
  }
  const huntEl = document.getElementById("huntStrip");
  const hunt = h.hunt || {};
  const slots = hunt.max_concurrent || (h.capacity && h.capacity.max_concurrent) || 4;
  if (huntEl) {
    huntEl.textContent = hunt.label
      || `Hunt ${hunt.enabled ? "on" : "idle"} · hunting ${hunt.hunting_count || 0}/${hunt.total || 0} · visited ${hunt.visited_count || 0}/${hunt.total || 0} · vehicles ${hunt.vehicles_seen || 0} · plates ${hunt.plates_read || 0}. ${slots} concurrent slots on this host, not all catalogue cameras at once.`;
  }
  document.querySelectorAll(".js-max-concurrent").forEach((el) => {
    if (document.activeElement !== el) el.value = String(slots);
  });
}

function concurrentFromUi() {
  const el = document.querySelector(".js-max-concurrent");
  const n = Number(el && el.value);
  if (!Number.isFinite(n) || n < 1) return 4;
  return Math.min(30, Math.floor(n));
}

function cameraBucket(c) {
  if (c.origin === "own_feed") return "own";
  if (c.origin === "government_catalogue") return "gov";
  return "placeholder";
}

async function loadCameras() {
  lastCameras = await j("/api/cameras");
  renderLedger();
}

function renderLedger() {
  const filter = (document.getElementById("ledgerFilter") || {}).value || "gov";
  const body = document.getElementById("ledger");
  if (!body) return;
  markers.clearLayers();
  body.innerHTML = "";
  const pts = [];
  lastCameras
    .filter((c) => filter === "all" || cameraBucket(c) === filter)
    .forEach((c) => {
    const color = c.analytics_active || c.worker_state === "running" ? "#2f6f4e" : c.last_hunted_at ? "#38a169" : c.catalogue_live ? "#2c5282" : c.status === "onboarded" || c.status === "connected" ? "#c4a35a" : "#9b2c2c";
    if (c.lat != null && c.lng != null && Number.isFinite(Number(c.lat)) && Number.isFinite(Number(c.lng))) {
      pts.push([c.lat, c.lng]);
      const marker = L.circleMarker([c.lat, c.lng], { radius: 7, color, fillOpacity: 0.85 })
        .bindPopup(
          `<b>${c.id}</b> · ${c.origin || ""}<br>${c.city || "location omitted"} · ${c.department}<br>${coordNote(c)}cat live ${c.catalogue_live} · decode ${c.decode_status}<br>hunting ${c.analytics_active ? "now" : "no"} · last hunted ${c.last_hunted_at_ist || "—"}<br>${c.status}: ${c.status_reason || ""}`
        )
        .addTo(markers);
      marker.on("click", () => showCamera(c));
    }
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${c.id}</td><td>${c.origin || ""}</td><td>${c.priority_class}</td><td>${c.processing_mode}</td><td class="${statusClass(c.status)}">${c.status}</td><td>${c.catalogue_live}</td><td>${c.decode_status}</td><td>${c.active_protocol || "—"}</td><td>${resolutionText(c)}</td><td>${c.analytics_active ? "yes" : "no"} / prev ${c.preview_active ? "yes" : "no"}</td>`;
    tr.onclick = () => showCamera(c);
    body.appendChild(tr);
  });
  if (!didFitCameras && pts.length) {
    map.fitBounds(pts, { padding: [40, 40], maxZoom: 10 });
    didFitCameras = true;
  }
}

function showCamera(c) {
  document.getElementById("workerCam").value = c.id;
  document.getElementById("cameraDetail").innerHTML = `
    <div class="card">
      <div><b>${c.id}</b> · ${c.name}</div>
      <div class="muted">${c.department} · ${c.city || ""} · last frame ${c.last_frame_at || "—"}</div>
      <div class="muted">origin ${c.origin || "—"} · codec ${c.codec || "unspecified"} · size ${resolutionText(c)} · pts ${c.last_pts_ms ?? "—"} · reconnects ${c.reconnect_count}</div>
      <div class="muted">last error: ${c.last_error || "none"}</div>
      <div class="row"><label class="muted">Plate recognition for this camera
        <select data-plate-mode><option value="inherit" ${c.plate_recognition_mode === "inherit" ? "selected" : ""}>inherit global setting</option><option value="on" ${c.plate_recognition_mode === "on" ? "selected" : ""}>on when global is on</option><option value="off" ${c.plate_recognition_mode === "off" ? "selected" : ""}>off for this camera</option></select>
      </label></div>
      <div class="muted">${coordNote(c, false)}</div>
      <div class="muted">${c.hls_preview_blocked ? "HLS stays server-side. Use Live frame." : ""}</div>
      <div class="row" style="margin-top:6px">
        <button data-prev="snapshot">Live frame</button>
        <button class="secondary" data-prev="hls">HLS preview</button>
        <button class="secondary" data-prev="whep">WHEP preview</button>
        <button class="secondary" data-start="${c.id}">Start worker</button>
        <button class="secondary" data-analyze="${c.id}">Bounded live analyze</button>
      </div>
    </div>`;
  document.getElementById("cameraDetail").querySelectorAll("[data-prev]").forEach((btn) => {
    btn.onclick = () => openPreview(c, btn.dataset.prev);
  });
  document.getElementById("cameraDetail").querySelector("[data-plate-mode]").onchange = async (event) => {
    await j(`/api/cameras/${encodeURIComponent(c.id)}`, { method: "PATCH", body: JSON.stringify({ plate_recognition_mode: event.target.value }) });
    await loadPlateRecognitionSettings();
    await refresh();
  };
  document.getElementById("cameraDetail").querySelector("[data-start]").onclick = async () => {
    await j(`/api/workers/${c.id}/start`, { method: "POST" });
    refresh();
  };
  document.getElementById("cameraDetail").querySelector("[data-analyze]").onclick = async () => {
    const btn = document.getElementById("cameraDetail").querySelector("[data-analyze]");
    btn.disabled = true;
    btn.textContent = "Analyzing…";
    try {
      const out = await j(`/api/cameras/${c.id}/analyze`, { method: "POST" });
      alert(`camera ${out.camera_id || c.id} sightings=${out.sightings || 0} alerts=${out.alerts || 0} ${out.error || ""}`);
      await refresh();
    } catch (err) {
      alert(String(err));
    } finally {
      btn.disabled = false;
      btn.textContent = "Bounded live analyze";
    }
  };
}

function stopSnapshotTimer() {
  if (snapshotTimer) {
    clearInterval(snapshotTimer);
    snapshotTimer = null;
  }
}

function showSnapshotPreview(c, note) {
  previewCam = c.id;
  const box = document.getElementById("previewBox");
  box.classList.remove("hidden");
  document.getElementById("previewTitle").textContent = `${c.id} live frame`;
  const video = document.getElementById("previewVideo");
  const img = document.getElementById("previewImage");
  video.classList.add("hidden");
  video.removeAttribute("src");
  img.classList.remove("hidden");
  const load = () => {
    img.src = `/api/cameras/${encodeURIComponent(c.id)}/snapshot?token=${encodeURIComponent(TOKEN)}&t=${Date.now()}`;
  };
  img.onerror = () => {
    document.getElementById("previewNote").textContent =
      "No live frame yet. Start a worker or bounded analyze first if RTSP is slow to open.";
  };
  load();
  stopSnapshotTimer();
  snapshotTimer = setInterval(load, 4000);
  document.getElementById("previewNote").textContent =
    note || "Operator snapshot from the server-side feed. Not a VMS archive. HLS is not sent to the browser.";
}

async function openPreview(c, protocol) {
  const out = await j(`/api/cameras/${c.id}/preview`, {
    method: "POST",
    body: JSON.stringify({ protocol }),
  });
  if (out.preview_blocked || !out.ok) {
    if (protocol !== "snapshot") {
      showSnapshotPreview(c, out.error || "Stream preview unavailable; trying live frame.");
      return;
    }
    alert(out.error || "preview blocked");
    return;
  }
  if (out.protocol === "snapshot" || out.snapshot) {
    showSnapshotPreview(c, out.note);
    return;
  }
  previewCam = c.id;
  const box = document.getElementById("previewBox");
  box.classList.remove("hidden");
  document.getElementById("previewTitle").textContent = `${c.id} ${out.protocol} preview`;
  const video = document.getElementById("previewVideo");
  const img = document.getElementById("previewImage");
  img.classList.add("hidden");
  video.classList.remove("hidden");
  if (hlsPlayer) {
    hlsPlayer.destroy();
    hlsPlayer = null;
  }
  if (out.protocol === "hls" && window.Hls && Hls.isSupported()) {
    hlsPlayer = new Hls();
    hlsPlayer.loadSource(out.url);
    hlsPlayer.attachMedia(video);
  } else {
    video.src = out.url;
  }
}

async function closePreview() {
  stopSnapshotTimer();
  if (previewCam) {
    try {
      await j(`/api/cameras/${previewCam}/preview/stop`, { method: "POST" });
    } catch (_e) {
      /* ignore */
    }
  }
  previewCam = null;
  if (hlsPlayer) {
    hlsPlayer.destroy();
    hlsPlayer = null;
  }
  document.getElementById("previewVideo").removeAttribute("src");
  const img = document.getElementById("previewImage");
  if (img) {
    img.removeAttribute("src");
    img.classList.add("hidden");
  }
  document.getElementById("previewBox").classList.add("hidden");
}

async function loadWatchlist() {
  const rows = await j("/api/watchlist");
  const root = document.getElementById("watchlistRows");
  if (!root) return;
  if (!rows.length) {
    root.innerHTML = `<div class="card muted">Watchlist is empty.</div>`;
  } else {
    root.innerHTML = rows
      .map(
        (w) => `<div class="card">
          <div><b>${w.plate_norm}</b> · ${w.purpose} · ${w.priority} · ${w.active ? "active" : "inactive"}</div>
          <div class="muted">${w.notes || ""}</div>
          <div class="row" style="margin-top:6px">
            <button class="secondary" data-hist="${w.plate_norm}">History</button>
            <button class="secondary" data-wl="${w.id}" data-on="${w.active ? "0" : "1"}">${w.active ? "Deactivate" : "Activate + rematch"}</button>
          </div>
        </div>`
      )
      .join("");
    root.querySelectorAll("[data-hist]").forEach((btn) => {
      btn.onclick = () => {
        document.getElementById("plateQuery").value = btn.dataset.hist;
        document.querySelector('.tab[data-tab="alerts"]').click();
        searchPlate();
      };
    });
    root.querySelectorAll("[data-wl]").forEach((btn) => {
      btn.onclick = async () => {
        const on = btn.dataset.on === "1";
        await j(`/api/watchlist/${btn.dataset.wl}`, {
          method: "PATCH",
          body: JSON.stringify({ active: on, rematch: on }),
        });
        loadWatchlist();
        loadAlerts();
      };
    });
  }
  const observed = await j("/api/observed-plates");
  const obs = document.getElementById("observedPlates");
  if (!observed.length) {
    obs.innerHTML = `<div class="card muted">No persisted sightings yet.</div>`;
    return;
  }
  obs.innerHTML = observed
    .slice(0, 40)
    .map(
      (p) => `<div class="card">
        <div><b>${p.plate_norm}</b> · ${p.count} sighting(s) · last ${p.last_camera} · ${p.syntax_ok ? "syntax ok" : "syntax flag no"}</div>
        <div class="muted">${p.last_time || ""} · ${p.model_id || ""} · ${p.watchlisted ? "already on watchlist" : "not watchlisted"}</div>
        ${
          p.watchlisted
            ? ""
            : `<button data-add="${p.plate_norm}">Add to watchlist and rematch</button>`
        }
      </div>`
    )
    .join("");
  obs.querySelectorAll("[data-add]").forEach((btn) => {
    btn.onclick = async () => {
      const out = await j("/api/watchlist", {
        method: "POST",
        body: JSON.stringify({
          plate_raw: btn.dataset.add,
          purpose: "operator_added_from_sighting",
          rematch: true,
        }),
      });
      alert(`Watchlist ${out.plate_norm}. Rematch created ${out.rematch && out.rematch.alerts_created} alert(s) from persisted sightings.`);
      loadWatchlist();
      loadAlerts();
    };
  });
}

function locationText(row) {
  if (row.location) return row.location;
  if (row.city) return row.city;
  if (row.lat != null && row.lng != null) return `${Number(row.lat).toFixed(4)}, ${Number(row.lng).toFixed(4)}`;
  return "location omitted";
}

function vehicleRowCard(row, extraHtml) {
  const veh = row.vehicle || {};
  const unread = veh.unreadable_reason || "";
  const gemma = row.gemma || {};
  const gemmaText = gemma.plate_text
    ? `YOLO+Gemma: ${gemma.plate_text}`
    : gemma.skipped
      ? `YOLO+Gemma: ${gemma.skipped}`
      : gemma.called
        ? "YOLO+Gemma: empty"
        : "";
  const vo = row.vision_only || {};
  const enhancement = row.enhancement || {};
  const confirmation = row.confirmation || {};
  const recognition = row.recognition || {};
  const enhancementText = enhancement.enabled
    ? `${enhancement.method || "OpenCV"} Â· ${enhancement.profile || "vision"} Â· ${enhancement.view_count || 1} view(s)`
    : enhancement.method === "none" ? "off" : "";
  const voModels = vo.models || {};
  const voLines = Object.keys(voModels)
    .map((k) => {
      const m = voModels[k] || {};
      const t = m.plate_text || (m.skipped ? m.skipped : "empty");
      return `Vision-only ${m.model || k}: ${t}`;
    })
    .join(" · ");
  const number = veh.number || row.plate_norm || row.plate || (unread ? `unreadable (${unread})` : "—");
  const type = veh.type || row.vehicle_type || "unknown";
  const color = veh.color || row.vehicle_color || "—";
  const when = row.observed_at_ist || row.source_time_ist || formatWhen(row.observed_at || row.source_time);
  const cam = row.camera_id || "—";
  const loc = locationText(row);
  return `<div class="card">
    <div class="vehicle-grid">
      <div><span class="k">Vehicle number</span><b>${number}</b></div>
      <div><span class="k">Vehicle type</span>${type}</div>
      <div><span class="k">Colour</span>${color}</div>
      <div><span class="k">Date / time (IST)</span>${when}</div>
      <div><span class="k">Camera</span>${cam}</div>
      <div><span class="k">Location</span>${loc}</div>
      ${gemmaText ? `<div><span class="k">YOLO+Gemma</span>${gemmaText}</div>` : ""}
      ${voLines ? `<div><span class="k">Vision-only</span>${voLines}</div>` : ""}
      ${enhancementText ? `<div><span class="k">Image enhancement</span>${enhancementText}</div>` : ""}
      <div><span class="k">Plate decision</span>${confirmation.status || "review"} · ${confirmation.support_count || 0}/${confirmation.required_frames || 2} frames</div>
      ${recognition.reader_agreement ? `<div><span class="k">Reader agreement</span>${recognition.reader_agreement}</div>` : ""}
    </div>
    ${extraHtml || ""}
  </div>`;
}

async function loadLiveVehicles() {
  const el = document.getElementById("liveVehicles");
  if (!el) return;
  const end = new Date();
  const start = new Date(end.getTime() - 24 * 60 * 60 * 1000);
  const data = await j(`/api/investigations/vehicles?${new URLSearchParams({ start: start.toISOString(), end: end.toISOString(), limit: "20", sort: "desc" })}`);
  const rows = data.observations || [];
  if (!rows.length) {
    el.className = "card muted";
    el.textContent = "No vehicle observations in the last 24 hours.";
    return;
  }
  el.className = "";
  el.innerHTML = rows.map(vehicleObservationCard).join("");
}

// Three visually distinct states. "estimated" must never look like a result,
// and "unknown" is a decision the system made, not a missing value.
function stateBadge(state) {
  const label = { verified: "VERIFIED", estimated: "ESTIMATED", unknown: "UNKNOWN" }[state] || "UNKNOWN";
  return `<span class="attr-badge attr-${state || "unknown"}">${label}</span>`;
}

function attributeCell(label, value, state, detail) {
  const shown = state === "unknown" ? "unknown" : value || "unknown";
  return `<div class="attr-row attr-row-${state || "unknown"}">
    <span class="k">${label}</span>
    <b>${shown}</b> ${stateBadge(state)}
    <span class="muted">${detail}</span>
  </div>`;
}

// This is the DEVELOPER card. It shows what the model actually produced, which
// is often NOT what production displays: the production console demotes a type
// to Unknown while the deployment gate is closed, and demotes a colour whose
// source is not the deterministic model. Rendering `type_state`/`color_state`
// here printed "unknown / abstained" over rows that carry a real prediction --
// false, and useless for deciding what to label. So the raw stored value leads,
// and the production verdict is reported beside it.
function rawAttribute(row, attribute) {
  const raw = row.raw || {};
  const value = raw[`vehicle_${attribute}`] ?? row[`vehicle_${attribute}`] ?? "unknown";
  const state = raw[`${attribute}_state`]
    || (row[`${attribute}_verified`] ? "verified" : value && value !== "unknown" ? "estimated" : "unknown");
  return {
    value,
    state,
    confidence: Number(raw[`${attribute}_confidence`] ?? row[`${attribute}_confidence`] ?? 0),
    source: raw[`${attribute}_source`] || row[`${attribute}_source`] || "",
    // Non-empty only when production refuses to show this value.
    demoted: row[`${attribute}_state`] !== state ? (row[`${attribute}_note`] || "hidden in production") : "",
  };
}

function attributeDetail(row, attribute, attr) {
  if (attr.state === "verified") return `by ${row.verified_by || "operator"}`;
  const parts = [`conf ${attr.confidence.toFixed(2)}`];
  if (attr.source) parts.push(attr.source);
  const reason = row[`${attribute}_reason`];
  if (attr.state === "unknown" && reason) parts.push(`abstained · ${String(reason).replaceAll("_", " ")}`);
  if (attribute === "type" && row.type_candidate) {
    parts.push(`candidate "${row.type_candidate}" (${Number(row.type_candidate_confidence || 0).toFixed(2)})`);
  }
  if (attr.demoted) parts.push(`⚠ production shows Unknown — ${attr.demoted}`);
  return parts.join(" · ");
}

function vehicleObservationCard(row) {
  const image = row.evidence_path ? `<img src="${evidenceUrl(row.evidence_path)}" alt="vehicle evidence" />` : "";
  const type = rawAttribute(row, "type");
  const colour = rawAttribute(row, "color");
  // plate_state (not plate_status) is what the API emits; the old name meant
  // this line never rendered at all.
  const plateShown = row.plate_state && row.plate_state !== "ok"
    ? row.plate_state_text || String(row.plate_state).replaceAll("_", " ")
    : "";

  return `<article class="card observation-card" data-observation-card="${row.id}">${image}
    <div class="vehicle-grid">
      ${attributeCell("Vehicle type", type.value, type.state, attributeDetail(row, "type", type))}
      ${attributeCell("Colour", colour.value, colour.state, attributeDetail(row, "color", colour))}
      <div><span class="k">Camera</span>${escapeText(row.camera_id)} ${row.city ? `· ${escapeText(row.city)}` : ""}</div>
      <div><span class="k">Observed</span>${row.first_seen_at_ist || formatWhen(row.first_seen_at)}</div>
      ${plateShown
        ? `<div><span class="k">Plate</span><span class="muted">${escapeText(plateShown)} — vehicle record kept</span></div>`
        : row.plate_text
          ? `<div><span class="k">Plate</span><b>${escapeText(row.plate_text)}</b></div>`
          : ""}
    </div>
    ${reviewControls(row)}
  </article>`;
}

// Manual correction. The model's raw output is preserved server-side, so a
// correction is always auditable against what the model actually said.
//
// Three states per attribute, not two. The old control offered "— leave —" and
// a value, which made "leave it alone" and "clear the verdict" the same thing,
// so a verdict once recorded could never be undone from the UI even though the
// API has supported it all along (send "" to clear).
const REVIEW_CLEAR = "__clear__";

function reviewControls(row) {
  const types = (window.__vehicleOptions?.vehicle_types || []);
  const colors = (window.__vehicleOptions?.vehicle_colors || []);
  const opt = (list, selected) => list
    .map((v) => `<option value="${v}"${v === selected ? " selected" : ""}>${v.replaceAll("_", " ")}</option>`)
    .join("");
  // Preselect the verdict that is actually stored. The old code read
  // metadata.attributes.verified_type, which nothing has ever written, so the
  // dropdowns always opened blank even on rows already labelled.
  const clearOption = (current) => current
    ? `<option value="${REVIEW_CLEAR}">— clear verdict —</option>`
    : "";
  return `<details class="review-box"${row.review_status === "verified" ? " open" : ""}>
    <summary>Review / correct${row.review_status === "verified" ? " · verified" : ""}</summary>
    <div class="review-grid">
      <label>Type
        <select data-review="type" data-id="${row.id}">
          <option value="">— leave —</option>${clearOption(row.verified_vehicle_type)}${opt(types, row.verified_vehicle_type)}
        </select>
      </label>
      <label>Colour
        <select data-review="color" data-id="${row.id}">
          <option value="">— leave —</option>${clearOption(row.verified_vehicle_color)}${opt(colors, row.verified_vehicle_color)}
        </select>
      </label>
      <button data-review-save="${row.id}">Save verdict</button>
      <span class="muted" data-review-msg="${row.id}"></span>
    </div>
  </details>`;
}

async function saveReview(id) {
  const read = (attribute) => document.querySelector(`[data-review="${attribute}"][data-id="${id}"]`)?.value ?? "";
  const msg = document.querySelector(`[data-review-msg="${id}"]`);
  const body = {};
  // "" means leave alone (omit the field); REVIEW_CLEAR means clear it (send "").
  for (const [attribute, field] of [["type", "vehicle_type"], ["color", "vehicle_color"]]) {
    const value = read(attribute);
    if (value === REVIEW_CLEAR) body[field] = "";
    else if (value) body[field] = value;
  }
  if (!Object.keys(body).length) {
    if (msg) msg.textContent = "nothing selected";
    return;
  }
  try {
    const res = await fetch(`/api/vehicle-observations/${id}/review`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${operatorToken()}` },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(await res.text());
    // Re-render from what the server actually stored, rather than leaving the
    // card showing pre-review state until the next poll.
    const updated = await res.json();
    const card = document.querySelector(`[data-observation-card="${id}"]`);
    if (card) card.outerHTML = vehicleObservationCard(updated);
    const after = document.querySelector(`[data-review-msg="${id}"]`);
    if (after) {
      after.textContent = updated.review_status === "verified"
        ? "saved — raw model output preserved"
        : "verdict cleared — back to the model's value";
    }
  } catch (err) {
    if (msg) msg.textContent = `failed: ${String(err).slice(0, 120)}`;
  }
}

function operatorToken() {
  return TOKEN;
}

// The header field is the only way to change it; it is remembered for the tab
// so a refresh does not send you back to the default.
function bindTokenField() {
  const field = document.getElementById("opToken");
  if (!field) return;
  field.value = TOKEN;
  field.addEventListener("change", () => {
    TOKEN = field.value.trim() || "p0-operator";
    sessionStorage.setItem(DEV_TOKEN_KEY, TOKEN);
    refresh();
  });
}

function setFilterNote(elementId, text) {
  const el = document.getElementById(elementId);
  if (el) el.textContent = text || "";
}

document.addEventListener("click", (event) => {
  const id = event.target?.dataset?.reviewSave;
  if (id) saveReview(id);
});

// Measured POC accuracy, rendered wherever attributes are shown. This is not
// optional chrome: the numbers are what make the attributes interpretable.
function limitationsPanel(accuracy) {
  if (!accuracy) return "";
  const t = accuracy.vehicle_type || {};
  const c = accuracy.vehicle_color || {};
  const pct = (v) => (v == null ? "—" : `${Math.round(v * 100)}%`);
  return `<div class="limitations">
    <h4>POC accuracy — measured, not estimated</h4>
    <table class="limits-table">
      <tr><th></th><th>Coverage</th><th>Precision when answered</th><th>Tracks</th></tr>
      <tr class="limits-bad"><td>Vehicle type</td><td>${pct(t.coverage)}</td><td><b>${pct(t.selective_precision)}</b></td><td>${t.scored_tracks ?? "—"}</td></tr>
      <tr><td>Vehicle colour</td><td>${pct(c.coverage)}</td><td><b>${pct(c.selective_precision)}</b></td><td>${c.scored_tracks ?? "—"}</td></tr>
    </table>
    <p class="muted">Measured on ${accuracy.labelled_tracks} blind-labelled tracks from ${accuracy.camera}
      (${accuracy.excluded_unclear} too dark to label, excluded).</p>
    <ul>${(accuracy.notes || []).map((n) => `<li>${n}</li>`).join("")}</ul>
  </div>`;
}

function datetimeLocalValue(date) {
  const pad = (n) => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

function setInvestigationWindow(hours) {
  const end = new Date();
  const start = new Date(end.getTime() - hours * 60 * 60 * 1000);
  document.getElementById("invStart").value = datetimeLocalValue(start);
  document.getElementById("invEnd").value = datetimeLocalValue(end);
}

async function loadInvestigationOptions() {
  const data = await j("/api/vehicle-observation-options");
  const type = document.getElementById("invType");
  const color = document.getElementById("invColor");
  const camera = document.getElementById("invCamera");
  if (!type || !color || !camera) return;
  // Values stay filterable so historic rows remain searchable, but anything no
  // model on this host can predict is marked so an operator is not misled into
  // thinking the system can find, say, an SUV or a silver car.
  const opt = (o) => {
    const value = o.value !== undefined ? o.value : o;
    const supported = o.supported !== undefined ? o.supported : true;
    const label = String(value).replaceAll("_", " ");
    return `<option value="${value}">${label}${supported ? "" : " (not supported)"}</option>`;
  };
  window.__vehicleOptions = data;
  type.innerHTML = '<option value="">Any type (optional)</option>' + (data.type_options || data.vehicle_types || []).map(opt).join("");
  color.innerHTML = '<option value="">Any colour (optional)</option>' + (data.color_options || data.vehicle_colors || []).map(opt).join("");
  // Neither filter is ever mandatory, and both carry their caveat next to the
  // control rather than buried in a footnote.
  setFilterNote("invTypeNote", data.type_filter?.warning);
  setFilterNote("invColorNote", data.color_filter?.warning);
  const limits = document.getElementById("attributeLimitations");
  if (limits && data.accuracy) limits.innerHTML = limitationsPanel(data.accuracy);
  camera.innerHTML = '<option value="">Any camera</option>' + (data.cameras || []).map((c) => `<option value="${c.id}">${c.id} · ${c.name || c.city || ""}</option>`).join("");
}

async function searchVehicleObservations() {
  const start = document.getElementById("invStart").value;
  const end = document.getElementById("invEnd").value;
  if (!start || !end) return;
  const q = new URLSearchParams({ start: new Date(start).toISOString(), end: new Date(end).toISOString(), limit: "100", sort: "asc" });
  const type = document.getElementById("invType").value;
  const color = document.getElementById("invColor").value;
  const camera = document.getElementById("invCamera").value;
  const conf = document.getElementById("invConfidence").value;
  if (type) q.set("vehicle_type", type);
  if (color) q.set("vehicle_color", color);
  if (camera) q.set("camera_id", camera);
  if (conf) q.set("min_confidence", conf);
  const data = await j(`/api/investigations/vehicles?${q}`);
  const out = document.getElementById("investigateOut");
  const results = document.getElementById("investigationResults");
  out.className = "card";
  // Every caveat the server attached to THIS query is shown with the results,
  // not tucked away, because a filtered list reads as authoritative otherwise.
  const warnings = (data.search_warnings || [])
    .map((w) => `<div class="search-warning">⚠ ${w}</div>`)
    .join("");
  out.innerHTML =
    `<b>${data.total || 0} matching observation(s)</b> · ${(Object.keys(data.camera_counts || {}).length)} camera(s)`
    + `<br><span class="muted">${data.disclaimer}</span>`
    + warnings
    + limitationsPanel(data.accuracy);
  results.innerHTML = (data.observations || []).map(vehicleObservationCard).join("") || '<div class="card muted">No matching observations in this time range.</div>';
  markers.clearLayers(); links.clearLayers(); routes.clearLayers();
  const points = [];
  (data.observations || []).forEach((row) => {
    if (row.lat == null || row.lng == null) return;
    points.push([row.lat, row.lng]);
    L.circleMarker([row.lat, row.lng], { radius: 7, color: "#2c5282", fillOpacity: 0.85 })
      .bindPopup(`${row.vehicle_type} · ${row.vehicle_color}<br>${row.camera_id}<br>${row.first_seen_at_ist || row.first_seen_at}`)
      .addTo(markers);
  });
  if (points.length) map.fitBounds(points, { padding: [36, 36] });
  document.getElementById("dlInvestigationCsv").href = `/api/investigations/vehicles/export.csv?${q}&token=${encodeURIComponent(TOKEN)}`;
}

async function loadPlateRecognitionSettings() {
  const el = document.getElementById("plateRecognitionSettings");
  if (!el) return;
  const data = await j("/api/settings/recognition");
  el.className = "card";
  el.innerHTML = `<b>Number-plate recognition: ${data.enabled ? "ON" : "OFF"}</b><br><span class="muted">${data.effective_camera_count || 0} camera(s) currently effective. When off, vehicle type and colour detection continues and no new plate alerts are created.</span>`;
  const button = document.getElementById("btnTogglePlateRecognition");
  if (button) button.textContent = data.enabled ? "Turn number-plate recognition off" : "Turn number-plate recognition on";
  return data;
}

async function loadLiveSightings() {
  const el = document.getElementById("liveSightings");
  if (!el) return;
  const rows = await j("/api/sightings?limit=25");
  if (!rows.length) {
    el.className = "card muted";
    el.textContent = "No persisted sightings yet.";
    return;
  }
  el.className = "";
  el.innerHTML = rows
    .slice()
    .reverse()
    .map(
      (s) =>
        `<div class="card"><b>${s.plate_raw || s.plate_norm}</b> · ${s.camera_id} · ${s.syntax_ok ? "syntax ok" : "not a valid plate"}<div class="muted">${s.source_time_ist || formatWhen(s.source_time)} · ${s.model_id || ""} · conf ${(s.confidence || 0).toFixed(2)}</div></div>`
    )
    .join("");
}

async function loadRecognitionDiagnostics() {
  const summaryEl = document.getElementById("recognitionSummary");
  const rowsEl = document.getElementById("recognitionAttempts");
  if (!summaryEl || !rowsEl) return;
  const data = await j("/api/recognition/diagnostics?limit=30");
  const summary = data.summary || {};
  const reasons = summary.reason_counts || {};
  summaryEl.className = "card";
  summaryEl.textContent = `${summary.attempt_count || 0} attempts · reasons ${Object.entries(reasons).map(([k, v]) => `${k}:${v}`).join(", ") || "none"}`;
  rowsEl.innerHTML = (data.attempts || []).map((a) => {
    const isLocalizedPlate = a.detector === "fast_alpr" && a.reason_code !== "no_plate_localized";
    const crop = [
      isLocalizedPlate && a.evidence_path ? `<figure><figcaption>Native plate crop</figcaption><img src="${evidenceUrl(a.evidence_path)}" alt="native plate crop" /></figure>` : "",
      a.context_evidence_path ? `<figure><figcaption>${isLocalizedPlate ? "Detector context" : "Vehicle context — no plate localized"}</figcaption><img src="${evidenceUrl(a.context_evidence_path)}" alt="boxed detector context" /></figure>` : "",
    ].join("");
    const confs = (a.character_confidences || []).map((v) => Number(v).toFixed(2)).join(" ");
    const verdict = a.accepted ? "accepted candidate" : (a.syntax_ok ? "review (syntax ok)" : "review/rejected");
    return `<div class="card">${crop}<b>${a.plate_norm || "empty"}</b> · ${a.reason_code} · ${a.camera_id}<div class="muted">track ${a.track_id || "—"} · ${a.detector || "—"} / ${a.recognizer || "—"} · conf ${Number(a.confidence || 0).toFixed(2)} · ${Number(a.latency_ms || 0).toFixed(1)} ms<br>native ${(a.native_size || []).join("×") || "—"} · chars ${confs || "—"} · ${verdict}</div></div>`;
  }).join("") || '<div class="card muted">No candidate attempts have been persisted.</div>';
}

// Watchlist purposes and authorities are operator-entered free text that is
// interpolated into innerHTML below, so it is escaped rather than trusted.
function escapeText(value) {
  return String(value == null ? "" : value).replace(/[&<>"']/g, (ch) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]
  ));
}

// Priority comes from the watchlist row that produced the match. An alert whose
// watchlist row carries no priority is labelled "unset", never silently ranked.
function priorityBadge(priority) {
  const p = (priority || "").toLowerCase();
  const label = { high: "HIGH", medium: "MEDIUM", low: "LOW" }[p] || "PRIORITY UNSET";
  return `<span class="prio-badge prio-badge-${p || "unset"}">${label}</span>`;
}

// The alert card reuses the older renderer, which prints vehicle type and colour
// without the verified/estimated badges the observation cards carry. These
// values may also be historical `legacy_sighting` rows holding classes the
// deployed model cannot support, so the card states what they are worth.
function alertAttributeNote(a) {
  const type = (a.vehicle_type || "").trim();
  const color = (a.vehicle_color || "").trim();
  if (!type && !color) return "";
  return `<div class="muted" style="margin-top:4px;font-size:11px">
    Type/colour shown are unverified model estimates recorded with this sighting —
    not confirmation of vehicle identity. The plate match is what triggered this alert.
  </div>`;
}

async function loadAlerts() {
  await loadLiveVehicles();
  await loadLiveSightings();
  const alerts = await j("/api/alerts");
  const root = document.getElementById("alerts");
  root.innerHTML = "";
  if (!alerts.length) {
    root.innerHTML = `<div class="card muted">No alerts yet. Run own-feed analysis on a watchlist plate. Alerts are never hardcoded.</div>`;
    return;
  }
  alerts.forEach((a) => {
    const img = a.evidence_path ? `<img src="${evidenceUrl(a.evidence_path)}" alt="crop" />` : "";
    const wrap = document.createElement("div");
    wrap.innerHTML = vehicleRowCard(
      {
        vehicle: {
          number: a.plate_norm,
          type: a.vehicle_type || "unknown",
          color: a.vehicle_color || "—",
        },
        observed_at_ist: a.source_time_ist || a.created_at_ist,
        camera_id: a.camera_id,
        location: a.location || a.city || a.camera_name,
        lat: a.lat,
        lng: a.lng,
      },
      `${img}
      <div class="row" style="margin-top:8px;align-items:center;gap:8px">
        ${priorityBadge(a.priority)}
        <span class="muted">${escapeText(a.purpose || "watchlist match")} · ${escapeText(a.status)} · ${escapeText(a.match_type)}</span>
      </div>
      ${alertAttributeNote(a)}
      <div class="row" style="margin-top:6px">
        <button data-id="${a.id}" data-s="acknowledged">Ack</button>
        <button class="secondary" data-id="${a.id}" data-s="confirmed">Confirm</button>
        <button class="secondary" data-id="${a.id}" data-s="rejected">Reject</button>
      </div>`
    );
    const card = wrap.firstElementChild;
    card.classList.add(`prio-${(a.priority || "unset").toLowerCase()}`);
    root.appendChild(card);
  });
  root.querySelectorAll("button[data-id]").forEach((btn) => {
    btn.onclick = async () => {
      await j(`/api/alerts/${btn.dataset.id}`, {
        method: "PATCH",
        body: JSON.stringify({ status: btn.dataset.s }),
      });
      loadAlerts();
    };
  });
}

async function loadWorkers() {
  const cap = await j("/api/capacity");
  const capEl = document.getElementById("capacityPanel");
  if (capEl) {
    capEl.innerHTML = `measured ${cap.measured_safe_fps || "—"} fps · recommended ${cap.recommended_target_fps || "—"} fps · gov decode ok ${cap.government_decode_ok_count || "—"}/${cap.government_decode_tested_count || "—"} · max captures ${cap.max_concurrent_captures}`;
  }
  const w = await j("/api/workers");
  document.getElementById("workerPanel").innerHTML = `
    <div>max ${w.max_concurrent} · running ${w.running_count} · queued ${w.queued_count} · open captures ${w.open_captures} · previews ${w.preview_count}</div>
    <div class="muted">queued: ${(w.queued || []).join(", ") || "none"}</div>
    ${(w.workers || [])
      .map((s) => `<div>${s.camera_id} · ${s.status} · frames ${s.frames} · pts ${s.last_pts_ms ?? "—"} · reconnect ${s.reconnect_attempt}</div>`)
      .join("") || "<div class='muted'>No analytics workers running.</div>"}
  `;
}

async function plotVehicle(data) {
  links.clearLayers();
  routes.clearLayers();
  if (data.inferred_links) {
    data.inferred_links.forEach((l) => {
      L.polyline(
        [
          [l.from[1], l.from[0]],
          [l.to[1], l.to[0]],
        ],
        { color: "#c4a35a", dashArray: "8 8", weight: 3 }
      )
        .bindPopup(l.label)
        .addTo(links);
    });
  }
  (data.possible_routes || []).forEach((r) => {
    if (!r.path || r.path.length < 2) return;
    const fallback = r.provider === "fallback_straight";
    L.polyline(r.path, {
      color: fallback ? "#2c5282" : "#1b365d",
      weight: 4,
      opacity: 0.8,
      dashArray: fallback ? "4 8" : null,
    })
      .bindPopup(
        `${r.label || "OSM map-matched possible path, not a verified route"}<br>${r.from_camera || ""} → ${r.to_camera || ""}<br>${r.provider || ""}`
      )
      .addTo(routes);
  });
  const pts = (data.sightings || []).filter((s) => s.lat != null).map((s) => [s.lat, s.lng]);
  (data.sightings || []).forEach((s, i) => {
    if (s.lat == null) return;
    L.circleMarker([s.lat, s.lng], { radius: 8, color: "#1b365d", fillOpacity: 0.9 })
      .bindPopup(`#${i + 1} ${s.camera_id}<br>${s.source_time || ""}<br>${s.plate_raw || s.plate_norm}`)
      .addTo(links);
  });
  if (pts.length) map.fitBounds(pts, { padding: [40, 40] });
}

async function searchPlate() {
  const plate = document.getElementById("plateQuery").value;
  const data = await j(`/api/vehicles/${encodeURIComponent(plate)}`);
  links.clearLayers();
  const hist = document.getElementById("history");
  if (!data.sightings.length) {
    hist.innerHTML = `<div class="card muted">No persisted sightings for ${data.plate_norm}.</div>`;
    return;
  }
  hist.innerHTML = data.sightings
    .map(
      (s) =>
        `<div class="card"><b>${s.camera_id}</b> · ${s.city || ""} · ${s.department || ""}<div class="muted">${s.source_time}<br>ingest ${s.ingest_time || "—"} · pts ${s.source_pts_ms ?? "—"}<br>raw ${s.plate_raw} → ${s.plate_voted || s.plate_norm} · ${s.model_id}</div></div>`
    )
    .join("");
  plotVehicle(data);
}

async function refresh() {
  await loadCoverage();
  await loadCameras();
  await loadAlerts();
  await loadWorkers();
  await loadWatchlist();
  await loadRecognitionDiagnostics();
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.onclick = () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("on"));
    tab.classList.add("on");
    document.querySelectorAll("[data-panel]").forEach((p) => {
      p.classList.toggle("hidden", p.dataset.panel !== tab.dataset.tab);
    });
  };
});

document.getElementById("btnRefresh").onclick = refresh;
document.getElementById("btnSearch").onclick = searchPlate;
document.getElementById("ledgerFilter").onchange = renderLedger;
document.getElementById("btnMonitor").onclick = async () => {
  const plate = document.getElementById("monPlate").value.trim();
  const day = document.getElementById("monDay").value;
  const q = new URLSearchParams({ routes: "true" });
  if (day) q.set("day", day);
  const data = await j(`/api/vehicles/${encodeURIComponent(plate)}?${q}`);
  const st = document.getElementById("monitorStatus");
  st.className = "card";
  const matchNote = data.match_error
    ? data.match_error
    : `${data.map_match_provider || "osrm"} · ${(data.possible_routes || []).length} snapped path(s)`;
  st.innerHTML = `<b>${data.plate_norm}</b> · ${data.sightings.length} sighting(s)<br><span class="muted">${data.path_disclaimer || ""} ${matchNote}</span>`;
  document.getElementById("monitorTimeline").innerHTML = (data.sightings || [])
    .map(
      (s, i) =>
        `<div class="card"><b>#${i + 1} ${s.camera_id}</b> · ${s.city || ""}<div class="muted">${s.source_time}<br>raw ${s.plate_raw} · ${s.model_id}</div></div>`
    )
    .join("");
  plotVehicle(data);
  const token = encodeURIComponent(TOKEN);
  document.getElementById("dlDayCsv").href = `/api/vehicles/${encodeURIComponent(plate)}/export.csv?${q}&token=${token}`;
  document.getElementById("dlDayGeo").href = `/api/vehicles/${encodeURIComponent(plate)}/export.geojson?${q}&token=${token}`;
};
// (Removed: a second btnInvestigate handler that called /api/cameras/active-at.
//  It was overwritten unconditionally further down by the vehicle-first search,
//  so it never ran. The endpoint still exists and is still tested.)
document.getElementById("btnWatchAdd").onclick = async () => {
  const plate = document.getElementById("wlPlate").value.trim();
  if (!plate) return;
  const out = await j("/api/watchlist", {
    method: "POST",
    body: JSON.stringify({
      plate_raw: plate,
      purpose: document.getElementById("wlPurpose").value.trim() || "stolen_vehicle",
      priority: document.getElementById("wlPriority").value,
      rematch: true,
    }),
  });
  alert(`Watchlist ${out.plate_norm}. Rematch created ${out.rematch && out.rematch.alerts_created} alert(s).`);
  loadWatchlist();
  loadAlerts();
};
document.getElementById("btnClosePreview").onclick = closePreview;
document.getElementById("btnAnalyze").onclick = async () => {
  document.getElementById("btnAnalyze").disabled = true;
  document.getElementById("btnAnalyze").textContent = "Analyzing…";
  try {
    const out = await j("/api/analyze-active", { method: "POST" });
    alert(`Sightings from ${out.ran} source(s). Coverage: ${out.coverage.honest_coverage}`);
    await refresh();
    await searchPlate();
  } catch (err) {
    alert(String(err));
  } finally {
    document.getElementById("btnAnalyze").disabled = false;
    document.getElementById("btnAnalyze").textContent = "Run own-feed analysis";
  }
};
// ---- Model 1 onboarding: manual entry, bulk CSV import, gap report --------

function onboardValue(id) {
  const el = document.getElementById(id);
  return el ? el.value.trim() : "";
}

function renderImportResult(out) {
  const root = document.getElementById("importOut");
  if (!root) return;
  const errs = (out.errors || []).map(
    (e) => `<div class="err">row ${e.row}${e.id ? ` (${escapeText(e.id)})` : ""}: ${escapeText(e.error)}</div>`
  ).join("");
  root.innerHTML = `<div class="card">
    <b>${out.created_count} created · ${out.updated_count} updated · ${out.error_count} rejected</b>
    <div class="muted" style="margin-top:5px">${escapeText(out.note || "")}</div>
    ${errs}
  </div>`;
}

document.getElementById("btnOnboard").onclick = async () => {
  const msg = document.getElementById("onboardMsg");
  const id = onboardValue("onbId");
  if (!id) { msg.textContent = "Camera ID is required."; return; }
  const body = { id, source_type: onboardValue("onbSourceType") };
  const optional = {
    name: "onbName", department: "onbDept", city: "onbCity", source_uri: "onbUri",
  };
  Object.entries(optional).forEach(([field, el]) => {
    const v = onboardValue(el);
    if (v) body[field] = v;
  });
  // Latitude and longitude only travel as a pair; a half pair is a data error,
  // not something to fill in with a guess.
  const lat = onboardValue("onbLat"), lng = onboardValue("onbLng");
  if (lat && lng) { body.lat = Number(lat); body.lng = Number(lng); }
  else if (lat || lng) {
    msg.textContent = "Supply both latitude and longitude, or neither.";
    return;
  }
  msg.textContent = "Adding…";
  try {
    await j("/api/cameras", { method: "POST", body: JSON.stringify(body) });
    msg.textContent = `${id} onboarded — status untested until its stream is probed.`;
    ["onbId", "onbName", "onbDept", "onbCity", "onbLat", "onbLng", "onbUri"]
      .forEach((el) => { const n = document.getElementById(el); if (n) n.value = ""; });
    await refresh();
  } catch (err) {
    msg.textContent = String(err);
  }
};

document.getElementById("btnImportCsv").onclick = async () => {
  const csv = onboardValue("onbCsv");
  if (!csv) {
    document.getElementById("importOut").innerHTML =
      '<div class="card err">Paste CSV rows first.</div>';
    return;
  }
  try {
    renderImportResult(await j("/api/cameras/import", { method: "POST", body: JSON.stringify({ csv }) }));
    await refresh();
  } catch (err) {
    document.getElementById("importOut").innerHTML =
      `<div class="card err">${escapeText(String(err))}</div>`;
  }
};

document.getElementById("btnGapJson").onclick = () => {
  window.open("/api/reports/gap-analysis.json?token=" + encodeURIComponent(TOKEN), "_blank");
};
document.getElementById("btnGapCsv").onclick = () => {
  window.open("/api/reports/gap-analysis.csv?token=" + encodeURIComponent(TOKEN), "_blank");
};

document.getElementById("btnSync").onclick = async () => {
  try {
    const out = await j("/api/catalogue/sync", { method: "POST" });
    alert(out.ok ? `Catalogue cameras ${out.cameras}` : `Catalogue sync failed: ${out.error}`);
    await refresh();
  } catch (err) {
    alert(String(err));
  }
};
document.getElementById("btnMeasure").onclick = async () => {
  const btn = document.getElementById("btnMeasure");
  btn.disabled = true;
  btn.textContent = "Measuring…";
  try {
    const out = await j("/api/capacity/measure", { method: "POST" });
    alert(
      `This batch: decode ok ${out.decode_ok_count}/${out.tested_count}. ` +
        `Already ok ${(out.already_decode_ok || []).length}. ` +
        `Untested left ${out.catalogue_remaining_untested}. ` +
        `Failed left ${out.catalogue_remaining_failed ?? "—"}. ` +
        `${out.disclaimer}`
    );
    await refresh();
  } catch (err) {
    alert(String(err));
  } finally {
    btn.disabled = false;
    btn.textContent = "Measure government decode";
  }
};
async function startHunt() {
  const out = await j("/api/hunt/start", { method: "POST", body: JSON.stringify({ max_concurrent: concurrentFromUi() }) });
  alert(out.disclaimer || out.label || `Hunt started. Hunting ${(out.hunting || []).length}/${out.total || 0}.`);
  refresh();
}
async function pinHunt() {
  const out = await j("/api/hunt/pin", { method: "POST", body: JSON.stringify({ max_concurrent: concurrentFromUi() }) });
  alert(out.disclaimer || `Pinned ${(out.started || []).length} working cameras. Queued ${(out.queued || []).length}.`);
  refresh();
}
async function stopHunt() {
  await j("/api/hunt/stop", { method: "POST" });
  refresh();
}
document.getElementById("btnHuntStart").onclick = startHunt;
document.getElementById("btnHuntStop").onclick = stopHunt;
const huntPin = document.getElementById("btnHuntPin");
if (huntPin) huntPin.onclick = pinHunt;
const huntStart2 = document.getElementById("btnHuntStart2");
const huntStop2 = document.getElementById("btnHuntStop2");
if (huntStart2) huntStart2.onclick = startHunt;
if (huntStop2) huntStop2.onclick = stopHunt;
const huntPin2 = document.getElementById("btnHuntPin2");
if (huntPin2) huntPin2.onclick = pinHunt;
document.querySelectorAll(".js-max-concurrent").forEach((el) => {
  el.addEventListener("change", () => {
    const n = concurrentFromUi();
    document.querySelectorAll(".js-max-concurrent").forEach((other) => {
      other.value = String(n);
    });
  });
});
document.getElementById("btnStartAccessible").onclick = async () => {
  const out = await j("/api/workers/start-accessible", {
    method: "POST",
    body: JSON.stringify({ decode_ok_only: true }),
  });
  alert(`Started ${ (out.started || []).length }. Queued ${ (out.queued || []).length }. ${out.disclaimer || ""}`);
  refresh();
};
document.getElementById("btnStartWorker").onclick = async () => {
  const id = document.getElementById("workerCam").value.trim();
  if (!id) return;
  await j(`/api/workers/${id}/start`, { method: "POST" });
  refresh();
};
document.getElementById("btnStopWorker").onclick = async () => {
  const id = document.getElementById("workerCam").value.trim();
  if (!id) return;
  await j(`/api/workers/${id}/stop`, { method: "POST" });
  refresh();
};
document.getElementById("btnStopAll").onclick = async () => {
  await j("/api/workers/stop-all", { method: "POST" });
  refresh();
};
document.getElementById("btnCost").onclick = async () => {
  const out = await j("/api/cost/estimate", {
    method: "POST",
    body: JSON.stringify({
      camera_count: Number(document.getElementById("c_count").value),
      avg_bitrate_kbps: Number(document.getElementById("c_br").value),
      target_analysis_fps: Number(document.getElementById("c_fps").value),
      active_cameras: Number(document.getElementById("c_active").value),
      measured_worker_fps: Number(document.getElementById("c_wfps").value),
      gpu_hourly_cost: Number(document.getElementById("c_gpu").value),
      storage_cost_per_gb: Number(document.getElementById("c_sto").value),
      evidence_events_per_day: Number(document.getElementById("c_ev").value),
    }),
  });
  document.getElementById("costOut").textContent = JSON.stringify(out, null, 2);
};
// Vehicle-first investigation replaces the older analytics-active-only search.
document.getElementById("btnInvestigate").onclick = searchVehicleObservations;
document.querySelectorAll("[data-investigation-window]").forEach((button) => {
  button.onclick = () => {
    setInvestigationWindow(Number(button.dataset.investigationWindow || 24));
    searchVehicleObservations();
  };
});
document.getElementById("btnTogglePlateRecognition").onclick = async () => {
  const current = await loadPlateRecognitionSettings();
  const out = await j("/api/settings/recognition", { method: "PATCH", body: JSON.stringify({ enabled: !current.enabled }) });
  await loadPlateRecognitionSettings();
  await loadCoverage();
  alert(`Number-plate recognition is now ${out.enabled ? "ON" : "OFF"}.`);
};

document.getElementById("dlJson").onclick = (e) => {
  e.preventDefault();
  window.open("/api/reports/sightings.json?token=" + encodeURIComponent(TOKEN), "_blank");
};
document.getElementById("dlCsv").onclick = (e) => {
  e.preventDefault();
  window.open("/api/reports/sightings.csv?token=" + encodeURIComponent(TOKEN), "_blank");
};

bindTokenField();
setInvestigationWindow(24);
loadInvestigationOptions().then(searchVehicleObservations).catch(() => {});
loadPlateRecognitionSettings().catch(() => {});
refresh();
// loadAlerts() belongs in this loop: the watchlist requirement is *automated
// real-time* alerting, and an alert list that only moves when an operator
// clicks Refresh is not real-time. Every poll is caught individually so one
// failing endpoint cannot stop the others, and a sustained outage is shown
// rather than left as a silently frozen screen.
let pollFailures = 0;

function pollOnce() {
  const jobs = [
    loadCoverage(), loadWorkers(), loadLiveVehicles(),
    loadLiveSightings(), loadRecognitionDiagnostics(), loadAlerts(),
  ].map((p) => Promise.resolve(p).catch((e) => e));
  Promise.all(jobs).then((results) => {
    const failed = results.filter((r) => r instanceof Error).length;
    pollFailures = failed === results.length ? pollFailures + 1 : 0;
    setPollStatus(pollFailures);
  });
}

function setPollStatus(consecutive) {
  const el = document.getElementById("pollStatus");
  if (!el) return;
  if (consecutive >= 2) {
    el.textContent = `⚠ Backend unreachable — figures below are stale (${consecutive} failed refreshes).`;
    el.classList.remove("hidden");
  } else {
    el.classList.add("hidden");
  }
}

setInterval(pollOnce, 4000);
