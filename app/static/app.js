const TOKEN = "p0-operator";
const map = L.map("map").setView([22.3, 71.2], 7);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: "&copy; OpenStreetMap",
}).addTo(map);

const markers = L.layerGroup().addTo(map);
const links = L.layerGroup().addTo(map);
const routes = L.layerGroup().addTo(map);
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
    ["Measured fps", h.measured_safe_fps || (h.capacity && h.capacity.measured_safe_fps) || "—"],
    ["Recommended fps", h.recommended_target_fps || (h.capacity && h.capacity.recommended_target_fps) || "—"],
    ["Database", db.type],
    ["PostGIS", db.postgis ? "yes" : "no"],
    ["Open captures", h.open_capture_count],
    ["Previews", h.preview_active_count],
    ["Catalogue live", h.catalogue_live_count],
    ["Decode ok", h.decode_ok_count],
    ["HTTP", h.catalogue_last_http_status || "—"],
    ["Review alerts", h.alerts_requiring_review],
    ["Snap-to-road", (h.map_match && h.map_match.provider) ? `OSM ${h.map_match.provider}` : "OSRM (free OSM)"],
  ]
    .map(([k, v]) => `<span><b>${k}</b> ${v}</span>`)
    .join("");
  const demo = document.getElementById("demoStrip");
  if (demo) {
    demo.textContent = `Demo · analytics ${h.analytics_active_count || 0} running · last sighting ${h.last_sighting_plate || "—"} @ ${h.last_sighting_camera || "—"} ${formatWhen(h.last_sighting_at)} · ${ollamaVisionText(h)}`;
  }
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
    const color = c.analytics_active ? "#2f6f4e" : c.catalogue_live ? "#2c5282" : c.status === "onboarded" || c.status === "connected" ? "#c4a35a" : "#9b2c2c";
    if (c.lat != null && c.lng != null && Number.isFinite(Number(c.lat)) && Number.isFinite(Number(c.lng))) {
      pts.push([c.lat, c.lng]);
      const marker = L.circleMarker([c.lat, c.lng], { radius: 7, color, fillOpacity: 0.85 })
        .bindPopup(
          `<b>${c.id}</b> · ${c.origin || ""}<br>${c.city || "location omitted"} · ${c.department}<br>${c.coords_are_placeholder ? "placeholder map position (catalogue omitted lat/lng)<br>" : ""}cat live ${c.catalogue_live} · decode ${c.decode_status}<br>${c.status}: ${c.status_reason || ""}`
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
      <div class="muted">${c.coords_are_placeholder ? "Map position is a placeholder — catalogue omitted lat/lng." : ""}</div>
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
  const number = veh.number || row.plate_norm || row.plate || "—";
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
    </div>
    ${extraHtml || ""}
  </div>`;
}

async function loadLiveVehicles() {
  const el = document.getElementById("liveVehicles");
  if (!el) return;
  const data = await j("/api/vehicle-events?limit=20&valid_only=true");
  const rows = Array.isArray(data) ? data : data.records || [];
  if (!rows.length) {
    el.className = "card muted";
    const reason = (data && data.empty_reason) || "No valid vehicle records yet.";
    const ollama = (data && data.ollama && data.ollama.label) || "";
    el.textContent = reason + (ollama ? ` ${ollama}` : "");
    return;
  }
  el.className = "";
  el.innerHTML = rows.slice().reverse().map((r) => vehicleRowCard(r)).join("");
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
      `${img}<div class="muted" style="margin-top:8px">${a.status} · ${a.match_type}</div>
      <div class="row" style="margin-top:6px">
        <button data-id="${a.id}" data-s="acknowledged">Ack</button>
        <button class="secondary" data-id="${a.id}" data-s="confirmed">Confirm</button>
        <button class="secondary" data-id="${a.id}" data-s="rejected">Reject</button>
      </div>`
    );
    root.appendChild(wrap.firstElementChild);
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
document.getElementById("btnInvestigate").onclick = async () => {
  const start = document.getElementById("invStart").value;
  const end = document.getElementById("invEnd").value;
  const q = new URLSearchParams();
  if (start) q.set("start", start);
  if (end) q.set("end", end);
  if (start && !end) q.set("at", start);
  const data = await j(`/api/cameras/active-at?${q}`);
  const el = document.getElementById("investigateOut");
  el.className = "card";
  el.innerHTML = `<div>${data.camera_count} camera(s) between ${data.from} and ${data.to}</div><div class="muted">${data.disclaimer}</div>` +
    (data.cameras || [])
      .map(
        (c) =>
          `<div><b>${c.id}</b> · ${c.origin} · sightings ${c.sightings_in_range} · ${c.analytics_was_active ? "analytics window" : "sighting only"}</div>`
      )
      .join("");
  markers.clearLayers();
  (data.cameras || []).forEach((c) => {
    if (c.lat == null) return;
    L.circleMarker([c.lat, c.lng], { radius: 8, color: "#2f6f4e", fillOpacity: 0.9 })
      .bindPopup(`${c.id}<br>analytics active in window`)
      .addTo(markers);
  });
};
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
    alert(`This batch: decode ok ${out.decode_ok_count}/${out.tested_count}. Already ok ${(out.already_decode_ok || []).length}. Remaining untested ${out.catalogue_remaining_untested}. Repeat Measure to test the next cameras. ${out.disclaimer}`);
    await refresh();
  } catch (err) {
    alert(String(err));
  } finally {
    btn.disabled = false;
    btn.textContent = "Measure government decode";
  }
};
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
document.getElementById("dlJson").onclick = (e) => {
  e.preventDefault();
  window.open("/api/reports/sightings.json?token=" + encodeURIComponent(TOKEN), "_blank");
};
document.getElementById("dlCsv").onclick = (e) => {
  e.preventDefault();
  window.open("/api/reports/sightings.csv?token=" + encodeURIComponent(TOKEN), "_blank");
};

refresh();
setInterval(() => {
  loadCoverage();
  loadWorkers();
  loadLiveVehicles();
  loadLiveSightings();
}, 4000);
