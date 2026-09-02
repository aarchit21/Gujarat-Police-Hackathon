const TOKEN = "p0-operator";
const map = L.map("map").setView([22.3, 71.2], 7);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: "&copy; OpenStreetMap",
}).addTo(map);

const markers = L.layerGroup().addTo(map);
const links = L.layerGroup().addTo(map);
let previewCam = null;
let hlsPlayer = null;

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

function evidenceUrl(path) {
  if (!path) return "";
  return `/api/evidence?rel=${encodeURIComponent(path)}&token=${encodeURIComponent(TOKEN)}`;
}

async function loadCoverage() {
  const h = await j("/api/health");
  const db = h.database || {};
  document.getElementById("hostMeta").textContent =
    `DB ${db.type || "?"} · catalogue ${h.catalogue_host || "?"} · auth ${h.catalogue_auth_mode || "none"} · gov ${h.government_feed_status}`;
  document.getElementById("coverageBanner").textContent =
    `${h.honest_coverage}. Catalogue ${h.government_catalogue_count} (not a hardcoded 50). Own-feed ${h.own_feed_count}. Catalogue live ${h.catalogue_live_count} is not analytics-active. Last sync ${h.catalogue_synced_at || "never"}.`;
  document.getElementById("statsBar").innerHTML = [
    ["Onboarded", h.onboarded_count],
    ["Own-feed", h.own_feed_count],
    ["Gov catalogue", h.government_catalogue_count],
    ["Connected", h.connected_count],
    ["Analytics", h.analytics_active_count],
    ["Blocked/deferred", h.blocked_count],
    ["Queued", h.queued_count],
    ["Gov feed", h.government_feed_status],
    ["Database", db.type],
    ["PostGIS", db.postgis ? "yes" : "no"],
    ["Open captures", h.open_capture_count],
    ["Previews", h.preview_active_count],
    ["Catalogue live", h.catalogue_live_count],
    ["Decode ok", h.decode_ok_count],
    ["HTTP", h.catalogue_last_http_status || "—"],
    ["Review alerts", h.alerts_requiring_review],
  ]
    .map(([k, v]) => `<span><b>${k}</b> ${v}</span>`)
    .join("");
}

async function loadCameras() {
  const cams = await j("/api/cameras");
  markers.clearLayers();
  const body = document.getElementById("ledger");
  body.innerHTML = "";
  cams.forEach((c) => {
    const color = c.analytics_active ? "#2f6f4e" : c.status === "onboarded" || c.status === "connected" ? "#c4a35a" : "#9b2c2c";
    L.circleMarker([c.lat, c.lng], { radius: 6, color, fillOpacity: 0.85 })
      .bindPopup(
        `<b>${c.id}</b> · ${c.origin || ""}<br>${c.city} · ${c.department}<br>priority ${c.priority_class} · ${c.processing_mode} · worker ${c.worker_state || "idle"}<br>compute ${c.compute_target || "—"} · net ${c.network_class}<br>cat live ${c.catalogue_live} · decode ${c.decode_status}<br>codec ${c.codec || "?"} ${c.width || "?"}x${c.height || "?"}<br>protocol ${c.active_protocol || "—"} · pts ${c.last_pts_ms ?? "—"}<br>reconnects ${c.reconnect_count} · preview ${c.preview_active}<br>${c.status}: ${c.status_reason || ""}`
      )
      .addTo(markers);
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${c.id}</td><td>${c.origin || ""}</td><td>${c.priority_class}</td><td>${c.processing_mode}</td><td class="${statusClass(c.status)}">${c.status}</td><td>${c.catalogue_live}</td><td>${c.decode_status}</td><td>${c.active_protocol || "—"}</td><td>${c.analytics_active ? "yes" : "no"} / prev ${c.preview_active ? "yes" : "no"}</td>`;
    tr.onclick = () => showCamera(c);
    body.appendChild(tr);
  });
}

function showCamera(c) {
  document.getElementById("workerCam").value = c.id;
  document.getElementById("cameraDetail").innerHTML = `
    <div class="card">
      <div><b>${c.id}</b> · ${c.name}</div>
      <div class="muted">${c.department} · ${c.city || ""} · last frame ${c.last_frame_at || "—"}</div>
      <div class="muted">origin ${c.origin || "—"} · codec ${c.codec || "?"} ${c.width || "?"}x${c.height || "?"} · pts ${c.last_pts_ms ?? "—"} · reconnects ${c.reconnect_count}</div>
      <div class="muted">last error: ${c.last_error || "none"}</div>
      <div class="muted">${c.hls_preview_blocked ? "HLS preview blocked (credential stays server-side)." : ""}</div>
      <div class="row" style="margin-top:6px">
        <button data-prev="hls">HLS preview</button>
        <button class="secondary" data-prev="whep">WHEP preview</button>
        <button class="secondary" data-start="${c.id}">Start worker</button>
      </div>
    </div>`;
  document.getElementById("cameraDetail").querySelectorAll("[data-prev]").forEach((btn) => {
    btn.onclick = () => openPreview(c, btn.dataset.prev);
  });
  document.getElementById("cameraDetail").querySelector("[data-start]").onclick = async () => {
    await j(`/api/workers/${c.id}/start`, { method: "POST" });
    refresh();
  };
}

async function openPreview(c, protocol) {
  const out = await j(`/api/cameras/${c.id}/preview`, {
    method: "POST",
    body: JSON.stringify({ protocol }),
  });
  if (out.preview_blocked || !out.ok) {
    alert(out.error || "preview blocked");
    return;
  }
  previewCam = c.id;
  const box = document.getElementById("previewBox");
  box.classList.remove("hidden");
  document.getElementById("previewTitle").textContent = `${c.id} ${out.protocol} preview`;
  const video = document.getElementById("previewVideo");
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
  document.getElementById("previewBox").classList.add("hidden");
}

async function loadAlerts() {
  const alerts = await j("/api/alerts");
  const root = document.getElementById("alerts");
  root.innerHTML = "";
  if (!alerts.length) {
    root.innerHTML = `<div class="card muted">No alerts yet. Run own-feed analysis on a watchlist plate. Alerts are never hardcoded.</div>`;
    return;
  }
  alerts.forEach((a) => {
    const img = a.evidence_path ? `<img src="${evidenceUrl(a.evidence_path)}" alt="crop" />` : "";
    const el = document.createElement("div");
    el.className = "card";
    el.innerHTML = `
      <div><b>${a.plate_norm}</b> · ${a.match_type} · ${a.city || a.camera_id} · ${a.status}</div>
      <div class="muted">${a.source_time || ""} · raw ${a.plate_raw || "—"} · voted ${a.plate_voted || "—"} · conf ${(a.confidence || 0).toFixed(2)} · ${a.model_id || ""}</div>
      ${img}
      <div class="row" style="margin-top:6px">
        <button data-id="${a.id}" data-s="acknowledged">Ack</button>
        <button class="secondary" data-id="${a.id}" data-s="confirmed">Confirm</button>
        <button class="secondary" data-id="${a.id}" data-s="rejected">Reject</button>
      </div>`;
    root.appendChild(el);
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
  const w = await j("/api/workers");
  document.getElementById("workerPanel").innerHTML = `
    <div>max ${w.max_concurrent} · running ${w.running_count} · queued ${w.queued_count} · open captures ${w.open_captures} · previews ${w.preview_count}</div>
    <div class="muted">queued: ${(w.queued || []).join(", ") || "none"}</div>
    ${(w.workers || [])
      .map((s) => `<div>${s.camera_id} · ${s.status} · frames ${s.frames} · pts ${s.last_pts_ms ?? "—"} · reconnect ${s.reconnect_attempt}</div>`)
      .join("") || "<div class='muted'>No analytics workers running.</div>"}
  `;
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
  const pts = data.sightings.filter((s) => s.lat != null).map((s) => [s.lat, s.lng]);
  if (pts.length) map.fitBounds(pts, { padding: [40, 40] });
}

async function refresh() {
  await loadCoverage();
  await loadCameras();
  await loadAlerts();
  await loadWorkers();
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
