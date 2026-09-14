/* =========================================================================
   Vehicle Investigation Console
   =========================================================================
   Rules this file follows, because the consequences of breaking them are
   operational rather than cosmetic:

   * Nothing is displayed as a fact unless the server said it is one. Colour is
     an estimate, automatic vehicle type is suppressed, and a camera is never
     drawn as available until this host has actually opened its stream.
   * No counts are invented. Every figure on screen comes from an API response;
     when a value is missing it is shown as "—", not as zero.
   * Backend failures surface as a short sentence the operator can act on. The
     raw text of an error never reaches the screen -- it is logged server-side.
   * Every action that writes disables its control until the request settles, so
     a second click cannot double-submit.
   * Developer diagnostics live in the console at /dev and are not rendered or
     fetched here.
   ========================================================================= */

"use strict";

// --------------------------------------------------------------- session ---
// The access token is held for this browser tab only. It is never written into
// the page, a URL or a link: evidence images are fetched with the same header
// as every other request and shown from an object URL.
const SESSION_KEY = "gp-operator-token";
let token = sessionStorage.getItem(SESSION_KEY) || "";

const el = (id) => document.getElementById(id);
const escapeHtml = (value) =>
  String(value == null ? "" : value).replace(/[&<>"']/g, (ch) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

const FALLBACK_ERROR = "Could not complete that request. Please try again.";

async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(path, {
      ...options,
      headers: {
        Authorization: `Bearer ${token}`,
        ...(options.body ? { "Content-Type": "application/json" } : {}),
        ...(options.headers || {}),
      },
    });
  } catch (_networkError) {
    throw new ApiError("The server could not be reached.", 0);
  }
  if (response.status === 401) {
    signOut("Your session has ended. Sign in again to continue.");
    throw new ApiError("Not signed in.", 401);
  }
  if (!response.ok) {
    // The API returns either {"detail": "..."} for a rejected request or
    // {"error": "...", "reference": "..."} for an unexpected failure. Both are
    // written to be shown; anything else is replaced rather than echoed, so an
    // unexpected body can never put a stack trace on screen.
    let message = FALLBACK_ERROR;
    try {
      const body = await response.json();
      const detail = typeof body.detail === "string" ? body.detail : body.error;
      if (typeof detail === "string" && detail && detail.length < 200 && !detail.includes("Traceback")) {
        message = detail;
      }
      if (body.reference) message += ` (reference ${body.reference})`;
    } catch (_parseError) {
      /* keep the fallback */
    }
    throw new ApiError(message, response.status);
  }
  const type = response.headers.get("content-type") || "";
  return type.includes("application/json") ? response.json() : response.text();
}

function errorText(error) {
  return error instanceof ApiError ? error.message : FALLBACK_ERROR;
}

// ------------------------------------------------------------- feedback ----
let toastTimer = null;

function toast(message, kind = "ok") {
  const node = el("toast");
  node.textContent = message;
  node.className = `toast is-${kind}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => node.classList.add("hidden"), 5000);
}

/** Run an action with its own control disabled, so it cannot be double-fired. */
async function withBusy(button, action) {
  if (!button || button.disabled) return undefined;
  button.disabled = true;
  button.classList.add("is-busy");
  try {
    return await action();
  } finally {
    button.disabled = false;
    button.classList.remove("is-busy");
  }
}

/** Used before any change to a value a person is accountable for. */
function confirmAction(text, confirmLabel = "Confirm") {
  return new Promise((resolve) => {
    const dialog = el("confirmDialog");
    el("confirmText").textContent = text;
    el("confirmOk").textContent = confirmLabel;
    const finish = (value) => {
      dialog.close();
      el("confirmOk").onclick = null;
      el("confirmCancel").onclick = null;
      resolve(value);
    };
    el("confirmOk").onclick = () => finish(true);
    el("confirmCancel").onclick = () => finish(false);
    dialog.onclose = () => resolve(false);
    dialog.showModal();
  });
}

// States are rendered by one helper so loading, empty and failure always look
// the same wherever they appear.
function showLoading(node, rows = 3) {
  node.innerHTML = `<div aria-busy="true" aria-label="Loading">${
    '<div class="skeleton-row"></div>'.repeat(rows)}</div>`;
}

function showEmpty(node, title, detail = "") {
  node.innerHTML = `<div class="state"><strong>${escapeHtml(title)}</strong>${
    detail ? `<p>${escapeHtml(detail)}</p>` : ""}</div>`;
}

function showError(node, error, retryLabel) {
  node.innerHTML = `<div class="state state-error">
    <strong>${escapeHtml(errorText(error))}</strong>
    ${retryLabel ? `<button class="btn-secondary" data-retry>${escapeHtml(retryLabel)}</button>` : ""}
  </div>`;
}

// ---------------------------------------------------------------- format ---
// Every time on screen is Indian Standard Time in one shape: 2026-09-14 20:00.
// The server sends a pre-formatted IST label for most fields; the rest are UTC
// ISO strings converted here, so the two can never look like different clocks.
function formatTime(row, isoField, istField) {
  const ist = istField ? row[istField] : "";
  if (ist) return String(ist).replace(" IST", "").slice(0, 16);
  const iso = row[isoField];
  if (!iso) return "—";
  let text = String(iso);
  if (!/Z$|[+-]\d\d:\d\d$/.test(text)) text += "Z";
  const date = new Date(text);
  if (Number.isNaN(date.getTime())) return "—";
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Kolkata", year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit", hour12: false,
  }).formatToParts(date).reduce((acc, part) => ({ ...acc, [part.type]: part.value }), {});
  return `${parts.year}-${parts.month}-${parts.day} ${parts.hour}:${parts.minute}`;
}

// Mirrors the server's display names so a dropdown, a confirmation prompt and
// the record itself never spell the same value three different ways.
const DISPLAY_NAMES = { suv: "SUV", two_wheeler: "Two-wheeler", taxi_cab: "Taxi" };

function titleCase(value) {
  const raw = String(value || "").trim().toLowerCase();
  if (DISPLAY_NAMES[raw]) return DISPLAY_NAMES[raw];
  const text = raw.replace(/_/g, " ");
  return text ? text[0].toUpperCase() + text.slice(1) : "";
}

function cameraLabel(row) {
  const id = row.camera_id || row.id;
  const name = row.camera_name || row.name || cameraNames.get(id) || id;
  return name || "Unknown camera";
}

//: id -> human name, filled from the options endpoint at sign-in. Endpoints
//  that return only a camera id can then still show an operator a place name.
const cameraNames = new Map();

// Departments arrive both as slugs ("government-catalogue") and as acronyms
// ("RTO"). Lower-casing everything first turned RTO into "Rto", so existing
// capitalisation is left alone and only an all-lowercase slug is tidied up.
function prettyDepartment(value) {
  const text = String(value || "").replace(/[-_]/g, " ").trim();
  if (!text) return "";
  // Catalogue-synced cameras carry "government-catalogue" as their department:
  // a data-source name, not a department, and the feed tag already says it.
  if (text.toLowerCase() === "government catalogue") return "";
  if (text !== text.toLowerCase()) return text;
  return text[0].toUpperCase() + text.slice(1);
}

function placeLabel(row) {
  return [row.city, prettyDepartment(row.department)].filter(Boolean).join(" · ")
    || "Location not recorded";
}

// ------------------------------------------------------------------ feeds --
// Two sources, and they demonstrate different things. The own feed is
// SYNTHETIC — frames this host generated — so it is labelled a test feed
// everywhere it appears. An alert raised on it is a real alert from a real
// persisted sighting, but it is not a live enforcement hit, and the interface
// must not let anyone read it as one.
const FEED = {
  own: {
    label: "Own test feed",
    short: "Test feed",
    detail: "Frames generated on this host with known number plates. Used to prove the plate-to-alert path end to end.",
  },
  government: {
    label: "Government cameras",
    short: "Government",
    detail: "Authorised live camera streams. Used to detect, track and describe real vehicles.",
  },
};

function feedOf(row) {
  return row && row.feed === "own" ? "own" : "government";
}

// A camera registered by hand is neither the test feed nor a government
// catalogue camera. `feed` has two values because the SEARCH filter has two
// buckets; the label an operator reads comes from `origin`, which keeps the
// three cases apart. Calling a locally added camera "Government" would be a
// small lie printed on every row.
const ORIGIN_TAG = {
  own_feed: { key: "own", short: "Test feed" },
  government_catalogue: { key: "government", short: "Government" },
  local_registry: { key: "local", short: "Local registry" },
};

function feedTag(row) {
  const byOrigin = row && ORIGIN_TAG[row.origin];
  const key = byOrigin ? byOrigin.key : feedOf(row);
  const short = byOrigin ? byOrigin.short : FEED[key].short;
  return `<span class="feed-tag feed-tag-${key}">${escapeHtml(short)}</span>`;
}

// ----------------------------------------------------------------- badges --
/** verified | estimated | unknown, always with its word, never colour alone. */
function trustBadge(state) {
  const label = { verified: "Verified", estimated: "Estimated", unknown: "Unknown" }[state] || "Unknown";
  return `<span class="badge badge-${state || "unknown"}">${label}</span>`;
}

const PLATE_LABEL = {
  ok: "Estimated",
  plate_unreadable: "Plate unreadable",
  plate_not_visible: "Plate not visible",
  not_checked: "Not checked",
};

function plateCell(row) {
  const state = row.plate_state || "not_checked";
  if (state === "ok" && row.plate_text) {
    return `<span class="plate-text">${escapeHtml(row.plate_text)}</span>
            <span class="sub">${trustBadge("estimated")} not confirmed by a person</span>`;
  }
  const badgeKind = state === "not_checked" ? "unknown" : "unknown";
  return `<span class="badge badge-${badgeKind}">${escapeHtml(PLATE_LABEL[state] || "Unknown")}</span>
          ${row.plate_detail ? `<span class="sub">${escapeHtml(row.plate_detail)}</span>` : ""}`;
}

function reviewBadge(row) {
  if (row.review_status === "verified") {
    return `<span class="badge badge-verified">Reviewed</span>`;
  }
  return `<span class="badge badge-estimated">Review required</span>`;
}

// ------------------------------------------------------------- evidence ----
// Evidence is fetched with the session header and shown from an object URL, so
// no access token is ever placed in the DOM or in a link the browser may log.
const objectUrls = new Set();

function releaseImages() {
  objectUrls.forEach((url) => URL.revokeObjectURL(url));
  objectUrls.clear();
}

async function hydrateImages(root) {
  const targets = root.querySelectorAll("img[data-evidence]");
  await Promise.all(
    Array.from(targets).map(async (image) => {
      const rel = image.dataset.evidence;
      image.removeAttribute("data-evidence");
      try {
        const response = await fetch(`/api/evidence?rel=${encodeURIComponent(rel)}`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!response.ok) throw new Error("unavailable");
        const url = URL.createObjectURL(await response.blob());
        objectUrls.add(url);
        image.src = url;
      } catch (_error) {
        const placeholder = document.createElement("div");
        placeholder.className = image.className.replace("thumb", "thumb-empty") || "thumb-empty";
        placeholder.textContent = "Image unavailable";
        image.replaceWith(placeholder);
      }
    })
  );
}

function evidenceImg(path, alt, className = "thumb") {
  if (!path) return `<div class="thumb-empty">No image</div>`;
  return `<img class="${className}" alt="${escapeHtml(alt)}" data-evidence="${escapeHtml(path)}" />`;
}

// ================================================================= router ==
const PAGE_TITLES = {
  overview: ["Investigate", "Overview"],
  search: ["Investigate", "Vehicle search"],
  movement: ["Investigate", "Vehicle movement"],
  alerts: ["Enforcement", "Alerts"],
  watchlist: ["Enforcement", "Watchlist"],
  cameras: ["Cameras", "Camera status"],
};

let currentPage = "overview";

function showPage(page) {
  if (!PAGE_TITLES[page]) page = "overview";
  currentPage = page;
  document.querySelectorAll(".navitem").forEach((button) => {
    const on = button.dataset.page === page;
    button.classList.toggle("is-current", on);
    button.setAttribute("aria-selected", on ? "true" : "false");
  });
  document.querySelectorAll(".page").forEach((section) => {
    section.classList.toggle("hidden", section.dataset.page !== page);
  });
  const [crumb, title] = PAGE_TITLES[page];
  el("pageCrumb").textContent = crumb;
  el("pageTitle").textContent = title;
  document.title = `${title} — Vehicle Investigation Console`;
  el("content").scrollTo?.({ top: 0 });

  if (page === "cameras") loadCameras();
  if (page === "alerts") loadAlerts();
  if (page === "watchlist") loadWatchlist();
  if (page === "overview") loadOverview();
  // Leaflet mis-measures a container that was display:none when it was built.
  setTimeout(() => { maps.camera?.invalidateSize(); maps.movement?.invalidateSize(); }, 60);
}

// =================================================================== maps ==
const MAP_AVAILABLE = typeof L !== "undefined" && L && typeof L.map === "function";
const maps = { camera: null, movement: null };
const layers = { cameraMarkers: null, movementMarkers: null, movementLines: null };

function buildMap(containerId) {
  if (!MAP_AVAILABLE) {
    const host = el(containerId);
    if (host && !host.querySelector(".map-note")) {
      host.innerHTML = `<div class="map-note">Map library unavailable. Every list, search and
        review function still works; coordinates are included in CSV exports.</div>`;
    }
    return null;
  }
  const map = L.map(containerId, { attributionControl: true }).setView([22.3, 71.2], 7);
  const tiles = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: "&copy; OpenStreetMap",
    maxZoom: 18,
  }).addTo(map);
  let failures = 0;
  tiles.on("tileerror", () => {
    failures += 1;
    if (failures !== 6) return;
    const host = el(containerId);
    if (host && !host.querySelector(".map-note")) {
      const note = document.createElement("div");
      note.className = "map-note";
      note.textContent =
        "Background map unavailable offline. Camera and sighting positions below are still plotted correctly.";
      host.appendChild(note);
    }
  });
  return map;
}

function ensureCameraMap() {
  if (maps.camera || !MAP_AVAILABLE) return maps.camera;
  maps.camera = buildMap("cameraMap");
  layers.cameraMarkers = L.layerGroup().addTo(maps.camera);
  return maps.camera;
}

function ensureMovementMap() {
  if (maps.movement || !MAP_AVAILABLE) return maps.movement;
  maps.movement = buildMap("movementMap");
  layers.movementLines = L.layerGroup().addTo(maps.movement);
  layers.movementMarkers = L.layerGroup().addTo(maps.movement);
  return maps.movement;
}

// =============================================================== overview ==
const TILE_NONE = "—";

function tile(label, value, note, warn = false) {
  const isNone = value === null || value === undefined || value === "";
  return `<div class="tile">
    <span class="tile-label">${escapeHtml(label)}</span>
    <span class="tile-value${isNone ? " is-none" : ""}">${escapeHtml(isNone ? TILE_NONE : value)}</span>
    ${note ? `<span class="tile-note${warn ? " is-warn" : ""}">${escapeHtml(note)}</span>` : ""}
  </div>`;
}

let overviewLoaded = false;

async function loadOverview() {
  const tiles = el("overviewTiles");
  if (!overviewLoaded) showLoading(tiles, 2);
  try {
    const data = await api("/api/ui/overview");
    const cameras = data.cameras || {};
    const vehicles = data.vehicles || {};
    tiles.innerHTML = [
      tile("Cameras monitoring", cameras.monitoring,
        `${cameras.total ?? 0} camera${cameras.total === 1 ? "" : "s"} registered`),
      tile("Cameras available", cameras.available,
        cameras.not_checked
          ? `${cameras.not_checked} not checked yet`
          : "All registered cameras have been checked",
        Boolean(cameras.not_checked)),
      tile("Vehicles seen (24h)", vehicles.observations,
        data.last_observation_at_ist
          ? `Last at ${String(data.last_observation_at_ist).replace(" IST", "")}`
          : "No vehicles recorded in this window"),
      tile("Awaiting review", vehicles.awaiting_review,
        "Colour and type need a person to confirm them", Boolean(vehicles.awaiting_review)),
      tile("Open alerts", data.alerts_open,
        `${data.watchlist_active ?? 0} plate${data.watchlist_active === 1 ? "" : "s"} on the watchlist`,
        Boolean(data.alerts_open)),
    ].join("");

    const chip = el("plateChip");
    chip.textContent = `Plate reading: ${data.plate_recognition_enabled ? "on" : "off"}`;
    chip.className = `chip ${data.plate_recognition_enabled ? "is-on" : "is-off"}`;
    chip.title = data.plate_recognition_enabled
      ? "Number plates are read where the image allows it."
      : "Number plates are not being read. Vehicles are still detected and recorded.";

    const badge = el("navAlertCount");
    badge.textContent = String(data.alerts_open || "");
    badge.classList.toggle("hidden", !data.alerts_open);
    renderFeeds(data.feeds);
    overviewLoaded = true;
  } catch (error) {
    // Figures that were once correct are left standing; the banner above says
    // they are stale. Replacing them with an error would throw away the last
    // known state for no gain.
    if (!overviewLoaded) showError(tiles, error, "Try again");
    return;
  }
  loadOverviewRecent();
  loadOverviewAlerts();
}

// Both paths, side by side, each with its own counts and its own next action.
// Every figure comes from /api/ui/overview; a source that produced nothing says
// so rather than borrowing the other's numbers.
function renderFeeds(feeds) {
  const node = el("feedPanel");
  if (!feeds) { showEmpty(node, "Feed figures unavailable"); return; }
  const hours = feeds.window_hours || 24;

  const card = (key, actionsHtml) => {
    const f = feeds[key] || {};
    const c = f.cameras || {};
    const meta = FEED[key];
    const rows = [
      ["Cameras", c.total],
      ["Available now", c.available],
      key === "government" ? ["Unreachable", c.unreachable] : null,
      c.not_checked ? ["Not checked", c.not_checked] : null,
      ["Monitoring", c.monitoring],
      [`Vehicles (${hours}h)`, f.observations],
      [`Plates read (${hours}h)`, f.sightings],
      ["Open alerts", f.alerts_open],
    ].filter(Boolean);
    return `<article class="feedcard feedcard-${key}">
      <div class="feedcard-head">
        <h3>${escapeHtml(meta.label)}</h3>
        ${feedTag({ feed: key })}
      </div>
      <p class="card-meta">${escapeHtml(meta.detail)}</p>
      <dl class="feedstats">${rows.map(([label, value]) => `
        <div><dt>${escapeHtml(label)}</dt><dd>${value == null ? "—" : escapeHtml(value)}</dd></div>`).join("")}
      </dl>
      <div class="card-actions">${actionsHtml}</div>
    </article>`;
  };

  node.innerHTML =
    card("government", `
      <button class="btn-secondary" data-feed-action="check">Check connections</button>
      <button class="btn-secondary" data-feed-action="monitor">Start monitoring available</button>
      <button class="btn-quiet" data-feed-view="government">View vehicles</button>`)
    + card("own", `
      <button class="btn-secondary" data-feed-action="analyse">Analyse own feed</button>
      <button class="btn-quiet" data-feed-view="own">View vehicles</button>`);

  node.querySelectorAll("[data-feed-view]").forEach((button) => {
    button.addEventListener("click", () => {
      showPage("search");
      el("fFeed").value = button.dataset.feedView;
      setRange(24);
      withBusy(el("btnSearch"), () => runSearch());
    });
  });
  node.querySelector('[data-feed-action="check"]').addEventListener("click", (e) => checkConnections(e.currentTarget));
  node.querySelector('[data-feed-action="monitor"]').addEventListener("click", (e) => startAvailable(e.currentTarget));
  node.querySelector('[data-feed-action="analyse"]').addEventListener("click", (e) => analyseOwnFeed(e.currentTarget));
}

// `/api/analyze-active` runs exactly the own-feed sources (image_dir / file),
// so this is the whole own-feed demonstration in one action.
function analyseOwnFeed(button) {
  return withBusy(button, async () => {
    try {
      const result = await api("/api/analyze-active", { method: "POST" });
      const ran = result.ran || 0;
      toast(ran
        ? `Analysed ${ran} own-feed source${ran === 1 ? "" : "s"}. Any watchlist match appears under Alerts.`
        : "No own-feed sources are configured on this host.", ran ? "ok" : "warn");
      loadOverview();
      loadAlerts();
    } catch (error) {
      toast(errorText(error), "bad");
    }
  });
}

async function loadOverviewRecent() {
  const node = el("overviewRecent");
  showLoading(node, 3);
  const end = new Date();
  const start = new Date(end.getTime() - 24 * 3600 * 1000);
  try {
    const data = await api(`/api/investigations/vehicles?${new URLSearchParams({
      start: start.toISOString(), end: end.toISOString(), limit: "6", sort: "desc",
    })}`);
    const rows = data.observations || [];
    if (!rows.length) {
      showEmpty(node, "No vehicles recorded in the last 24 hours",
        "Start monitoring a camera, or analyse recorded footage from the Cameras page.");
      return;
    }
    // A compact list, not the results table: this panel is half the width of
    // the page and a squeezed eight-column table clips the trust badges, which
    // are the part that must never be hard to read.
    node.innerHTML = `<ul class="feed">${rows.map((row) => `
      <li class="feed-item is-clickable" data-observation="${row.id}" tabindex="0" role="button">
        ${evidenceImg(row.evidence_path, `Vehicle seen at ${cameraLabel(row)}`)}
        <div class="feed-text">
          <span class="feed-title">${escapeHtml(cameraLabel(row))}</span>
          <span class="sub">${escapeHtml(formatTime(row, "first_seen_at", "first_seen_at_ist"))} · ${
            escapeHtml(placeLabel(row))}</span>
          <span class="feed-tags">
            <span>${escapeHtml(row.color_display || "Unknown")} ${trustBadge(row.color_state)}</span>
            <span>${plateBadgeOnly(row)}</span>
          </span>
        </div>
      </li>`).join("")}</ul>`;
    wireObservationTable(node);
    await hydrateImages(node);
  } catch (error) {
    showError(node, error, "Try again");
  }
}

function plateBadgeOnly(row) {
  const state = row.plate_state || "not_checked";
  if (state === "ok" && row.plate_text) {
    return `<span class="plate-text">${escapeHtml(row.plate_text)}</span> ${trustBadge("estimated")}`;
  }
  return `<span class="badge badge-unknown">${escapeHtml(PLATE_LABEL[state] || "Unknown")}</span>`;
}

async function loadOverviewAlerts() {
  const node = el("overviewAlerts");
  showLoading(node, 2);
  try {
    const alerts = (await api("/api/alerts")).filter((a) => a.status === "new");
    if (!alerts.length) {
      showEmpty(node, "No open alerts", "Watchlist matches will appear here as soon as they are raised.");
      return;
    }
    node.innerHTML = `<div class="cards">${alerts.slice(0, 4).map(alertCard).join("")}</div>`;
    await hydrateImages(node);
    wireAlertActions(node);
  } catch (error) {
    showError(node, error, "Try again");
  }
}

// ================================================================= search ==
let options = { vehicle_types: [], vehicle_colors: [], cameras: [] };
let lastSearchQuery = null;
// One screenful at a time. Dropping 200 rows into the page buries the filters
// and pulls 200 evidence images the operator has not asked to see.
const PAGE_SIZE = 50;
let searchOffset = 0;
let searchRows = [];

function datetimeLocal(date) {
  const pad = (n) => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

function setRange(hours) {
  const end = new Date();
  el("fEnd").value = datetimeLocal(end);
  el("fStart").value = datetimeLocal(new Date(end.getTime() - hours * 3600 * 1000));
}

async function loadSearchOptions() {
  try {
    const data = await api("/api/vehicle-observation-options");
    options = data;
    (data.cameras || []).forEach((c) => cameraNames.set(c.id, c.name || c.id));
    const supportedTypes = new Set(data.supported_vehicle_types || []);
    const supportedColors = new Set(data.supported_vehicle_colors || []);

    // Every recorded value stays searchable so older records remain findable,
    // but anything this system cannot recognise on its own is labelled, rather
    // than being offered as though the system could find it for you.
    const render = (values, supported, suffix) =>
      values.filter((v) => v !== "unknown").map((v) =>
        `<option value="${escapeHtml(v)}">${escapeHtml(titleCase(v))}${
          supported.has(v) ? "" : ` ${suffix}`}</option>`).join("");

    el("fType").innerHTML = '<option value="">Any type</option>'
      + render(data.vehicle_types || [], supportedTypes, "(manual records only)");
    el("fColor").innerHTML = '<option value="">Any colour</option>'
      + render(data.vehicle_colors || [], supportedColors, "(manual records only)");
    el("fCamera").innerHTML = '<option value="">All cameras</option>'
      + (data.cameras || []).map((c) =>
        `<option value="${escapeHtml(c.id)}">${escapeHtml(c.name || c.id)}${
          c.city ? ` — ${escapeHtml(c.city)}` : ""}</option>`).join("");

    setFieldNote("fTypeNote", data.type_filter?.warning);
    setFieldNote("fColorNote", data.color_filter?.warning);
  } catch (_error) {
    setFieldNote("fTypeNote", "Filter options could not be loaded. Searching by time and camera still works.");
  }
}

function setFieldNote(id, text) {
  const node = el(id);
  if (!node) return;
  node.textContent = text || "";
  node.className = text ? "field-note is-warn" : "field-note";
}

function searchQuery() {
  const start = el("fStart").value;
  const end = el("fEnd").value;
  if (!start || !end) return null;
  if (new Date(start) > new Date(end)) return "invalid";
  const query = new URLSearchParams({
    start: new Date(start).toISOString(),
    end: new Date(end).toISOString(),
    limit: String(PAGE_SIZE),
    sort: "desc",
  });
  const add = (id, key) => { const v = el(id).value; if (v) query.set(key, v); };
  add("fFeed", "feed");
  add("fCamera", "camera_id");
  add("fColor", "vehicle_color");
  add("fType", "vehicle_type");
  add("fPlate", "plate_state");
  add("fReview", "review");
  return query;
}

async function runSearch(append = false) {
  const query = searchQuery();
  const results = el("searchResults");
  const meta = el("searchMeta");
  const warnings = el("searchWarnings");
  if (!query) {
    showEmpty(results, "Choose a time range", "Pick a start and end time, or use one of the quick ranges.");
    return;
  }
  if (query === "invalid") {
    showError(results, new ApiError("The start time must be before the end time.", 400));
    return;
  }
  if (!append) { searchOffset = 0; searchRows = []; showLoading(results, 6); warnings.innerHTML = ""; }
  query.set("offset", String(searchOffset));
  lastSearchQuery = query;
  try {
    const data = await api(`/api/investigations/vehicles?${query}`);
    const page = data.observations || [];
    searchRows = append ? searchRows.concat(page) : page;
    if (!append) {
      warnings.innerHTML = (data.search_warnings || [])
        .map((w) => `<div class="warning-strip"><span aria-hidden="true">⚠</span><span>${escapeHtml(w)}</span></div>`)
        .join("");
    }
    // The plate filter is applied to each page after it is fetched, so the
    // wording has to say what was actually counted rather than implying the
    // whole range was filtered.
    const total = data.total || 0;
    const scanned = Math.min(searchOffset + PAGE_SIZE, total);
    const cameras = Object.keys(data.camera_counts || {}).length;
    meta.textContent = !searchRows.length ? "" : el("fPlate").value
      ? `${searchRows.length} match${searchRows.length === 1 ? "" : "es"} in the ${scanned} most recent of ${total} records`
      : `Showing ${searchRows.length} of ${total} vehicle${total === 1 ? "" : "s"}`
        + ` across ${cameras} camera${cameras === 1 ? "" : "s"}, most recent first`;
    const more = scanned < total;
    if (!searchRows.length && !more) {
      // The own feed is a plate-reading source: it produces sightings and
      // alerts, not tracked vehicle observations. Telling someone to "widen the
      // time range" when no time range would ever help is a dead end.
      showEmpty(results, "No vehicles match this search",
        el("fFeed").value === "own"
          ? "The own test feed reads number plates rather than tracking vehicles. Its results appear under Alerts and in the watchlist."
          : "Try a wider time range, or clear the colour and plate filters.");
      return;
    }
    results.innerHTML = (searchRows.length
        ? observationTable(searchRows)
        : `<div class="state"><strong>Nothing matched in the ${scanned} records checked so far</strong>
             <p>Keep looking further back, or clear the plate filter.</p></div>`)
      + (more ? `<div class="state"><button class="btn-secondary" id="btnMore">Check ${
          Math.min(PAGE_SIZE, total - scanned)} older records</button></div>` : "");
    wireObservationTable(results);
    el("btnMore")?.addEventListener("click", (event) => withBusy(event.currentTarget, () => {
      searchOffset += PAGE_SIZE;
      return runSearch(true);
    }));
    await hydrateImages(results);
  } catch (error) {
    showError(results, error, "Try again");
  }
}

function observationTable(rows) {
  return `<div class="table-wrap"><table>
    <caption class="visually-hidden">Vehicle observations</caption>
    <thead><tr>
      <th scope="col">Vehicle</th>
      <th scope="col">Camera</th>
      <th scope="col">Seen</th>
      <th scope="col">Type</th>
      <th scope="col">Colour</th>
      <th scope="col">Number plate</th>
      <th scope="col">Review</th>
      <th scope="col"><span class="visually-hidden">Actions</span></th>
    </tr></thead>
    <tbody>${rows.map(observationRow).join("")}</tbody>
  </table></div>`;
}

function observationRow(row) {
  const first = formatTime(row, "first_seen_at", "first_seen_at_ist");
  const last = formatTime(row, "last_seen_at", "last_seen_at_ist");
  return `<tr class="is-clickable" data-observation="${row.id}" tabindex="0">
    <td>${evidenceImg(row.evidence_path, `Vehicle seen at ${cameraLabel(row)}`)}</td>
    <td>${escapeHtml(cameraLabel(row))}<span class="sub">${feedTag(row)} ${escapeHtml(placeLabel(row))}</span></td>
    <td class="num">${escapeHtml(first)}<span class="sub">${
      last && last !== first ? `to ${escapeHtml(last)}` : "single frame"}</span></td>
    <td>${escapeHtml(row.type_display || "Unknown")} ${trustBadge(row.type_state)}</td>
    <td>${escapeHtml(row.color_display || "Unknown")} ${trustBadge(row.color_state)}</td>
    <td>${plateCell(row)}</td>
    <td>${reviewBadge(row)}</td>
    <td><button class="btn-quiet" data-open="${row.id}">Open</button></td>
  </tr>`;
}

function wireObservationTable(root) {
  root.querySelectorAll("[data-observation]").forEach((tr) => {
    const open = () => openObservation(tr.dataset.observation);
    tr.addEventListener("click", (event) => {
      if (event.target.closest("button") && !event.target.closest("[data-open]")) return;
      open();
    });
    tr.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); open(); }
    });
  });
}

// ====================================================== observation detail =
let openObservationId = null;

async function openObservation(id) {
  openObservationId = id;
  const dialog = el("detailDialog");
  const body = el("detailBody");
  showLoading(body, 4);
  if (!dialog.open) dialog.showModal();
  try {
    const row = await api(`/api/vehicle-observations/${encodeURIComponent(id)}`);
    el("detailTitle").textContent = `Vehicle at ${cameraLabel(row)}`;
    body.innerHTML = observationDetail(row);
    await hydrateImages(body);
    wireReviewForm(body, row);
  } catch (error) {
    showError(body, error);
  }
}

function observationDetail(row) {
  const first = formatTime(row, "first_seen_at", "first_seen_at_ist");
  const last = formatTime(row, "last_seen_at", "last_seen_at_ist");
  // Only ever attached to the attribute the person actually confirmed.
  const verifiedLine = row.verified_by
    ? `<span class="sub">confirmed by ${escapeHtml(row.verified_by)}${
        row.verified_at ? ` · ${escapeHtml(formatTime(row, "verified_at", ""))}` : ""}</span>`
    : "";

  return `
    <div class="evidence-pair">
      <figure>
        ${evidenceImg(row.evidence_path, "Best view of this vehicle", "")}
        <figcaption>Best available view of the vehicle, chosen automatically from the frames it appeared in.</figcaption>
      </figure>
      <figure>
        ${evidenceImg(row.context_evidence_path, "Surrounding camera frame", "")}
        <figcaption>The surrounding frame, for context. Full video remains with the department that owns the camera.</figcaption>
      </figure>
    </div>

    <dl class="datalist">
      <dt>Camera</dt><dd>${escapeHtml(cameraLabel(row))}<span class="sub">${escapeHtml(placeLabel(row))}</span></dd>
      <dt>Feed</dt><dd>${feedTag(row)}<span class="sub">${escapeHtml(FEED[feedOf(row)].detail)}</span></dd>
      <dt>First seen</dt><dd>${escapeHtml(first)}</dd>
      <dt>Last seen</dt><dd>${escapeHtml(last)}</dd>
      <dt>Vehicle type</dt><dd>${escapeHtml(row.type_display || "Unknown")} ${trustBadge(row.type_state)} ${
        row.type_state === "verified" ? verifiedLine : ""}</dd>
      <dt>Colour</dt><dd>${escapeHtml(row.color_display || "Unknown")} ${trustBadge(row.color_state)} ${
        row.color_state === "verified" ? verifiedLine : ""}</dd>
      <dt>Number plate</dt><dd>${plateCell(row)}</dd>
      <dt>Review</dt><dd>${reviewBadge(row)}</dd>
      <dt>Reference</dt><dd><span class="plate-text">OBS-${escapeHtml(String(row.id).padStart(6, "0"))}</span></dd>
    </dl>

    ${row.type_note || row.color_note
      ? `<div class="warning-strip"><span aria-hidden="true">⚠</span><span>${
          escapeHtml(row.type_note || row.color_note)}</span></div>`
      : ""}

    <div class="warning-strip">
      <span aria-hidden="true">⚠</span>
      <span>Colour is an estimate produced automatically and needs a person to confirm it.
      Automatic vehicle type is not in operational use and is shown as Unknown until verified.
      This record describes one passage past one camera; it does not identify a vehicle.</span>
    </div>

    <form class="review-form" id="reviewForm">
      <h3>Record your verdict</h3>
      <p class="field-note">What you save here replaces the automatic value everywhere it is shown and is recorded against your token. The original automatic reading is kept.</p>
      <div class="review-grid">
        <div class="field">
          <label for="rType">Vehicle type</label>
          <select id="rType">
            <option value="">Leave unchanged</option>
            ${reviewOptions(options.vehicle_types, options.supported_vehicle_types, row.verified_vehicle_type)}
          </select>
        </div>
        <div class="field">
          <label for="rColor">Colour</label>
          <select id="rColor">
            <option value="">Leave unchanged</option>
            ${reviewOptions(options.vehicle_colors, options.supported_vehicle_colors, row.verified_vehicle_color)}
          </select>
        </div>
        <div class="field field-actions">
          <button type="submit" id="btnSaveReview">Save verdict</button>
        </div>
      </div>
    </form>`;
}

function reviewOptions(values, supported, selected) {
  const canPredict = new Set(supported || []);
  const list = (values || []).filter((v) => v !== "unknown");
  const group = (label, items) => items.length
    ? `<optgroup label="${label}">${items.map((v) =>
        `<option value="${escapeHtml(v)}"${v === selected ? " selected" : ""}>${escapeHtml(titleCase(v))}</option>`
      ).join("")}</optgroup>`
    : "";
  return group("Recognised automatically", list.filter((v) => canPredict.has(v)))
    + group("Recorded by a person only", list.filter((v) => !canPredict.has(v)));
}

function wireReviewForm(root, row) {
  const form = root.querySelector("#reviewForm");
  if (!form) return;
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const type = root.querySelector("#rType").value;
    const color = root.querySelector("#rColor").value;
    if (!type && !color) {
      toast("Choose a vehicle type or a colour before saving.", "warn");
      return;
    }
    const parts = [type ? `type to ${titleCase(type)}` : "", color ? `colour to ${titleCase(color)}` : ""]
      .filter(Boolean).join(" and ");
    const ok = await confirmAction(
      `Set the ${parts} for this vehicle? This becomes the confirmed value and is recorded against you.`,
      "Save verdict"
    );
    if (!ok) return;
    await withBusy(root.querySelector("#btnSaveReview"), async () => {
      try {
        const body = {};
        if (type) body.vehicle_type = type;
        if (color) body.vehicle_color = color;
        await api(`/api/vehicle-observations/${encodeURIComponent(row.id)}/review`, {
          method: "POST",
          body: JSON.stringify(body),
        });
        toast("Verdict saved.", "ok");
        await openObservation(row.id);
        if (currentPage === "search" && lastSearchQuery) runSearch();
        if (currentPage === "overview") loadOverview();
      } catch (error) {
        toast(errorText(error), "bad");
      }
    });
  });
}

// =============================================================== movement ==
async function runMovement() {
  const plate = el("mPlate").value.trim();
  const list = el("movementList");
  if (!plate) return;
  ensureMovementMap();
  showLoading(list, 4);
  const query = new URLSearchParams({ routes: "true" });
  const day = el("mDay").value;
  if (day) query.set("day", day);
  try {
    const data = await api(`/api/vehicles/${encodeURIComponent(plate)}?${query}`);
    const sightings = data.sightings || [];
    if (!sightings.length) {
      showEmpty(list, `No sightings for ${plate}`,
        day ? "Try another date, or leave the date blank for the last 24 hours."
            : "This plate has not been recorded by any camera in the last 24 hours.");
      drawMovement([]);
      return;
    }
    // A camera reading the same plate frame after frame is one passage, not
    // forty sightings. Consecutive readings at one camera are shown as a single
    // stop with its time span, which is what an investigator actually follows.
    const stops = groupPassages(sightings);
    list.innerHTML = `<p class="panel-meta" style="margin-bottom:12px">${
      stops.length} stop${stops.length === 1 ? "" : "s"} from ${sightings.length} reading${
      sightings.length === 1 ? "" : "s"}</p>
      <ol class="cards">${stops.map((stop, index) => `
      <li class="card">
        <div class="card-head">
          <span class="card-title">${index + 1}. ${escapeHtml(cameraLabel(stop.first))}</span>
          ${stop.count > 1 ? `<span class="badge badge-unknown badge-plain">${stop.count} readings</span>` : ""}
        </div>
        <p class="card-meta">${escapeHtml(formatTime(stop.first, "source_time", "source_time_ist"))}${
          stop.count > 1 ? ` to ${escapeHtml(formatTime(stop.last, "source_time", "source_time_ist"))}` : ""}</p>
        <p class="card-meta">${escapeHtml(placeLabel(stop.first))}</p>
      </li>`).join("")}</ol>`;
    drawMovement(stops.map((stop) => ({ ...stop.first, _count: stop.count })), data.possible_routes);
  } catch (error) {
    showError(list, error, "Try again");
  }
}

function groupPassages(sightings) {
  const stops = [];
  sightings.forEach((s) => {
    const previous = stops[stops.length - 1];
    if (previous && previous.first.camera_id === s.camera_id) {
      previous.last = s;
      previous.count += 1;
      return;
    }
    stops.push({ first: s, last: s, count: 1 });
  });
  return stops;
}

function drawMovement(sightings, routes = []) {
  if (!maps.movement) return;
  layers.movementMarkers.clearLayers();
  layers.movementLines.clearLayers();
  const points = [];
  // Road-matched paths between consecutive stops. A road the vehicle COULD
  // have taken, never one it was observed taking, so it is drawn under the
  // markers and labelled as such wherever it is touched.
  (routes || []).forEach((route) => {
    if (!route.path || route.path.length < 2) return;
    const guessed = route.provider === "fallback_straight";
    L.polyline(route.path, {
      color: guessed ? "#6B788C" : "#2A5B99",
      weight: 4, opacity: 0.7, dashArray: guessed ? "5 8" : null,
    })
      .bindPopup(`${escapeHtml(cameraNames.get(route.from_camera) || route.from_camera || "")} → ${
        escapeHtml(cameraNames.get(route.to_camera) || route.to_camera || "")}<br>${
        guessed ? "Straight line — no road match available." : "A road route that was possible in the time available. Not the route taken."}`)
      .addTo(layers.movementLines);
  });
  sightings.forEach((s, index) => {
    if (s.lat == null || s.lng == null) return;
    points.push([s.lat, s.lng]);
    L.circleMarker([s.lat, s.lng], { radius: 9, color: "#1C4173", fillColor: "#2A5B99", fillOpacity: 0.9, weight: 2 })
      .bindPopup(`<strong>${index + 1}. ${escapeHtml(cameraLabel(s))}</strong><br>${
        escapeHtml(formatTime(s, "source_time", "source_time_ist"))}${
        s._count > 1 ? `<br>${s._count} readings at this camera` : ""}`)
      .addTo(layers.movementMarkers);
  });
  if (points.length > 1) {
    L.polyline(points, { color: "#C4A24B", weight: 3, dashArray: "8 7" })
      .bindPopup("Inferred from the order of sightings. Not a recorded route.")
      .addTo(layers.movementLines);
  }
  if (points.length) maps.movement.fitBounds(points, { padding: [40, 40], maxZoom: 14 });
}

// ================================================================= alerts ==
function alertCard(a) {
  const priority = (a.priority || "unset").toLowerCase();
  const label = { high: "High", medium: "Medium", low: "Low" }[priority] || "Priority not set";
  return `<article class="card is-alert">
    <div class="card-head">
      <span class="card-title plate-text">${escapeHtml(a.plate_norm)}</span>
      <span class="prio prio-${escapeHtml(priority)}">${escapeHtml(label)}</span>
    </div>
    ${evidenceImg(a.evidence_path, `Vehicle matching ${a.plate_norm}`, "thumb")}
    <p class="card-meta">
      ${feedTag(a)} ${escapeHtml(cameraLabel(a))} · ${escapeHtml(formatTime(a, "source_time", "source_time_ist"))}
    </p>
    <p class="card-meta">${escapeHtml(titleCase(a.purpose) || "Watchlist match")} · ${escapeHtml(titleCase(a.status))}</p>
    <p class="card-meta">Raised by an exact match to a watchlist plate. Vehicle type and colour on this record are automatic estimates and do not confirm identity.</p>
    ${feedOf(a) === "own"
      ? `<p class="card-meta is-warn">This camera is the own test feed — frames generated on this host with
         known plates. The match is real, but the vehicle is not.</p>`
      : ""}
    <div class="card-actions">
      <button class="btn-secondary" data-alert="${a.id}" data-status="acknowledged">Acknowledge</button>
      <button data-alert="${a.id}" data-status="confirmed">Confirm match</button>
      <button class="btn-secondary" data-alert="${a.id}" data-status="rejected">Not a match</button>
    </div>
  </article>`;
}

function wireAlertActions(root) {
  root.querySelectorAll("[data-alert]").forEach((button) => {
    button.addEventListener("click", async () => {
      const status = button.dataset.status;
      const wording = {
        acknowledged: "Acknowledge this alert?",
        confirmed: "Confirm this is the watchlist vehicle? This is recorded against you.",
        rejected: "Record this as not a match? This is recorded against you.",
      }[status];
      if (status !== "acknowledged" && !(await confirmAction(wording, "Confirm"))) return;
      await withBusy(button, async () => {
        try {
          await api(`/api/alerts/${encodeURIComponent(button.dataset.alert)}`, {
            method: "PATCH",
            body: JSON.stringify({ status }),
          });
          toast("Alert updated.", "ok");
          loadAlerts();
          loadOverview();
        } catch (error) {
          toast(errorText(error), "bad");
        }
      });
    });
  });
}

async function loadAlerts() {
  const node = el("alertList");
  showLoading(node, 3);
  try {
    const alerts = await api("/api/alerts");
    if (!alerts.length) {
      showEmpty(node, "No alerts",
        "An alert is raised only when a camera reads a plate that exactly matches an active watchlist entry.");
      return;
    }
    node.innerHTML = `<div class="cards">${alerts.map(alertCard).join("")}</div>`;
    await hydrateImages(node);
    wireAlertActions(node);
  } catch (error) {
    showError(node, error, "Try again");
  }
}

// ============================================================== watchlist ==
async function loadWatchlist() {
  const node = el("watchlistRows");
  showLoading(node, 2);
  try {
    const rows = await api("/api/watchlist");
    const active = rows.filter((r) => r.active);
    if (!rows.length) {
      showEmpty(node, "The watchlist is empty", "Add a plate above to be alerted when it is seen.");
    } else {
      node.innerHTML = `<div class="cards">${rows.map((w) => `
        <article class="card">
          <div class="card-head">
            <span class="card-title plate-text">${escapeHtml(w.plate_norm)}</span>
            <span class="prio prio-${escapeHtml((w.priority || "unset").toLowerCase())}">${
              escapeHtml(titleCase(w.priority) || "Priority not set")}</span>
          </div>
          <p class="card-meta">${escapeHtml(titleCase(w.purpose))} · ${w.active ? "Active" : "Inactive"}</p>
          <div class="card-actions">
            <button class="btn-quiet" data-trace="${escapeHtml(w.plate_norm)}">Trace movement</button>
            <button class="${w.active ? "btn-secondary" : ""}" data-watch="${w.id}" data-active="${w.active ? "0" : "1"}">
              ${w.active ? "Deactivate" : "Reactivate"}
            </button>
          </div>
        </article>`).join("")}</div>`;
      wireWatchlistActions(node);
    }
    el("navAlertCount").dataset.watchlist = String(active.length);
  } catch (error) {
    showError(node, error, "Try again");
  }
  loadObservedPlates();
}

function wireWatchlistActions(root) {
  root.querySelectorAll("[data-trace]").forEach((button) => {
    button.addEventListener("click", () => {
      el("mPlate").value = button.dataset.trace;
      showPage("movement");
      runMovement();
    });
  });
  root.querySelectorAll("[data-watch]").forEach((button) => {
    button.addEventListener("click", async () => {
      const activate = button.dataset.active === "1";
      const ok = await confirmAction(
        activate
          ? "Reactivate this plate? Sightings already recorded will be re-checked and may raise alerts."
          : "Deactivate this plate? No new alerts will be raised for it.",
        activate ? "Reactivate" : "Deactivate"
      );
      if (!ok) return;
      await withBusy(button, async () => {
        try {
          await api(`/api/watchlist/${encodeURIComponent(button.dataset.watch)}`, {
            method: "PATCH",
            body: JSON.stringify({ active: activate, rematch: activate }),
          });
          toast(activate ? "Plate reactivated." : "Plate deactivated.", "ok");
          loadWatchlist();
          loadAlerts();
          loadOverview();
        } catch (error) {
          toast(errorText(error), "bad");
        }
      });
    });
  });
}

async function loadObservedPlates() {
  const node = el("observedPlates");
  showLoading(node, 2);
  try {
    const rows = await api("/api/observed-plates");
    const usable = rows.filter((p) => p.plate_norm);
    if (!usable.length) {
      showEmpty(node, "No plates have been read yet",
        "Plates are read only when the vehicle is close enough and the image is clear enough.");
      return;
    }
    node.innerHTML = `<div class="table-wrap"><table>
      <thead><tr>
        <th scope="col">Plate</th><th scope="col">Times seen</th>
        <th scope="col">Last camera</th><th scope="col">Status</th>
        <th scope="col"><span class="visually-hidden">Actions</span></th>
      </tr></thead>
      <tbody>${usable.slice(0, 40).map((p) => `<tr>
        <td><span class="plate-text">${escapeHtml(p.plate_norm)}</span></td>
        <td class="num">${escapeHtml(p.count)}</td>
        <td>${escapeHtml(cameraNames.get(p.last_camera) || p.last_camera || "—")}<span class="sub">${
          escapeHtml(formatTime(p, "last_time", ""))}</span></td>
        <td>${p.syntax_ok
          ? `<span class="badge badge-estimated">Valid format</span>`
          : `<span class="badge badge-unknown">Unusual format</span>`}</td>
        <td>${p.watchlisted
          ? `<span class="card-meta">On the watchlist</span>`
          : `<button class="btn-quiet" data-addwatch="${escapeHtml(p.plate_norm)}">Add to watchlist</button>`}</td>
      </tr>`).join("")}</tbody></table></div>`;
    node.querySelectorAll("[data-addwatch]").forEach((button) => {
      button.addEventListener("click", async () => {
        const plate = button.dataset.addwatch;
        if (!(await confirmAction(
          `Add ${plate} to the watchlist? Sightings already recorded will be re-checked and may raise alerts.`,
          "Add to watchlist"
        ))) return;
        await withBusy(button, () => addToWatchlist(plate, "operator_added_from_sighting", "high"));
      });
    });
  } catch (error) {
    showError(node, error, "Try again");
  }
}

async function addToWatchlist(plate, purpose, priority) {
  try {
    const result = await api("/api/watchlist", {
      method: "POST",
      body: JSON.stringify({ plate_raw: plate, purpose, priority, rematch: true }),
    });
    const created = result.rematch?.alerts_created || 0;
    toast(created
      ? `${result.plate_norm} added. ${created} alert${created === 1 ? "" : "s"} raised from earlier sightings.`
      : `${result.plate_norm} added to the watchlist.`, "ok");
    loadWatchlist();
    loadAlerts();
    loadOverview();
  } catch (error) {
    toast(errorText(error), "bad");
  }
}

// ================================================================ cameras ==
let cameras = [];
let selectedCamera = null;

const AVAILABILITY_BADGE = {
  monitoring: "verified",
  available: "verified",
  unreachable: "alert",
  not_checked: "unknown",
};

// Shared by the Cameras page and the Feed sources panel, so the same action
// cannot behave two different ways depending on where it was clicked.
function checkConnections(button, retestFailed = false) {
  return withBusy(button, async () => {
    try {
      const result = await api("/api/capacity/measure", {
        method: "POST",
        body: JSON.stringify({ retest_failed: retestFailed }),
      });
      const tested = result.tested_count || 0;
      const untested = result.catalogue_remaining_untested || 0;
      // "Checked 0 cameras: 0 available. 0 still to check." is technically true
      // and completely unhelpful. It means every camera already has a result,
      // so say that and point at the action that would change something.
      toast(tested
        ? `Checked ${tested} camera${tested === 1 ? "" : "s"}: ${result.decode_ok_count || 0} available.`
          + (untested ? ` ${untested} still to check.` : " All cameras now have a result.")
        : "Every camera already has a connection result. Use Re-check unreachable to try the failed ones again.",
        tested ? "ok" : "warn");
      await loadCameras();
      loadOverview();
    } catch (error) {
      toast(errorText(error), "bad");
    }
  });
}

function startAvailable(button) {
  return withBusy(button, async () => {
    try {
      const result = await api("/api/workers/start-accessible", {
        method: "POST",
        body: JSON.stringify({ decode_ok_only: true }),
      });
      const started = (result.started || []).length;
      toast(started
        ? `Monitoring started on ${started} camera${started === 1 ? "" : "s"}.`
        : "No cameras are available to monitor. Run Check connections first.", started ? "ok" : "warn");
      await loadCameras();
      loadOverview();
    } catch (error) {
      toast(errorText(error), "bad");
    }
  });
}

async function loadCameras() {
  const node = el("cameraList");
  if (!cameras.length) showLoading(node, 5);
  ensureCameraMap();
  try {
    cameras = await api("/api/cameras");
    renderCameras();
  } catch (error) {
    showError(node, error, "Try again");
  }
}

function renderCameras() {
  const node = el("cameraList");
  const filter = el("cameraFilter").value;
  const feedFilter = el("cameraFeedFilter").value;
  const term = el("cameraSearch").value.trim().toLowerCase();
  const rows = cameras.filter((c) => {
    if (filter !== "all" && c.availability !== filter) return false;
    if (feedFilter !== "all" && feedOf(c) !== feedFilter) return false;
    if (!term) return true;
    return [c.id, c.name, c.city, c.department].filter(Boolean)
      .some((value) => String(value).toLowerCase().includes(term));
  });

  if (!rows.length) {
    showEmpty(node, "No cameras match", "Change the filter or clear the search box.");
  } else {
    node.innerHTML = `<div class="table-wrap is-scroll"><table>
      <thead><tr>
        <th scope="col">Camera</th><th scope="col">Feed</th><th scope="col">Location</th>
        <th scope="col">Status</th><th scope="col">Last checked</th>
        <th scope="col"><span class="visually-hidden">Actions</span></th>
      </tr></thead>
      <tbody>${rows.map((c) => `
        <tr class="is-clickable${selectedCamera === c.id ? " is-selected" : ""}" data-camera="${escapeHtml(c.id)}" tabindex="0">
          <td>${escapeHtml(c.name || c.id)}<span class="sub">${escapeHtml(c.id)}</span></td>
          <td>${feedTag(c)}</td>
          <td>${escapeHtml(placeLabel(c))}</td>
          <td><span class="badge badge-${AVAILABILITY_BADGE[c.availability] || "unknown"}">${
            escapeHtml(c.availability_label)}</span></td>
          <td class="num">${c.decode_tested_at_ist
            ? escapeHtml(formatTime(c, "decode_tested_at", "decode_tested_at_ist"))
            : '<span class="sub">Never</span>'}</td>
          <td><button class="btn-quiet" data-camera-open="${escapeHtml(c.id)}">Details</button></td>
        </tr>`).join("")}</tbody>
    </table></div>`;
    node.querySelectorAll("[data-camera]").forEach((tr) => {
      const open = () => selectCamera(tr.dataset.camera);
      tr.addEventListener("click", open);
      tr.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); open(); }
      });
    });
  }
  drawCameraMap();
  if (selectedCamera) renderCameraDetail();
  else showEmpty(el("cameraDetail"), "Select a camera", "Choose a camera from the list to see its status and controls.");
}

function drawCameraMap() {
  if (!maps.camera) return;
  layers.cameraMarkers.clearLayers();
  const points = [];
  cameras.forEach((c) => {
    if (c.lat == null || c.lng == null || !Number.isFinite(Number(c.lat))) return;
    const colour = { monitoring: "#186B43", available: "#2A5B99", unreachable: "#A4241D" }[c.availability] || "#6B788C";
    points.push([c.lat, c.lng]);
    L.circleMarker([c.lat, c.lng], { radius: 7, color: colour, fillColor: colour, fillOpacity: 0.85, weight: 2 })
      .bindPopup(`<strong>${escapeHtml(c.name || c.id)}</strong><br>${escapeHtml(placeLabel(c))}<br>${
        escapeHtml(c.availability_label)}`)
      .on("click", () => selectCamera(c.id))
      .addTo(layers.cameraMarkers);
  });
  if (points.length && !maps.camera._gpFitted) {
    maps.camera.fitBounds(points, { padding: [40, 40], maxZoom: 11 });
    maps.camera._gpFitted = true;
  }
}

function selectCamera(id) {
  selectedCamera = id;
  document.querySelectorAll("[data-camera]").forEach((tr) => {
    tr.classList.toggle("is-selected", tr.dataset.camera === id);
  });
  renderCameraDetail();
  el("cameraDetailPanel").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderCameraDetail() {
  const camera = cameras.find((c) => c.id === selectedCamera);
  const node = el("cameraDetail");
  if (!camera) {
    showEmpty(node, "Camera not found", "It may have been removed. Refresh the list.");
    return;
  }
  const monitoring = camera.availability === "monitoring";
  const recorded = camera.source_type === "image_dir" || camera.source_type === "file";
  node.innerHTML = `
    <dl class="datalist">
      <dt>Camera</dt><dd>${escapeHtml(camera.name || camera.id)}<span class="sub">${escapeHtml(camera.id)}</span></dd>
      <dt>Feed</dt><dd>${feedTag(camera)}<span class="sub">${escapeHtml(FEED[feedOf(camera)].detail)}</span></dd>
      <dt>Location</dt><dd>${escapeHtml(placeLabel(camera))}${
        camera.coords_are_inferred ? '<span class="sub">Position estimated from the camera name, not surveyed</span>' : ""}</dd>
      <dt>Status</dt><dd><span class="badge badge-${AVAILABILITY_BADGE[camera.availability] || "unknown"}">${
        escapeHtml(camera.availability_label)}</span><span class="sub">${escapeHtml(camera.availability_detail)}</span></dd>
      <dt>Last frame</dt><dd>${camera.last_frame_at_ist
        ? escapeHtml(formatTime(camera, "last_frame_at", "last_frame_at_ist"))
        : '<span class="sub">No frame received yet</span>'}</dd>
    </dl>

    <div class="card-actions" style="margin-top:12px">
      <button ${monitoring ? 'class="btn-secondary"' : ""} data-monitor="${monitoring ? "stop" : "start"}">
        ${monitoring ? "Stop monitoring" : "Start monitoring"}
      </button>
      ${recorded ? `<button class="btn-secondary" data-analyse>Analyse recorded video</button>` : ""}
      <button class="btn-secondary" data-live-frame>View live frame</button>
    </div>
    <div id="cameraFrame" style="margin-top:12px"></div>`;

  node.querySelector("[data-monitor]").addEventListener("click", async (event) => {
    const action = event.currentTarget.dataset.monitor;
    await withBusy(event.currentTarget, async () => {
      try {
        await api(`/api/workers/${encodeURIComponent(camera.id)}/${action}`, { method: "POST" });
        toast(action === "start"
          ? "Monitoring requested. The status updates once a frame arrives."
          : "Monitoring stopped.", "ok");
        await loadCameras();
      } catch (error) {
        toast(errorText(error), "bad");
      }
    });
  });

  node.querySelector("[data-analyse]")?.addEventListener("click", async (event) => {
    await withBusy(event.currentTarget, async () => {
      try {
        const result = await api(`/api/cameras/${encodeURIComponent(camera.id)}/analyze`, { method: "POST" });
        toast(`Analysis finished. ${result.sightings || 0} plate reading${
          result.sightings === 1 ? "" : "s"}, ${result.alerts || 0} alert${result.alerts === 1 ? "" : "s"}.`, "ok");
        await loadCameras();
        loadOverview();
      } catch (error) {
        toast(errorText(error), "bad");
      }
    });
  });

  node.querySelector("[data-live-frame]").addEventListener("click", async (event) => {
    const frame = el("cameraFrame");
    await withBusy(event.currentTarget, async () => {
      showLoading(frame, 1);
      try {
        const response = await fetch(`/api/cameras/${encodeURIComponent(camera.id)}/snapshot`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!response.ok) throw new ApiError("No live frame is available from this camera right now.", response.status);
        const url = URL.createObjectURL(await response.blob());
        objectUrls.add(url);
        frame.innerHTML = `<figure class="evidence-pair"><figure>
          <img src="${url}" alt="Live frame from ${escapeHtml(camera.name || camera.id)}" />
          <figcaption>Single frame taken just now for the operator. Not a recording.</figcaption>
        </figure></figure>`;
      } catch (error) {
        showError(frame, error);
      }
    });
  });
}

// ================================================================== boot ===
function bindNav() {
  document.querySelectorAll(".navitem").forEach((button) => {
    button.addEventListener("click", () => showPage(button.dataset.page));
  });
  document.querySelectorAll("[data-goto]").forEach((button) => {
    button.addEventListener("click", () => showPage(button.dataset.goto));
  });
}

function bindSearch() {
  el("searchForm").addEventListener("submit", (event) => {
    event.preventDefault();
    withBusy(el("btnSearch"), runSearch);
  });
  document.querySelectorAll("[data-range]").forEach((button) => {
    button.addEventListener("click", () => {
      setRange(Number(button.dataset.range));
      withBusy(el("btnSearch"), runSearch);
    });
  });
  el("btnExport").addEventListener("click", () => withBusy(el("btnExport"), exportSearch));
}

// One download path for every export. The token travels in the header, so it
// never appears in a link the browser or a proxy might record; the server
// writes an audit entry for each export it produces.
async function download(path, filename) {
  try {
    const response = await fetch(path, { headers: { Authorization: `Bearer ${token}` } });
    if (!response.ok) throw new ApiError("The export could not be produced.", response.status);
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    toast("Export downloaded. Access to evidence is recorded.", "ok");
  } catch (error) {
    toast(errorText(error), "bad");
  }
}

async function exportSearch() {
  const query = searchQuery();
  if (!query || query === "invalid") {
    toast("Choose a valid time range before exporting.", "warn");
    return;
  }
  // The export covers the whole filtered range, not the page on screen, so the
  // paging and plate-state parameters are dropped rather than carried over.
  const params = new URLSearchParams(query);
  ["plate_state", "review", "limit", "offset", "sort"].forEach((key) => params.delete(key));
  await download(`/api/investigations/vehicles/export.csv?${params}`,
    `vehicle-observations-${new Date().toISOString().slice(0, 10)}.csv`);
}

function bindMovement() {
  el("movementForm").addEventListener("submit", (event) => {
    event.preventDefault();
    withBusy(el("btnMovement"), runMovement);
  });
  const exportMovement = (extension) => (event) => withBusy(event.currentTarget, async () => {
    const plate = el("mPlate").value.trim();
    if (!plate) { toast("Enter a number plate first.", "warn"); return; }
    const query = new URLSearchParams();
    const day = el("mDay").value;
    if (day) query.set("day", day);
    await download(`/api/vehicles/${encodeURIComponent(plate)}/export.${extension}?${query}`,
      `${plate}-sightings.${extension}`);
  });
  el("btnMovementCsv").addEventListener("click", exportMovement("csv"));
  el("btnMovementGeo").addEventListener("click", exportMovement("geojson"));
}

function bindWatchlist() {
  el("watchlistForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const plate = el("wPlate").value.trim();
    if (!plate) return;
    const ok = await confirmAction(
      `Add ${plate.toUpperCase()} to the watchlist? Sightings already recorded will be re-checked and may raise alerts.`,
      "Add to watchlist"
    );
    if (!ok) return;
    await withBusy(el("btnWatchAdd"), async () => {
      await addToWatchlist(plate, el("wPurpose").value, el("wPriority").value);
      el("wPlate").value = "";
    });
  });
}

function bindCameras() {
  el("cameraFilter").addEventListener("change", renderCameras);
  el("cameraSearch").addEventListener("input", renderCameras);

  el("btnCheckCameras").addEventListener("click", (event) => checkConnections(event.currentTarget));
  el("btnRecheckFailed").addEventListener("click", (event) => checkConnections(event.currentTarget, true));
  el("btnStartAvailable").addEventListener("click", (event) => startAvailable(event.currentTarget));
  el("cameraFeedFilter").addEventListener("change", renderCameras);

  el("btnStopAll").addEventListener("click", async (event) => {
    if (!(await confirmAction("Stop monitoring on every camera? No new vehicles will be recorded until it is restarted.", "Stop all"))) return;
    await withBusy(event.currentTarget, async () => {
      try {
        await api("/api/workers/stop-all", { method: "POST" });
        toast("Monitoring stopped on all cameras.", "ok");
        await loadCameras();
        loadOverview();
      } catch (error) {
        toast(errorText(error), "bad");
      }
    });
  });

  el("cameraForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const id = el("nId").value.trim();
    if (!id) return;
    const lat = el("nLat").value.trim();
    const lng = el("nLng").value.trim();
    if (Boolean(lat) !== Boolean(lng)) {
      toast("Enter both latitude and longitude, or neither.", "warn");
      return;
    }
    await withBusy(el("btnAddCamera"), async () => {
      try {
        const body = { id, source_type: "rtsp" };
        const optional = { name: "nName", department: "nDept", city: "nCity" };
        Object.entries(optional).forEach(([key, field]) => {
          const value = el(field).value.trim();
          if (value) body[key] = value;
        });
        if (lat && lng) { body.lat = Number(lat); body.lng = Number(lng); }
        await api("/api/cameras", { method: "POST", body: JSON.stringify(body) });
        toast(`${id} added. It stays Not checked until its connection is tested.`, "ok");
        el("cameraForm").reset();
        await loadCameras();
        loadOverview();
      } catch (error) {
        toast(errorText(error), "bad");
      }
    });
  });
}

function bindDialogs() {
  el("btnCloseDetail").addEventListener("click", () => el("detailDialog").close());
  el("detailDialog").addEventListener("close", () => { openObservationId = null; });
  document.addEventListener("click", (event) => {
    if (event.target.matches("[data-retry]")) {
      const page = event.target.closest(".page")?.dataset.page || currentPage;
      showPage(page);
    }
  });
}

// ---- polling ---------------------------------------------------------------
// The alert queue has to move without the operator clicking anything, but a
// dead backend must look dead rather than showing frozen figures as if live.
let pollFailures = 0;

async function poll() {
  if (document.hidden || !token) return;
  try {
    await api("/api/ui/overview");
    pollFailures = 0;
    el("connectionBanner").classList.add("hidden");
    loadOverview();
    if (currentPage === "alerts") loadAlerts();
  } catch (_error) {
    pollFailures += 1;
    if (pollFailures >= 2) {
      const banner = el("connectionBanner");
      banner.textContent = "The server cannot be reached. The figures below are from the last successful update.";
      banner.classList.remove("hidden");
    }
  }
}

// ---- sign in / out ---------------------------------------------------------
function signOut(message) {
  token = "";
  sessionStorage.removeItem(SESSION_KEY);
  releaseImages();
  el("app").classList.add("hidden");
  el("signIn").classList.remove("hidden");
  const error = el("signInError");
  if (message) {
    error.textContent = message;
    error.classList.remove("hidden");
  } else {
    error.classList.add("hidden");
  }
  el("operatorToken").value = "";
  el("operatorToken").focus();
}

async function startSession() {
  el("signIn").classList.add("hidden");
  el("app").classList.remove("hidden");
  try {
    const config = await api("/api/ui/config");
    el("envLabel").textContent = config.app_env || "production";
    // Shown only where the server actually serves /dev. In production the
    // link stays hidden AND the route 404s, so this is a convenience, not the
    // thing keeping the developer console out of reach.
    el("devConsoleLink").classList.toggle("hidden", !config.developer_ui);
  } catch (_error) {
    /* the banner covers a backend that is down */
  }
  await loadSearchOptions();
  setRange(24);
  showPage("overview");
  poll();
}

el("signInForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const entered = el("operatorToken").value.trim();
  const error = el("signInError");
  error.classList.add("hidden");
  if (!entered) return;
  await withBusy(el("signInButton"), async () => {
    // Validated against an endpoint that requires the operator role, so a wrong
    // token fails here rather than half way through an investigation.
    const response = await fetch("/api/settings/recognition", {
      headers: { Authorization: `Bearer ${entered}` },
    }).catch(() => null);
    if (!response || !response.ok) {
      error.textContent = response
        ? "That access token was not recognised."
        : "The server could not be reached. Check that the console is running.";
      error.classList.remove("hidden");
      return;
    }
    token = entered;
    sessionStorage.setItem(SESSION_KEY, token);
    await startSession();
  });
});

el("btnSignOut").addEventListener("click", () => signOut());

bindNav();
bindSearch();
bindMovement();
bindWatchlist();
bindCameras();
bindDialogs();
setInterval(poll, 15000);
document.addEventListener("visibilitychange", () => { if (!document.hidden) poll(); });

if (token) startSession();
else el("operatorToken").focus();
