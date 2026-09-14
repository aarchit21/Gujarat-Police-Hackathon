/* =========================================================================
   Bulk labelling — developer console only (/dev)
   =========================================================================
   Builds the human-labelled set that fine-tuning needs. 5,255 observations are
   unlabelled and 24 are labelled, so the loop has to be keyboard-driven: a
   three-keystroke rhythm (type, colour, commit) rather than open-a-details,
   pick-two-dropdowns, click-save.

   Three rules this file follows:

   * It loads AFTER console.js and shares its TOKEN, j(), evidenceUrl(),
     escapeText() and formatWhen(). It binds with addEventListener, never
     .onclick — console.js sets .onclick on every .tab, and assigning over it
     would break panel switching for the whole console.
   * Nothing here goes into pollOnce(). The 4-second refresh would wipe the
     stage and steal focus mid-run.
   * Every verdict records HOW it was produced. Labels taken with the model's
     answer on screen are confirmation-prone; the export has to be able to tell
     them apart from blind ones rather than treating all of them as ground
     truth. See `context` below.
   ========================================================================= */

"use strict";

(() => {
  const $ = (id) => document.getElementById(id);

  // Digits for type, letters for colour: no mode switch, and no collisions
  // with the control keys below.
  const TYPE_KEYS = {
    1: "car", 2: "suv", 3: "two_wheeler", 4: "truck",
    5: "bus", 6: "van", 7: "auto_rickshaw", 8: "taxi_cab", 0: "unknown",
  };
  const COLOR_KEYS = {
    w: "white", k: "black", s: "silver", e: "gray", r: "red", b: "blue",
    g: "green", y: "yellow", o: "orange", n: "brown", t: "other", u: "unknown",
  };
  const CONTROL_KEYS = new Set(["a", "x", "c", "z", "m", "?", " ", "Enter",
                                "ArrowRight", "ArrowLeft"]);

  const PAGE = 100;

  const L = {
    scope: null,
    queue: [],
    index: 0,
    total: 0,
    offset: 0,
    done: new Set(),      // committed this session
    skipped: new Set(),   // advanced past deliberately
    history: [],          // [{id, before:{type,color}}] -> powers undo
    pending: {},          // id -> {vehicle_type, vehicle_color}
    outbox: Promise.resolve(),
    failures: [],
    blind: false,         // predictions visible by default (chosen default)
    contextCropUsed: false,
  };

  // ---------------------------------------------------------------- scope --
  function readScope() {
    const start = $("lbStart").value;
    const end = $("lbEnd").value;
    if (!start || !end) return null;
    const scope = {
      start: new Date(start).toISOString(),
      end: new Date(end).toISOString(),
      review: $("lbReview").value,
      limit: String(PAGE),
    };
    const add = (id, key) => { const v = $(id).value; if (v) scope[key] = v; };
    add("lbCamera", "camera_id");
    add("lbType", "vehicle_type");
    add("lbColor", "vehicle_color");
    add("lbMinConf", "min_type_confidence");
    const source = $("lbSource").value;
    if (source) { scope.type_source = source; scope.color_source = source; }
    return scope;
  }

  function fillOptions() {
    const options = window.__vehicleOptions || {};
    const fill = (id, values, blank) => {
      const node = $(id);
      if (!node || node.dataset.filled) return;
      node.innerHTML = `<option value="">${blank}</option>` + (values || [])
        .map((v) => `<option value="${v}">${String(v).replaceAll("_", " ")}</option>`).join("");
      node.dataset.filled = "1";
    };
    fill("lbType", options.vehicle_types, "Any");
    fill("lbColor", options.vehicle_colors, "Any");
    const camera = $("lbCamera");
    if (camera && !camera.dataset.filled && (options.cameras || []).length) {
      camera.innerHTML = '<option value="">Any camera</option>' + options.cameras
        .map((c) => `<option value="${c.id}">${escapeText(c.id)} · ${escapeText(c.name || c.city || "")}</option>`).join("");
      camera.dataset.filled = "1";
    }
  }

  // ---------------------------------------------------------------- queue --
  async function loadQueue(append = false) {
    const scope = readScope();
    if (!scope) { stageMessage("Set a start and end time first."); return; }
    L.scope = scope;
    if (!append) { L.index = 0; L.queue = []; L.offset = 0; }
    stageMessage("Loading…");
    try {
      const query = new URLSearchParams({ ...scope, offset: String(L.offset) });
      const data = await j(`/api/dev/label-queue?${query}`);
      L.total = data.total || 0;
      // Rows already handled this session are filtered client-side. With
      // review=pending a labelled row LEAVES the result set, so paging by
      // offset would silently skip an equal number of unlabelled rows — see
      // advance() for why offset only moves when that cannot happen.
      const fresh = (data.observations || [])
        .filter((row) => !L.done.has(row.id) && !L.skipped.has(row.id));
      L.queue = append ? L.queue.concat(fresh) : fresh;
      if (!L.queue.length) {
        stageMessage(L.total
          ? "Every row in this scope has been handled. Widen the scope or change the filters."
          : "Nothing matches this scope.");
        renderProgress();
        return;
      }
      render();
    } catch (err) {
      stageMessage(`Could not load the queue: ${String(err).slice(0, 160)}`);
    }
  }

  function current() {
    return L.queue[L.index] || null;
  }

  async function advance(step = 1) {
    L.index += step;
    if (L.index < 0) L.index = 0;
    if (L.index >= L.queue.length) {
      // Only page forward when the current page cannot have shifted under us.
      if (L.scope && L.scope.review === "pending") {
        L.offset = 0;
      } else {
        L.offset += PAGE;
      }
      L.index = L.queue.length;
      await loadQueue(false);
      return;
    }
    render();
    preload();
  }

  function preload() {
    // Crop latency, not the annotator, should never set the pace.
    for (let i = L.index + 1; i <= L.index + 5 && i < L.queue.length; i += 1) {
      const row = L.queue[i];
      if (row && row.evidence_path) new Image().src = evidenceUrl(row.evidence_path);
    }
  }

  // --------------------------------------------------------------- commit --
  function verdictFor(row) {
    return L.pending[row.id] || {};
  }

  function setVerdict(row, field, value) {
    L.pending[row.id] = { ...verdictFor(row), [field]: value };
    render();
  }

  function acceptModel(row) {
    const raw = row.raw || {};
    const type = raw.type_candidate || (raw.vehicle_type !== "unknown" ? raw.vehicle_type : "");
    const color = raw.vehicle_color !== "unknown" ? raw.vehicle_color : (raw.color_candidate || "");
    if (!type && !color) { flash("The model abstained on this one — nothing to accept."); return; }
    L.pending[row.id] = {
      ...(type ? { vehicle_type: type } : {}),
      ...(color ? { vehicle_color: color } : {}),
      accepted_model: true,
    };
    render();
  }

  function commit() {
    const row = current();
    if (!row) return;
    const verdict = verdictFor(row);
    const body = {};
    if (verdict.vehicle_type) body.vehicle_type = verdict.vehicle_type;
    if (verdict.vehicle_color) body.vehicle_color = verdict.vehicle_color;
    if (!Object.keys(body).length) { flash("Pick a type or a colour first, or press x to skip."); return; }

    // What the label is worth, recorded with the label itself.
    body.context = {
      ui: "bulk-v1",
      prediction_visible: !L.blind,
      accepted_model: Boolean(verdict.accepted_model),
      context_crop_used: L.contextCropUsed,
      scope: L.scope,
    };
    const annotator = ($("lbAnnotator").value || "").trim().replace(/[^A-Za-z0-9_.-]/g, "").slice(0, 32);
    if (annotator) body.note = `annotator:${annotator}`;

    L.history.push({
      id: row.id,
      before: { type: row.verified_vehicle_type || "", color: row.verified_vehicle_color || "" },
    });
    L.done.add(row.id);
    delete L.pending[row.id];
    queueWrite(row.id, body);
    advance(1);
  }

  // Serial outbox: writes never reorder, and the UI never waits for one.
  function queueWrite(id, body) {
    L.outbox = L.outbox.then(async () => {
      try {
        await j(`/api/vehicle-observations/${id}/review`, { method: "POST", body: JSON.stringify(body) });
        L.failures = L.failures.filter((f) => f.id !== id);
      } catch (err) {
        L.failures.push({ id, body, error: String(err).slice(0, 160) });
      }
      renderFailures();
      renderProgress();
    });
  }

  function undo() {
    const last = L.history.pop();
    if (!last) { flash("Nothing to undo."); return; }
    L.done.delete(last.id);
    // Restore exactly what was there before, including "nothing" — the API
    // treats "" as clear-the-verdict, which is a real undo rather than a
    // cosmetic one.
    queueWrite(last.id, {
      vehicle_type: last.before.type,
      vehicle_color: last.before.color,
      context: { ui: "bulk-v1", undo: true },
    });
    const at = L.queue.findIndex((row) => row.id === last.id);
    if (at >= 0) L.index = at;
    render();
    flash("Undone.");
  }

  function skip() {
    const row = current();
    if (!row) return;
    L.skipped.add(row.id);
    advance(1);
  }

  async function clearVerdict() {
    const row = current();
    if (!row) return;
    queueWrite(row.id, { vehicle_type: "", vehicle_color: "", context: { ui: "bulk-v1", cleared: true } });
    delete L.pending[row.id];
    L.done.delete(row.id);
    flash("Verdict cleared.");
  }

  // ---------------------------------------------------------------- render --
  function stageMessage(text) {
    $("lbStage").className = "card muted";
    $("lbStage").textContent = text;
  }

  function flash(text) {
    const node = $("lbProgress");
    if (!node) return;
    const previous = node.textContent;
    node.textContent = text;
    setTimeout(() => { if (node.textContent === text) node.textContent = previous; }, 2200);
  }

  function renderProgress() {
    const node = $("lbProgress");
    if (!node) return;
    const failed = L.failures.length ? ` · ${L.failures.length} failed to save` : "";
    node.textContent = L.total
      ? `${L.done.size} labelled · ${L.skipped.size} skipped · ${L.total} in scope${failed}`
      : "";
  }

  function renderFailures() {
    const node = $("lbFailures");
    if (!node) return;
    if (!L.failures.length) { node.innerHTML = ""; return; }
    node.innerHTML = `<div class="card err">
      <b>${L.failures.length} verdict(s) did not save.</b>
      <div class="muted">${escapeText(L.failures[0].error)}</div>
      <button id="btnLabelRetry" style="margin-top:6px">Retry failed</button>
    </div>`;
    $("btnLabelRetry").addEventListener("click", () => {
      const retry = L.failures.slice();
      L.failures = [];
      retry.forEach((f) => queueWrite(f.id, f.body));
      renderFailures();
    });
  }

  function keycap(key, label, active) {
    return `<button class="lb-chip${active ? " on" : ""}" data-key="${key}">
      <kbd>${key === " " ? "␣" : key}</kbd>${escapeText(label.replaceAll("_", " "))}</button>`;
  }

  function modelPanel(row) {
    if (L.blind) {
      return `<div class="lb-model lb-model-blind">Model reading hidden — labelling blind.</div>`;
    }
    const raw = row.raw || {};
    const bar = (probs) => Object.entries(probs || {})
      .sort((a, b) => b[1] - a[1]).slice(0, 4)
      .map(([k, v]) => `<span class="lb-prob"><i style="width:${Math.round(v * 60)}px"></i>${escapeText(k)} ${v.toFixed(2)}</span>`)
      .join("");
    return `<div class="lb-model">
      <div><span class="k">Model type</span><b>${escapeText(raw.type_candidate || raw.vehicle_type || "unknown")}</b>
        <span class="muted">${Number(raw.type_candidate_confidence || 0).toFixed(2)} · ${escapeText(raw.type_source || "")}</span></div>
      <div class="lb-probs">${bar(raw.type_probs)}</div>
      <div><span class="k">Model colour</span><b>${escapeText(raw.vehicle_color || "unknown")}</b>
        <span class="muted">${Number(raw.color_confidence || 0).toFixed(2)} · ${escapeText(raw.color_source || "")}</span></div>
      <div class="lb-probs">${bar(raw.color_probs)}</div>
    </div>`;
  }

  function render() {
    fillOptions();
    const row = current();
    const stage = $("lbStage");
    if (!row) { stageMessage("Queue finished. Load another scope."); renderProgress(); return; }
    const verdict = verdictFor(row);
    const crop = L.contextCropUsed && row.context_evidence_path
      ? row.context_evidence_path : row.evidence_path;

    stage.className = "";
    stage.innerHTML = `<div class="card lb-card">
      <div class="lb-main">
        <figure class="lb-crop">
          ${crop ? `<img src="${evidenceUrl(crop)}" alt="vehicle crop" />` : '<div class="muted">no crop stored</div>'}
          <figcaption class="muted">
            ${escapeText(row.camera_id)} · ${escapeText(row.first_seen_at_ist || formatWhen(row.first_seen_at))}
            · id ${row.id}${L.contextCropUsed ? " · context frame (m)" : " · tight crop (m)"}
            ${row.review_status === "verified" ? ' · <b>already labelled</b>' : ""}
          </figcaption>
        </figure>
        ${modelPanel(row)}
      </div>

      <div class="lb-keys">
        <div class="lb-row"><span class="k">Type</span>
          ${Object.entries(TYPE_KEYS).map(([k, v]) => keycap(k, v, verdict.vehicle_type === v)).join("")}
        </div>
        <div class="lb-row"><span class="k">Colour</span>
          ${Object.entries(COLOR_KEYS).map(([k, v]) => keycap(k, v, verdict.vehicle_color === v)).join("")}
        </div>
        <div class="lb-row"><span class="k">Controls</span>
          ${keycap("a", "accept model")}${keycap(" ", "save + next")}${keycap("x", "skip")}
          ${keycap("c", "clear")}${keycap("z", "undo")}${keycap("m", "other crop")}
        </div>
      </div>
    </div>`;

    stage.querySelectorAll("[data-key]").forEach((button) => {
      button.addEventListener("click", () => handleKey(button.dataset.key));
    });
    renderProgress();
  }

  // ------------------------------------------------------------- keyboard --
  function labelPanelVisible() {
    const panel = document.querySelector('[data-panel="label"]');
    return panel && !panel.classList.contains("hidden");
  }

  function handleKey(key) {
    const row = current();
    if (!row) return;
    if (TYPE_KEYS[key]) { setVerdict(row, "vehicle_type", TYPE_KEYS[key]); return; }
    if (COLOR_KEYS[key]) { setVerdict(row, "vehicle_color", COLOR_KEYS[key]); return; }
    if (key === "a") { acceptModel(row); return; }
    if (key === " " || key === "Enter") { commit(); return; }
    if (key === "ArrowRight") { advance(1); return; }
    if (key === "ArrowLeft") { advance(-1); return; }
    if (key === "x") { skip(); return; }
    if (key === "c") { clearVerdict(); return; }
    if (key === "z") { undo(); return; }
    if (key === "m") { L.contextCropUsed = !L.contextCropUsed; render(); }
  }

  document.addEventListener("keydown", (event) => {
    if (!labelPanelVisible()) return;
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    const tag = (document.activeElement && document.activeElement.tagName) || "";
    if (["INPUT", "SELECT", "TEXTAREA"].includes(tag)) return;
    const key = event.key;
    if (TYPE_KEYS[key] || COLOR_KEYS[key] || CONTROL_KEYS.has(key)) {
      event.preventDefault();
      handleKey(key);
    }
  });

  // ------------------------------------------------------------------ boot --
  // addEventListener, NOT .onclick: console.js owns .onclick on every .tab.
  // Labelling needs the crop as large as possible, so it takes the whole width
  // and the map column steps aside while it is open.
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelector(".layout")
        ?.classList.toggle("layout-wide", tab.dataset.tab === "label");
    });
  });

  document.querySelector('.tab[data-tab="label"]')?.addEventListener("click", () => {
    fillOptions();
    if (!$("lbStart").value) {
      const end = new Date();
      const start = new Date(end.getTime() - 7 * 24 * 3600 * 1000);
      const local = (d) => new Date(d.getTime() - d.getTimezoneOffset() * 60000)
        .toISOString().slice(0, 16);
      $("lbStart").value = local(start);
      $("lbEnd").value = local(end);
    }
  });

  $("btnLabelStart")?.addEventListener("click", () => loadQueue(false));
  $("btnLabelBlind")?.addEventListener("click", (event) => {
    L.blind = !L.blind;
    event.currentTarget.textContent = L.blind ? "Show model reading" : "Hide model reading";
    event.currentTarget.setAttribute("aria-pressed", String(L.blind));
    render();
  });
})();
