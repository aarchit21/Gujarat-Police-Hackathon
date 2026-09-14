# Gujarat Police Vehicle Investigation Platform

**Gujarat Police Hackathon — CCTV Integration Challenge.** A unified camera
registry, live vehicle analytics and watchlist alerting for 26 departments,
designed to scale from 32 cameras today to ~80,000 statewide.

This is **not** a statewide VMS and **not** an 80,000-camera load test. It is a
running reference implementation with measured accuracy and stated limits.

## Quick start

```bash
conda activate gujhac                      # or: python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/pull_vehicle_attributes.py  # OpenVINO attribute weights (~22 MB)
python scripts/seed.py                     # cameras + representative watchlist

APP_ENV=production ENABLE_DEVELOPER_UI=false \
  python -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

The server prints the URL to open. Sign in with the operator access token —
`p0-operator` by default; see [Operator access token](#operator-access-token).

Optional ANPR stack (plate detection and OCR):
`pip install -r requirements-cpu-anpr.txt` then
`python scripts/setup_cpu_anpr.py`. The platform detects, tracks and describes
vehicles without it.

## Submission documents

Everything for the hackathon submission is in [`final_report/`](final_report/):

| File | What it is |
|---|---|
| `Gujarat_Police_Hackathon_Solution_Presentation.pptx` | 16-slide solution presentation — model choice and justification, solution overview, key features, measured accuracy, scaling to 80,000 |
| `Gujarat_Police_Hackathon_HLD.docx` | High-Level Design / technical proposal — architecture, integration, analytics, security, full scalability plan, department information requirements |
| `DEMO_VIDEO_SCRIPTS.md` | Shot-by-shot scripts for both required demo videos (own feed, government feed) |
| `diagrams/` | The five architecture diagrams embedded in the HLD |
| `_build_*.py` | Regenerate the documents after a re-measurement rather than hand-editing them |

```bash
pip install -r requirements-dev.txt
cd final_report && python _build_diagrams.py && python _build_presentation.py && python _build_hld.py
```

---

Official solution: **Customised Model 5 Hybrid Architecture**

- **Model 1** is always on: vendor-neutral camera registry and GIS.
- **Model 2** is the first feed-to-alert path: unified viewing of accessible feeds plus ANPR metadata, without replacing departmental recording.
- Regional/edge and remote GPU processing are **deployment strategies inside this hybrid**, not a new official model.
- **Do not build a Central VMS.** Full video stays in departmental stores. The centre stores plate crops, metadata, and authorised incident references.

Working story:

> Onboard heterogeneous feeds into a vendor-neutral registry, persist reviewable ANPR sightings, exact-match an authorised watchlist, and show timestamped evidence plus **inferred** GIS movement, while departmental recording stays local.

## Vehicle type and colour (deterministic, no LLM/VLM)

Vehicle detection, type and colour are the core function and run **independently of
ANPR**. A missing, unreadable or failed plate never removes a vehicle observation and
never changes its type or colour.

```
frame -> yolov8n (COCO 2,3,5,7; FP16 on CUDA, FP32 on CPU, batch 1)
      -> ByteTrack persistent track ids          (needs `lap`; greedy-IoU fallback otherwise)
      -> crop-quality gate                        (size, sharpness, exposure, clipping, visibility)
      -> best N crops per track
      -> OpenVINO vehicle-attributes-recognition-barrier-0042   (type[4] + colour[7] probabilities)
      -> quality x confidence weighted aggregation, type and colour separately
      -> abstention gate (min prob, top-2 margin, temporal agreement, min observations)
      -> optional ANPR (off by default)           -- cannot affect type or colour
      -> ONE observation per track -> DB + JSONL + best crop + context frame
```

The previous Ollama/Gemma path is gone. It produced unsupported attributes — including a
rule that promoted a COCO `car` into `suv`/`taxi_cab`/`auto_rickshaw`/`van` on the model's
say-so — and on this host it logged **137 consecutive authentication failures with zero
successes**. Vehicle attributes now come only from `app/services/vehicle_attributes.py`.

### What is genuinely supported

| | Emitted | **Never emitted** (no weight on this host supports it) |
|---|---|---|
| Type | `car`, `van`, `truck`, `bus` (OMZ) · `two_wheeler` (COCO id 3) | `suv`, `auto_rickshaw`, `taxi_cab` → always `unknown` |
| Colour | `white`, `gray`, `yellow`, `red`, `green`, `blue`, `black` | `silver`, `brown`, `orange`, `other` → always `unknown` |

`gray` is never rewritten to `silver`, and `car` is never promoted to `suv`. Those labels
remain filterable so historic rows stay searchable, but the API marks them unsupported.

Model: Intel Open Model Zoo `vehicle-attributes-recognition-barrier-0042`, **Apache-2.0**.
Input `[1,3,72,72]` NCHW **BGR**, raw 0-255, no mean/scale. Output order is pinned in code
and in a test — note the published `accuracy-check.yml` `label_map` is *alphabetical* and is
**not** the output index order; using it silently swaps `car` and `bus`.

Vendor accuracy on **their barrier/toll dataset** (not Indian CCTV): type avg 87.34%
(`bus` only 68.57%), colour avg 82.71% (`yellow` only 61.50%). **No accuracy has been
measured for this deployment** — there is no labelled Indian validation set.

### Install and run

```bash
CONDA=/home/useraakash/miniconda3/envs/gujhac/bin

# Attribute weights (~22 MB, Apache-2.0). Idempotent, records SHA-256.
$CONDA/python scripts/pull_vehicle_attributes.py

# Runtime deps: OpenVINO for the attribute model, lap for ByteTrack.
$CONDA/pip install "openvino>=2024.0" "lap>=0.5.12"

# If torch cannot see the GPU, its CUDA build does not match the driver.
# This host needed cu124 (driver 535.183.01 / CUDA 12.2):
$CONDA/pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0

$CONDA/python scripts/check_host.py      # preflight: torch/CUDA, weights, ByteTrack
```

All paths below are relative to the repository root, so `cd` there first. There is no
`python` on this host's PATH — use the conda interpreter explicitly (or activate the env
with `conda activate gujhac`).

```bash
cd /home/useraakash/guj_pol_archit/Gujarat-Police-Hackathon
CONDA=/home/useraakash/miniconda3/envs/gujhac/bin

# Video file
$CONDA/python scripts/vehicle_pipeline.py --source clip.mp4 --out data/runs/a.jsonl --contact-sheet

# Catalogue camera (credentials applied from .env, never logged)
$CONDA/python scripts/vehicle_pipeline.py --camera cam01 --max-frames 300 --stride 3 --anpr \
    --record data/samples/cam01.mp4 --out data/runs/live.jsonl --contact-sheet

# Directory of images, CPU only
$CONDA/python scripts/vehicle_pipeline.py --source data/frames/cam-surat --device cpu \
    --out data/runs/cpu.jsonl
```

Modes: `baseline` (pretrained, above), `custom` (fine-tuned weights — none exist, so it
exits non-zero naming every missing file and its exact class order), `evaluate` (per-class
P/R/F1, macro-F1, confusion, coverage and abstention rate against a labelled JSONL manifest).

### Measured on this host

RTX A4500, `yolov8n` @ imgsz 640 FP16 CUDA + OpenVINO attributes on CPU, cam01 night feed:

| | |
|---|---|
| Detect + track latency | p50 **5.0 ms**, p95 8.1 ms → ~190 fps compute-only |
| Attribute inference | p50 **1.5 ms** per crop (CPU, OpenVINO) |
| Peak VRAM | **13.0 MB** allocated / 35.7 MB reserved — far under the 2-3 GB target |
| Offline mp4 throughput | 34.3 fps wall-clock at stride 3 |
| CPU-only fallback | 44.6 fps compute-only, 0 MB VRAM, FP16 correctly reported false |

Wall-clock fps on a live RTSP camera is bounded by the stream's own real-time rate, not by
compute; the metrics file reports both numbers separately.

### Measured accuracy — 59 hand-labelled tracks, cam01 night

Ground truth: `data/labels/tracks.jsonl`. Tracks were labelled **blind**, from contact
sheets carrying no model predictions (`scripts/label_tracks.py`), across two disjoint video
segments. 14 of 59 crops were too dark or blurred to call and are excluded as `unclear`.

> **Annotation caveat.** These labels were produced by the assistant that wrote this code,
> from the same crops, not by an independent human annotator. They are a sanity check, not a
> gold standard. The numbers below are the right order of magnitude, not certified.

| | Coverage | Selective precision |
|---|---|---|
| **Vehicle type** (28 supported-class tracks) | 64% | **50%** |
| **Vehicle colour** (42 supported-class tracks) | 71% | **83%** |

On the 17 tracks whose true class the model cannot represent (`auto_rickshaw`, `suv`), it
correctly abstained **82%** of the time; 3 were false attributions (`auto_rickshaw → truck`
twice, `suv → car`).

**Vehicle type is not fit for use on this camera.** Two findings say so:

1. Only 2 of 15 buses were called `bus`; 6 became `truck`. This model gives Indian city
   buses a `bus` probability of ~0.002 while saying `truck` at 0.99.
2. **Raising the confidence threshold makes type precision worse** — 50% at 0.60 down to
   30% at 0.95. The errors are confident and systematic, not noisy, so no threshold reaches
   the 95%-precision objective. Thresholds cannot fix a domain mismatch.

Conflict-policy comparison on the deterministic 20-track segment (all 35 tracks matched):

| `VATTR_CONFLICT_POLICY` | Coverage | Selective precision |
|---|---|---|
| `detector` (default) | 90.0% | 50.0% |
| `abstain` | 55.0% | 36.4% |
| `classifier` | 85.0% | 29.4% |

`detector` wins on both axes — COCO YOLO genuinely separates bus from truck, and abstaining
was discarding the cases it got right. This default was chosen on 20 tracks of the same
data, so it is a fit, not an independent validation.

**Colour is the usable signal**, and it plateaus: real runs at min-prob 0.55 / 0.65 / 0.75
gave 81% / 83% / 81% precision at 78% / 67% / 59% coverage. The dominant error is
`white → red` (4 cases) — night-time brake lights and red signage cast on white bodies.
`silver` is unreachable, so both true silver cars were missed (one → `gray`, one → `red`).

**The 95% precision objective is not reachable with this pretrained model at any threshold.**
Reaching it requires fine-tuning on labelled Indian data.

### Safeguards this measurement forced

| Safeguard | Behaviour |
|---|---|
| Vehicle type is **suppressed** | The operational `vehicle_type` is `unknown` unless a human verified the record or a model clears `type_deployment_gate()`. The gate fails closed and the shipped model is deliberately not listed. |
| Raw candidate always kept | `metadata_json["attributes"]` retains the candidate label, confidence, temporal agreement, model source, reason and full probability vectors, for diagnostics only. |
| Colour is **estimated**, never verified | Exposed, but always carried with `verified=false`, `review_required=true` and its confidence and agreement. |
| Manual correction | `POST /api/vehicle-observations/{id}/review` records a human verdict that wins over the model and is never overwritten by a later frame. The model's answer survives for audit in `review_history`. |
| Search | Colour is an optional filter and every response carries a warning that it both misses and wrongly includes vehicles. Type filters match **verified records only**. Neither is ever mandatory. |
| ANPR | Optional. Every plate failed in the smoke run and all 35 vehicle observations were still persisted with evidence. |

The UI badges each attribute `VERIFIED` / `ESTIMATED` / `UNKNOWN` (colour plus border
weight plus an explicit text label, not colour alone) and renders the measured accuracy
table wherever attributes are shown.

Frozen evidence — labels, predictions, evaluator output and a note on one discarded
contaminated measurement — is in [`docs/poc_evaluation/`](docs/poc_evaluation/README.md).

### Multi-vehicle crops (found after the POC evaluation)

In dense traffic the crop handed to the classifier often contains a *second* vehicle.
Measured on cam01: **243 of 276 crops** came from frames holding more than one vehicle,
**20% of crops contained >25%** of a neighbour, and 5% contained more neighbour than
subject. ByteTrack was not at fault — 0 of 247 track transitions were suspicious.

Two defects followed, both now fixed:

1. `extend_box` grows the detector box 1.6× before the 115→72 centre crop, which reaches
   into the next vehicle. `crop_quality` now measures `foreign_fraction` and rejects a crop
   as `multi_vehicle_crop` above `VATTR_MAX_FOREIGN_FRACTION` (default 0.25).
2. The aggregation weight was `quality.score * probs.max()` computed **per attribute**.
   Because the type and colour heads peak on different crops, the two attributes were
   weighted differently over the same track — which is how one record ended up with one
   vehicle's type beside another's colour. Both attributes now share a single weight per
   crop.

Re-measured on the 35-track offline segment (a different basis from the headline table
above, so not comparable row-for-row):

| | Coverage before → after | Precision before → after |
|---|---|---|
| Vehicle type | 55% → 60% | **36% → 67%** (4/11 → 8/12) |
| Vehicle colour | 78% → 63% | 81% → 71% (17/21 → 12/17) |

Type improved materially. Colour moved by two tracks on n≤21, which is not distinguishable
from noise. Thresholds were deliberately **not** retuned to improve that, because tuning on
the evaluation set would invalidate it.

### Exporting reviews for future fine-tuning

```bash
$CONDA/python scripts/vehicle_pipeline.py --mode export-reviews --out data/labels/reviews.jsonl
```

Dumps human-verified observations with the crop, box, corrected label and the model's
original answer. It trains nothing. Two cautions it prints: split by `video` (adjacent
frames of one vehicle are near-duplicates and will inflate accuracy), and drop rows with
`crop_foreign_fraction > 0` — a two-vehicle crop teaches the wrong thing.

### Camera slot rotation

Slot selection used to sort by id with `PIN_DEFAULT` forced first, and strict decode
tiering meant untested cameras never got a slot — so they stayed untested forever. On this
catalogue that left **25 of 32 cameras unreachable** while the same 6 were pinned on every
click. Selection now rotates by least-recently-hunted within decode tiers, and
`HUNT_EXPLORE_FRACTION` (default 0.5) reserves half the slots for untested cameras.
Measured: 8 clicks at 4 slots now reach **20 distinct cameras** instead of 4.
`HUNT_ROTATION=fixed` restores the old behaviour.

Reproduce:

```bash
cd /home/useraakash/guj_pol_archit/Gujarat-Police-Hackathon
CONDA=/home/useraakash/miniconda3/envs/gujhac/bin
$CONDA/python scripts/label_tracks.py --runs data/runs/offline data/runs/final --out data/labels
# fill in data/labels/tracks.jsonl, then:
$CONDA/python scripts/vehicle_pipeline.py --mode evaluate --manifest data/labels/tracks.jsonl \
    --predictions data/runs/offline/observations.jsonl data/runs/final/observations.jsonl
```

## Interfaces: production console and developer console

There are two front ends, and only one of them is served by default.

| | Production console | Developer console |
|---|---|---|
| URL | `/` | **`/dev`** |
| Served when | always | `APP_ENV` is not `production` **and** `ENABLE_DEVELOPER_UI=true` |
| Files | `app/static/` (public mount) | `app/devui/` (**not** mounted; served by a gated route) |
| Audience | investigators | engineers |

**`/` is always the production console, in both modes.** The developer console
is at `/dev`, never at `/`. Run either of these and the server prints the URLs
it is actually serving:

```bash
# Production — what you demo
APP_ENV=production ENABLE_DEVELOPER_UI=false \
  python -m uvicorn app.main:app --host 127.0.0.1 --port 8010
#   production console   http://127.0.0.1:8010/
#   /dev returns 404

# Development — your own tooling, plus the production console alongside it
APP_ENV=development ENABLE_DEVELOPER_UI=true \
  python -m uvicorn app.main:app --host 127.0.0.1 --port 8010
#   production console   http://127.0.0.1:8010/
#   developer console    http://127.0.0.1:8010/dev
```

In developer mode the production console also shows a **Developer console**
link in its top bar. In production that element is never created and the route
404s.

### Production console (`/`)

An investigator's workflow: overview → cameras → vehicle search → observation
detail → manual review → export. It asks for the operator access token at
sign-in (the token is no longer embedded in the JavaScript) and holds it for
the browser tab only. Evidence images are fetched with that header and shown
from object URLs, so no credential is ever written into the DOM or into a URL.

What it will and will not say:

* **Vehicle type is always `Unknown`** unless a person verified it. The type
  deployment gate has not been cleared, so no automatic type reaches the
  screen — including on the ~1,500 legacy rows that still hold one.
* **Colour is shown as an estimate**, and only when it came from the
  deterministic OpenVINO attribute model. Values written by the retired VLM
  path or the OpenCV HSV baseline are presented as `Unknown` with a note.
* **Unsupported classes are never shown as predictions.** `suv`,
  `auto_rickshaw`, `silver`, `orange` and the rest are stored and exported but
  displayed as `Unknown`. A person may still record one as a verdict.
* **A plate failure never removes the vehicle.** Every observation carries one
  of: the plate text (marked *Estimated*), `Plate unreadable`,
  `Plate not visible`, or `Not checked`, each with a plain-language reason.
* **A camera is only `Available` once this host has opened its stream.**
  `Not checked` is its own state, never rounded to online or offline.
* Backend failures show one sentence plus a reference; the detail is logged
  server-side.

### Two feed paths, shown separately

The console splits every figure by where the footage came from, because the two
paths demonstrate different things and one combined number credits each with the
other's results:

| | Own test feed | Government cameras |
|---|---|---|
| Source | frames generated on this host (`scripts/generate_own_feed.py`) | authorised live RTSP |
| Demonstrates | plate read → watchlist match → **alert**, end to end | detection, tracking and colour on real traffic at scale |
| Action | *Analyse own feed* | *Check connections*, *Start monitoring available* |

The own feed is **synthetic**, so it is labelled a **test feed** everywhere it
appears, and an alert raised on it carries an explicit line saying the match is
real but the vehicle is not. Overview has a *Feed sources* panel, Cameras has a
Feed column and filter, and Vehicle search has a `Feed source` filter
(`?feed=own|government`) whose two halves always sum to the unfiltered total.

### Developer console (`/dev`)

The previous console, unchanged plus a labelling tab: raw JSON, probability
vectors, model ids and hashes, decode/FPS panels, detector and OCR diagnostics,
threshold controls, the worker manager, the capacity and cost estimator.
Nothing was deleted — it is simply not served in production, and neither are the
routes it depends on:

| Route | Production | Developer |
|---|---|---|
| `GET /dev`, `/dev/console.{js,css}` | 404 | 200 |
| `GET /api/recognition/diagnostics` | 404 | 200 |
| `GET /api/diagnostics/{camera_id}` | 404 | 200 |
| `GET /api/cameras` | no stream URIs, no raw decoder errors | full payload |
| `GET /api/vehicle-observations/{id}` | no `metadata`, no `raw` block | probability vectors included |
| `GET /api/vehicles/{plate}`, `/api/alerts` | no model id / hash / run id | full payload |
| `GET /api/dev/label-queue` | 404 | 200 |

### Bulk labelling (`/dev` → Labelling)

Building the fine-tuning set. The production search **cannot** find the rows
worth labelling, by design: the type deployment gate forces `vehicle_type` to
`unknown` in the column on all 3,569 gated rows (the model's answer survives only
in `metadata_json.attributes.type_candidate`), and the colour filter hides
values from retired sources. Measured on the current database:

| filter | production `/api/investigations/vehicles` | developer `/api/dev/label-queue` |
|---|---|---|
| `vehicle_type=truck` | 0 | **576** |
| `vehicle_type=bus` | 0 | **379** |
| `vehicle_color=white` | 371 | **948** |
| `vehicle_color=silver` | 0 | **343** |
| `color_source=legacy` | not filterable | **1,584** |

So the queue matches the **raw** model output, on a route that returns 404 in
production. The 59 frozen evaluation tracks in `data/labels/tracks.jsonl` are
excluded, so the published 50%/83% figures stay a held-out measurement.

Keyboard-driven — digits for type, letters for colour, no mode switch:

| | |
|---|---|
| Type | `1` car · `2` suv · `3` two-wheeler · `4` truck · `5` bus · `6` van · `7` auto-rickshaw · `8` taxi · `0` unknown |
| Colour | `w` white · `k` black · `s` silver · `e` grey · `r` red · `b` blue · `g` green · `y` yellow · `o` orange · `n` brown · `t` other · `u` unknown |
| Control | `a` accept the model's reading · `Space` save + next · `→` next · `←` back · `x` skip · `c` clear · `z` undo · `m` tight/context crop |

Every chip prints its key on screen. Saves are optimistic with a serial outbox,
so typing never waits on the network; a failed save lands in a **Retry failed**
strip rather than disappearing. `z` is a real undo — it re-posts the previous
values (`""` to clear), the same path the API already supported.

**What the labels are worth.** The model's reading is shown by default (there is
a *Hide model reading* toggle). That is much faster, and it is the right
trade-off for fine-tuning — but an annotator who sees "truck 0.83" agrees with it
more often, so these labels tend to reproduce the model's *systematic* errors:
Indian buses read as trucks, white read as red under brake lights, both named in
`POC_ACCURACY["notes"]` as its real failure modes. Every verdict therefore
records how it was produced, and `--mode export-reviews` reports it:

```
python scripts/vehicle_pipeline.py --mode export-reviews --out data/labels/reviews_export.jsonl
```

Each row carries `provenance` (`blind` / `assisted` / `unknown`),
`prediction_visible`, `accepted_model`, `context_crop_used`, `annotator` and the
`scope` that was being swept. Rows whose verdict is `unknown` are marked
`usable: false` — an abstention is not a training target, and it would otherwise
walk into the export as an example labelled "unknown". Use assisted labels to
**train**, never to re-measure accuracy.

Recognition attempts, observation metadata, audit events and the frozen
evaluation artefacts are **still written** in production. The gates control who
can read them back, not whether they exist. CSV and GeoJSON evidence exports
still carry the model identifiers, because an exhibit has to say which model
produced a reading.

To turn the developer console on:

```
APP_ENV=development
ENABLE_DEVELOPER_UI=true
```

Both are required. `/dev` stays unreachable while `APP_ENV=production`, so a
production deployment cannot expose it by setting one variable by mistake.

`/dev` itself checks only the developer flag, not the token. The console it
serves still has to sign in for every API call it makes, so an unauthenticated
visitor gets an empty shell — but treat the flag as "this machine is a
developer machine", not as an access control.

## Operator access token

### The value

**`p0-operator`** — the default, used unless you override it.

It is the `ADMIN_TOKEN` setting (`Settings.admin_token` in
[`app/config.py`](app/config.py)). It is a **shared demo token committed as a
default**, not a secret. Change it before this runs anywhere real.

### Signing in

Open the console at `/`, type the token into **Operator access token**, and
press Sign in. The console validates it against `GET /api/settings/recognition`
(an operator-only route), so a wrong token is rejected at the door rather than
half way through an investigation.

The token is held in `sessionStorage` for that browser tab only:

* it survives a page reload, so a refresh does not sign you out;
* it is gone when the tab closes, and **Sign out** in the top bar clears it;
* it travels only in the `Authorization: Bearer …` header — never in a URL, a
  link, or the page itself. Evidence images and CSV/GeoJSON exports are fetched
  with the same header and handed to the browser as object URLs.

Previously `app/static/app.js` contained `const TOKEN = "p0-operator"`. It no
longer does, and `tests/test_security_source.py` fails if a token is ever
re-embedded in a file served from `/static`.

### Changing it

Set it in `.env` (or the environment) and restart the server — it is read once
at startup:

```
ADMIN_TOKEN=<your-token>
VENDOR_INGEST_TOKEN=<token-for-vendor-event-POSTs>
```

```bash
# verify the new value is the one being enforced
curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer <your-token>" \
  http://127.0.0.1:8000/api/settings/recognition     # expect 200
curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer wrong" \
  http://127.0.0.1:8000/api/settings/recognition     # expect 401
```

Anyone already signed in keeps working until their tab is closed or a request
returns 401, at which point the console returns them to the sign-in screen with
"Your session has ended."

### What it authorises

The token is required for **everything that writes, exports or reveals
evidence** — 31 routes, including:

| | |
|---|---|
| Review and enforcement | `POST /api/vehicle-observations/{id}/review`, `PATCH /api/alerts/{id}`, `POST`/`PATCH /api/watchlist…` |
| Camera operations | `POST /api/cameras`, `/api/cameras/{id}/analyze`, `/api/workers/…`, `/api/hunt/…`, `/api/capacity/measure` |
| Evidence and exports | `GET /api/evidence`, `/api/cameras/{id}/snapshot`, `…/export.csv`, `…/export.geojson`, `/api/reports/…` |
| Settings | `GET`/`PATCH /api/settings/recognition` |

Two other credentials exist and are separate:

* `VENDOR_INGEST_TOKEN` (default `p0-vendor`) — only `POST /api/vendor/events`.
* `CCTV_ACCESS_TOKEN` / `CCTV_ACCESS_USERNAME` — the **government feed**
  credentials. Server-side only. They are never sent to the browser, and
  `redact_secrets()` strips them from anything logged.

### Known limitations of this scheme

Honest, because a P0 auth model should not be mistaken for a production one:

* **Read endpoints are open.** `GET /api/cameras`, `/api/alerts`,
  `/api/investigations/vehicles`, `/api/ui/overview` and the rest of the list
  above answer without a token. The production payloads carry no stream URIs,
  no credentials and no model identifiers, but anyone who can reach the port
  can read the observation list. Put the service behind a network boundary or
  an SSH tunnel — do not expose `0.0.0.0` to an untrusted network.
* **One shared token, so there is no per-person identity.** Every audit entry
  and every `verified_by` records the actor as `operator`. "Who confirmed this
  colour" is answerable only down to "someone holding the operator token".
  Real accounts are out of P0 scope.
* **`REQUIRE_AUTH=false` weakens it in a specific way**: a request with *no*
  token is accepted, while a request with a *wrong* token is still rejected.
  Leave it `true`.
* There is no expiry, rotation or revocation. Changing `ADMIN_TOKEN` and
  restarting is the whole rotation procedure.

## Architecture

A camera uses one processing mode:

| Mode | Meaning |
|---|---|
| `vendor_metadata` | Consume authorised ANPR/event metadata from the camera or VMS |
| `local_worker` | CPU FastALPR/FastPlateOCR, Tesseract uncertainty check, queued Ollama verification |
| `remote_gpu` | Send **selected JPEG frames** to `REMOTE_INFERENCE_URL` |
| `shared_regional` | Represented as a shared worker (same local path in P0) |
| `central_on_demand` | Process only when an operator starts the worker |
| `deferred` | Registered, but analytics cannot run yet |

Priority classes: **A** continuous critical, **B** continuous or vehicle-triggered, **C** scheduled/on-demand, **D** registry/health only until infrastructure exists.

A GPU is **not** required in every district. This host does **not** pull 80,000 live feeds into one process.

## Database: PostgreSQL/PostGIS production, SQLite fallback

PostgreSQL with PostGIS is the production and scale target. SQLite is an explicit **local-development and automated-test fallback**. It is not the statewide database.

Configure with `DATABASE_URL`. Docker is **not** required.

Native PostgreSQL example (install PostgreSQL + PostGIS on the host, create a database, then):

```powershell
$env:DATABASE_URL = "postgresql+psycopg://USER:PASSWORD@127.0.0.1:5432/cctv"
$env:DB_POOL_SIZE = "5"
$env:DB_MAX_OVERFLOW = "10"
```

Create the database yourself, for example:

```sql
CREATE DATABASE cctv;
CREATE EXTENSION IF NOT EXISTS postgis;
```

On startup the app:

- creates tables
- **adds missing columns** (does not drop user data)
- creates indexes on camera id, `plate_norm`, `source_time`, watchlist, and alert status
- enforces alert dedup uniqueness `(watchlist_id, camera_id, passage_id)`
- attempts PostGIS (`geom geography(Point,4326)` + GIST) when the dialect is PostgreSQL

Time-based partitioning of `sightings`, replication, backups, and retention are documented in `app/sql/postgres_scale.sql` as production recommendations. They are not auto-applied and have **not** been load-tested at 80,000 cameras.

The health endpoint and UI show the active database type. SQLite is labelled as a dev fallback.

## All-day demo (vehicle monitoring)

Stop any `--reload` server (reload kills workers). Then:

```powershell
# .env: CCTV_ACCESS_TOKEN. Local Ollama needs OLLAMA_URL=http://127.0.0.1:11434 and an empty OLLAMA_API_KEY.
$env:DEMO_AUTOSTART_WORKERS = "true"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Leave that window open all day. Open http://127.0.0.1:8000 and sign in with the
operator token.

**Start monitoring available** (Cameras page) holds open the configured capture
slots; it does not claim every catalogue camera is decoded simultaneously. CPU
FastALPR handles the immediate plate path. Uncertain high-quality crops enter
one bounded background Ollama queue, so cloud latency does not consume a
capture slot. The rotating **Hunt** controls live in the developer console.

- **Overview** — cameras monitoring / available / not checked, vehicles in the
  last 24 hours, how many await review, open alerts. Every figure is counted
  from the database; nothing is estimated.
- **Vehicle search** — time range, camera, colour, plate state and review
  state → results with the best crop, attributes, plate state and review
  action. CSV export.
- **Vehicle movement** — plate + optional date → sightings grouped into stops,
  numbered on the map, joined by dashed **inferred** links (not a proven
  route), CSV export.
- **Alerts / Watchlist** — exact match only; own-feed `GJ01AB1234` is the
  guaranteed demo hit.
- **Cameras** — status, check connections, start/stop monitoring, analyse
  recorded video, single live frame, add a camera.

Snap-to-road possible paths (OSRM Match) and the GeoJSON export remain
available through `/api/vehicles/{plate}/export.geojson`.

No Google Maps API. Default matching is the public OSRM Match server (no credit card). Optional Mapbox / Geoapify tokens stay in `.env` and are never sent to the browser.

## Run (Linux GPU server, conda env `gujhac`)

Python 3.11, RTX A4500 (20 GB). Tesseract is installed into the env (no sudo):

```bash
create and activate a virtual environment
conda install -c conda-forge "tesseract>=5" -y
python -m pip install -r requirements.txt
python -m pip install -r requirements-cpu-anpr.txt
python -m pip uninstall -y opencv-python   # keep opencv-python-headless only
python -m pip install --force-reinstall opencv-python-headless
python scripts/setup_cpu_anpr.py
# After setup succeeds, set CPU_ANPR_MODELS_READY=true in .env
python scripts/pull_yolo.py
ollama pull qwen3-vl:8b
python scripts/check_host.py
python scripts/seed.py
# This GPU box already binds 8000–8003 for other services; 8010 is free.
python -m uvicorn app.main:app --host 0.0.0.0 --port 8010
```

Open http://\<server\>:8010 (or SSH-tunnel to 127.0.0.1:8010). Do not use `--reload` for the all-day demo.

`.env` for this host:

```
OLLAMA_URL=http://127.0.0.1:11434
OLLAMA_API_KEY=
OLLAMA_VISION_MODEL=qwen3-vl:8b
OLLAMA_VISION_ENABLED=true
```

A leftover Cloud API key with a localhost URL no longer redirects to ollama.com. Cloud is used only when `OLLAMA_URL` is `https://ollama.com`.

## Run (Windows)

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-cpu-anpr.txt
python scripts/setup_cpu_anpr.py
# After setup succeeds, set CPU_ANPR_MODELS_READY=true in .env
python scripts/check_host.py
python scripts/seed.py
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000

Sign in with the operator access token — `p0-operator` by default. See
[Operator access token](#operator-access-token) for how to change it and what
it authorises.

1. **Cameras → Analyse recorded video** — OpenCV frames + CPU ANPR on the generated Ahmedabad and Surat plates.
2. **Vehicle movement** → `GJ01AB1234` — sightings grouped into stops, evidence crops, dashed inferred path.
3. **Alerts** — acknowledge / confirm / reject. Coverage stays honest: only feeds that are actually being processed are `analytics_active`.

Own-feed verification:

```powershell
python scripts/verify_own_feed.py
python -m pytest -q
```

## Live ANPR (CPU plate detector to local OCR to optional cloud verifier)

1. FastALPR's plate-specific ONNX detector runs with `CPUExecutionProvider`; YOLOv8n remains a vehicle-region fallback and ignores people.
2. FastPlateOCR reads the native plate crop. Uncertain results are checked by deterministic Tesseract variants; disagreement is review-only.
3. A bounded one-worker queue sends only uncertain crops to Ollama. Vision-only frames are isolated from auto-confirmation because they have no tracked plate box.
4. An alert requires the same strict plate on two distinct tracked frames before exact watchlist matching.

```powershell
python -m pip install ultralytics
python scripts/pull_yolo.py
```

First run copies or downloads `yolov8n.pt` (~6 MB) into `data/models/` (gitignored). If ultralytics/torch is missing, the worker falls back to the OpenCV blob crop. Detection can use CUDA or CPU (`YOLO_DEVICE=auto`). On the 20 GB A4500, YOLOv8n plus `qwen3-vl:8b` should sit around 8–10 GB.

**Local Ollama** (no API key). On a 20 GB GPU use `qwen3-vl:8b` (~6 GB download, ~8 GB VRAM):

```powershell
$env:OLLAMA_URL = "http://127.0.0.1:11434"
$env:OLLAMA_VISION_MODEL = "qwen3-vl:8b"
$env:OLLAMA_VISION_ENABLED = "true"
```

**Ollama Cloud** (no local `ollama serve`; **API key required**):

1. Create a key at https://ollama.com/settings/keys
2. Put it only in `.env` (gitignored), never in source or the UI:

```powershell
$env:OLLAMA_URL = "https://ollama.com"
$env:OLLAMA_API_KEY = "<your cloud key>"
$env:OLLAMA_VISION_MODEL = "gemma4:31b"
$env:OLLAMA_VISION_ENABLED = "true"
```

Cloud is used only when `OLLAMA_URL` is `https://ollama.com`. An explicit local URL is never rewritten, even if a leftover `OLLAMA_API_KEY` is present. Cloud requests send `Authorization: Bearer <key>`. Local requests send no key. Retired Cloud models (`gemma3:*`, `llava:*`) map to `gemma4:31b`. Cloud vision uses `/api/chat`.

Records use `model_id=ollama:<actual-model>`. The prompt never includes the watchlist. The key is never returned by `/api/health`. This is not YOLO, Awiros, or PP-OCRv5.

## CPU ANPR and fail-safe confirmation

- FastALPR `yolo-v9-t-384-license-plate-end2end` + FastPlateOCR `cct-s-v2-global-model`, pinned in `requirements-cpu-anpr.txt`
- CPU execution only; model downloads happen in `scripts/setup_cpu_anpr.py`, not inside an unprepared live worker
- Native-pixel quality gates, strict Indian/BH/Gujarat-government syntax, per-character confidence, IoU/centroid plate tracks, and two-frame confirmation
- Tesseract checks uncertain crops; OpenCV enhancement is deterministic and non-generative
- Raw outputs, quality, latency, reason, model hash, and evidence appear under `/api/recognition/diagnostics`
- Layout substitutions and fuzzy candidates are review suggestions only, never automatic match keys

Run `python scripts/benchmark_anpr.py` to measure this host. Its output explicitly does not claim camera capacity.

## Optional remote GPU provider

```powershell
$env:REMOTE_INFERENCE_URL = "https://gpu.example.internal/infer"
$env:REMOTE_INFERENCE_TOKEN = "..."
$env:REMOTE_INFERENCE_TIMEOUT_SECONDS = "8"
$env:REMOTE_INFERENCE_ALLOWED_HOSTS = "gpu.example.internal"
$env:REMOTE_FALLBACK_LOCAL = "true"
```

The worker sends **selected JPEG frames** (PTS-sampled), not an unrestricted live copy. The host must be on the allowlist (SSRF control). Expected JSON:

```json
{"plate_text":"GJ01AB1234","confidence":0.9,"model_id":"...","model_hash":"...","bbox":[x,y,w,h]}
```

Production never invents a remote plate. Tests mock the HTTP endpoint. On failure the error is recorded; the frame is skipped.

## Vendor metadata path

`POST /api/vendor/events` with `Authorization: Bearer <VENDOR_INGEST_TOKEN>`.

Required: `event_id`, `camera_id`, `source_time`, `plate_raw`, `confidence`, `vendor_model_id`. Payload size is limited. Events must provide a stable `passage_id` and distinct frame index or PTS to satisfy the same two-frame gate. The second confirmed persisted sighting may exact-match and open an alert. Replayed `event_id` values are rejected.

## Workers

```
POST /api/workers/{camera_id}/start
POST /api/workers/{camera_id}/stop
POST /api/workers/stop-all
GET  /api/workers
```

Bounded in-process threads. No Kafka, Kubernetes, or distributed scheduler. Concurrent local/remote workers and open captures are capped (`MAX_CONCURRENT_WORKERS`, `MAX_OPEN_CAPTURES`). Overflow cameras are **queued** and remain `analytics_active=false`. Duplicate workers for the same camera are refused. `analytics_active` is true only while frames are actually being processed.

Day 3 (Workers tab):

1. **Measure government decode** — sequential RTSP probe of at most `MAX_CONCURRENT_CAPTURES` catalogue cameras. Records who actually decoded. If measured throughput is below the FPS hypothesis, sampling is lowered. This is not an 80,000-camera test.
2. **Start accessible workers** — starts only cameras that already decoded (plus own-feed). Extra cameras queue and stay `analytics_active=false`.

## Government-feed ingest (authoritative catalogue)

The organiser catalogue is:

`https://cctv.corp8.cloud/cameras.json`

That file is the runtime source of truth for **which cameras exist**. If an entry already includes RTSP/WHEP/HLS URLs, those URLs are used as-is. If `cameras.json` only returns `id` and `name` (the current organiser payload), the documented stream contract is applied to that catalogue id:

- RTSP: `rtsp://103.250.160.189:8554/stream/<id>`
- WHEP: `http://103.250.160.189:8889/stream/<id>/whep`
- HLS: `https://cctv.corp8.cloud/<id>/index.m3u8`

Camera IDs are never guessed. The app does not scan `cam01`–`cam30` unless those ids are in the catalogue.

```powershell
$env:INGEST_CATALOGUE_URL = "https://cctv.corp8.cloud/cameras.json"
$env:CCTV_AUTH_MODE = "none"   # or bearer | basic | custom_header
$env:CCTV_ACCESS_TOKEN = "<rotated token — not stored in git>"
$env:RTSP_TRANSPORT = "tcp"
$env:MAX_CONCURRENT_CAPTURES = "4"
```

Authentication is configurable. Do not assume Bearer unless that is the configured mode.

| Mode | Behaviour |
|---|---|
| `none` | GET with no credentials |
| `bearer` | `Authorization: Bearer <CCTV_ACCESS_TOKEN>` |
| `basic` | HTTP basic with `CCTV_ACCESS_USERNAME` and `CCTV_ACCESS_TOKEN` |
| `custom_header` | Header `CCTV_AUTH_HEADER_NAME: <CCTV_ACCESS_TOKEN>` |
| `form` | Optional same-origin login POST if the portal uses a password form (not a stream-control API) |

The token is never written to the UI, API responses, logs, tests, or git. Put it in `.env` (gitignored) or the process environment.

The app **only** discovers cameras with `GET` that URL (after optional login). It never publishes streams and never calls a gateway control API.

The official brief mentions approximately 50 cameras; the current organiser document describes cam01–cam30. This P0 imports **whatever cameras.json returns** and reports that count. It does not invent cameras to reach 50. Own-feed cameras are seeded separately.

HTML/login pages, HTTP 401/403, timeouts and invalid JSON are treated as catalogue failures, not as a camera list.

Sync: `POST /api/catalogue/sync`

- Upserts by catalogue camera ID
- Adds newly available cameras
- Marks missing cameras unavailable **without deleting history**
- Preserves local priority, processing mode, and analytics policy
- Stores `catalogue_live` separately from tested `decode_status`
- **Never** treats `catalogue_live=true` as `analytics_active=true`
- Does not expose protected RTSP credentials to frontend JavaScript

Government streams are **live-only**. They cannot be downloaded, sought, or processed faster than real time. This P0 never writes every decoded frame to disk and never loads an entire stream into memory.

### RTSP over TCP

Before `VideoCapture`:

```python
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
```

A separately installed FFmpeg **executable** is not required. If port 8554 / RTSP fails, the worker attempts the catalogue HLS URL, records the protocol actually used, and does **not** report RTSP as healthy if only HLS works.

### WHEP / HLS preview

Browser previews are **on demand** (not 50 tiles). WHEP is the low-latency preview; HLS is the dashboard/restricted-network fallback. Preview-active and analytics-active are separate states. A playing preview is **not** proof that ANPR is running. RTSP credentials are never sent to the browser.

### PTS timing

Do **not** use `CAP_PROP_FPS` or frame-arrival time for sampling, dwell, tracking, or speed.

`pts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)` is stored separately from `ingest_time_utc`. `clock_offset_ms` is applied only when known. PTS deltas drive sampling, passage duration, character-vote windows, and discontinuity detection. Reported FPS is informational.

The gateway may replay a buffered GOP so the first frames arrive faster than real time. Arrival time is not used as motion or speed.

### Reconnect

Exponential backoff starting near 2 seconds, capped near 30 seconds, reset after stable decoding, cancelled immediately when the worker is stopped. No tight reconnect loop. `analytics_active=false` while frames are not being processed. Reconnect counts and redacted diagnostics are stored.

Initial H.264/H.265 join warnings (RPS / missing POC) are treated as non-fatal for a bounded keyframe wait. Mixed codecs and resolutions are handled per camera; inference may resize a frame but boxes are scaled back. Mixed-resolution frames are never stacked into one fixed-shape batch.

PTS regression or a large PTS jump ends the current passage, resets the character vote, writes an audit event, and continues. Tracks are never joined across that discontinuity.

## Camera registry onboarding and gap analysis (reference Model 1)

Three onboarding routes, all additive — an import never deletes a camera and never
blanks a field the file omits:

| Route | Endpoint | UI |
|---|---|---|
| API (government catalogue) | `POST /api/catalogue/sync` | **Sync catalogue** button |
| Manual entry | `POST /api/cameras` | Camera ledger → *Onboard cameras* |
| Bulk import | `POST /api/cameras/import` | Camera ledger → *Onboard cameras* → CSV box |

Bulk import accepts CSV text or a JSON list. The header must contain `id` (or
`camera_id`); optional columns are `name, department, city, lat, lng, source_type,
source_uri, substream_uri, priority_class, processing_mode, analytics_policy,
network_class, vendor, model`.

**Health columns cannot be imported.** `decode_status`, `analytics_active` and
`status` are measured by this host, never asserted by a file. Every newly
onboarded camera starts `decode_status=untested`, and `lat`/`lng` must arrive as
a pair or not at all — a half pair is rejected rather than half-guessed.

```bash
curl -X POST http://127.0.0.1:8010/api/cameras/import \
  -H "Authorization: Bearer $CCTV_ADMIN_TOKEN" -H 'Content-Type: application/json' \
  -d '{"csv":"id,name,city,lat,lng,source_type\nCAM-NEW-001,North Gate,Surat,21.1702,72.8311,rtsp\n"}'
```

### Gap-analysis report

`GET /api/reports/gap-analysis.json` and `/api/reports/gap-analysis.csv` (operator
token required) report where the fleet is blind:

- `never_probed`, `decode_failed`, `placeholder_coords`, `coordinates_out_of_bounds`
- `no_vehicle_observations`, `no_plate_reads`
- per-city rollup and `uncovered_cities_no_working_camera`

It is a gap analysis **of the inventory this host holds**, not of road coverage — a
location with no registered camera cannot appear in it. `never_probed` means this
host has not opened the stream yet, not that the camera is down. Camera URLs are
redacted, so the report is safe to attach to a submission.

## Honest government-feed blocker

Government-feed status is whatever this host actually observes after `POST /api/catalogue/sync` and a bounded RTSP probe. Do not claim 50 live government cameras. Do not treat catalogue `live=true` as `analytics_active`. Protected HLS credentials stay server-side; browser HLS preview is blocked unless a safe URL is available. WHEP may be used for on-demand preview. Analytics uses RTSP-over-TCP.

```powershell
python scripts/check_host.py
python scripts/probe_government_feed.py
```

`probe_government_feed.py` syncs the catalogue, tests TCP/HTTPS, opens **one** RTSP camera, and runs a bounded ANPR sample. It does not open every catalogue stream.

## Cost / capacity estimator

The UI posts user-supplied assumptions (camera count, bitrate, target FPS, active cameras, measured worker FPS, GPU hourly cost, storage cost, evidence volume). Results are labelled **estimate only**. No savings percentage is hard-coded.

## Known host limitations

- Windows laptop (Python 3.14, GTX 1650) or Linux GPU server (conda `gujhac`, Python 3.11, RTX A4500 20 GB) — throughput is a measured hypothesis, not a statewide rating
- CPU detector throughput must be measured per camera; Tesseract uncertainty checks are much slower than FastPlateOCR
- No Node.js and no external FFmpeg executable on PATH at plan time
- OpenCV may still use its internal FFmpeg backend for RTSP
- Government catalogue credentials and a real remote GPU endpoint are external blockers
- District GPU infrastructure is **not** deployed by this P0
- The console runs **offline**: Leaflet and hls.js are vendored under
  `app/static/vendor/`. Only the OSM basemap tiles and the web fonts are still
  remote, and both degrade to an on-screen notice rather than a blank page
- **Plate text is not recoverable from the current government feeds.** Vehicle
  detection, tracking, colour and evidence capture all work on them; ANPR does
  not. The registration-number trace is demonstrable on the own feed only
- GIS links are inferred from timestamped sightings, not proven road polylines
- No live VAHAN / NAPIX / NAFIS / face / person ReID
- No Kafka, Kubernetes, or Elasticsearch
- Production legal compliance is not claimed

## Gates

See `docs/FOUR_DAY_P0_PLAN.md` and `AGENTS.md`.

## Layout

- `app/` FastAPI + static Leaflet UI
- `scripts/` seed, own-feed generator, host check, own-feed verify
- `docs/` brief, plan, literature, deck
- `papers/` local evidence pack (`_excluded/` holds the misfiled dropsonde PDF)
- `app/sql/postgres_scale.sql` production scale notes (not auto-applied)
