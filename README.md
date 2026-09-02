# Gujarat CCTV Hybrid P0

Solo four-day hackathon build. **Grok / Codex generates the code.** The human owns government-feed access, gate calls, and demo honesty.

This is **not** a statewide VMS and **not** an 80,000-camera load test.

Official solution: **Customised Model 5 Hybrid Architecture**

- **Model 1** is always on: vendor-neutral camera registry and GIS.
- **Model 2** is the first feed-to-alert path: unified viewing of accessible feeds plus ANPR metadata, without replacing departmental recording.
- Regional/edge and remote GPU processing are **deployment strategies inside this hybrid**, not a new official model.
- **Do not build a Central VMS.** Full video stays in departmental stores. The centre stores plate crops, metadata, and authorised incident references.

Working story:

> Onboard heterogeneous feeds into a vendor-neutral registry, persist reviewable ANPR sightings, exact-match an authorised watchlist, and show timestamped evidence plus **inferred** GIS movement, while departmental recording stays local.

## Architecture

A camera uses one processing mode:

| Mode | Meaning |
|---|---|
| `vendor_metadata` | Consume authorised ANPR/event metadata from the camera or VMS |
| `local_worker` | OpenCV + Tesseract on this departmental/local host |
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

## Run (Windows)

```powershell
python -m pip install -r requirements.txt
python scripts\check_host.py
python scripts\seed.py
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000

Default operator token: `p0-operator` (override with `ADMIN_TOKEN`). Vendor ingest token: `p0-vendor` (`VENDOR_INGEST_TOKEN`).

1. **Run own-feed analysis** — OpenCV + Tesseract on generated Ahmedabad and Surat frames.
2. Search `GJ01AB1234` — sightings, evidence crops, dashed inferred path.
3. Review alerts (ack / confirm / reject). Coverage stays honest: only feeds that are actually being processed are `analytics_active`.

Own-feed verification:

```powershell
python scripts\verify_own_feed.py
python -m pytest -q
```

## Local Tesseract/OpenCV provider

Day-1 path on this host:

- OpenCV colour/contour plate localisation
- Tesseract 5 at `C:\Program Files\Tesseract-OCR\tesseract.exe`
- A–Z / 0–9 whitelist
- Indian plate normalisation + GJ / Bharat-series syntax flag
- Independent multi-frame character-consistency vote (not a patent implementation)
- Exact normalised watchlist match
- Evidence crop only
- `model_id=tesseract-opencv-p0` plus a hash of the local provider file

Awiros, PP-OCRv5 and YOLO are **candidates only**. They are not written into records unless that provider is actually installed and used.

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

Production never invents a remote plate. Tests mock the HTTP endpoint. On failure the error is recorded; local Tesseract is used only when fallback is allowed.

## Vendor metadata path

`POST /api/vendor/events` with `Authorization: Bearer <VENDOR_INGEST_TOKEN>`.

Required: `event_id`, `camera_id`, `source_time`, `plate_raw`, `confidence`, `vendor_model_id`. Payload size is limited. The service persists a **Sighting** first, then exact-matches the watchlist, then opens an alert from that row. Replayed `event_id` values are rejected. The original payload is stored as a hash for audit.

## Workers

```
POST /api/workers/{camera_id}/start
POST /api/workers/{camera_id}/stop
POST /api/workers/stop-all
GET  /api/workers
```

Bounded in-process threads. No Kafka, Kubernetes, or distributed scheduler. Concurrent local/remote workers and open captures are capped (`MAX_CONCURRENT_WORKERS`, `MAX_OPEN_CAPTURES`). Overflow cameras are **queued** and remain `analytics_active=false`. Duplicate workers for the same camera are refused. `analytics_active` is true only while frames are actually being processed.

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

## Honest government-feed blocker

Government-feed status is whatever this host actually observes after `POST /api/catalogue/sync` and a bounded RTSP probe. Do not claim 50 live government cameras. Do not treat catalogue `live=true` as `analytics_active`. Protected HLS credentials stay server-side; browser HLS preview is blocked unless a safe URL is available. WHEP may be used for on-demand preview. Analytics uses RTSP-over-TCP.

```powershell
python scripts\check_host.py
python scripts\probe_government_feed.py
```

`probe_government_feed.py` syncs the catalogue, tests TCP/HTTPS, opens **one** RTSP camera, and runs a bounded ANPR sample. It does not open every catalogue stream.

## Cost / capacity estimator

The UI posts user-supplied assumptions (camera count, bitrate, target FPS, active cameras, measured worker FPS, GPU hourly cost, storage cost, evidence volume). Results are labelled **estimate only**. No savings percentage is hard-coded.

## Known host limitations

- Windows, Python 3.14, GTX 1650 — throughput is a measured hypothesis, not a statewide rating
- Tesseract OCR quality is limited; this is the honest Day-1 provider
- No Node.js and no external FFmpeg executable on PATH at plan time
- OpenCV may still use its internal FFmpeg backend for RTSP
- Government catalogue credentials and a real remote GPU endpoint are external blockers
- District GPU infrastructure is **not** deployed by this P0
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
