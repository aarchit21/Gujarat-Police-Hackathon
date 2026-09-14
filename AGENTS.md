# Solo 4-day P0 — instructions for Grok / Codex

This repository is a **solo** Gujarat Police Hackathon build. **You generate all application code.** The human runs the host, supplies government-feed access, and records demos.

Read `docs/FOUR_DAY_P0_PLAN.md` before editing.

## Locks

- Phased hybrid: Model 1 registry+GIS always on; Model 2 as the first feed-to-alert path. Do not build a central VMS.
- Vehicle ANPR + watchlist only. No faces, no person ReID, no FRS.
- Exact normalised match auto-alerts. Fuzzy is review-queue only and droppable.
- GIS links between cameras are **inferred**, never proven road polylines.
- Departmental recording stays local. We store crops + metadata, not statewide video.
- No Kafka, Kubernetes, Elasticsearch, or live VAHAN/NAPIX/NAFIS.
- Awiros is a **candidate** OCR, not a lock. Tesseract/OpenCV is the Day-1 path on this host.
- Do **not** hard-code 2 FPS. Sampling is a calibrated hypothesis.
- Do **not** claim 50 cameras healthy unless analytics is actually running.
- Do **not** copy patent claims. Independent character-consistency vote only.
- Mock UIs are disallowed. Every alert must come from a persisted sighting row.
- Vehicle type and colour come **only** from `app/services/vehicle_attributes.py` (OpenVINO
  OMZ `vehicle-attributes-recognition-barrier-0042`, Apache-2.0). Never an LLM or VLM — not
  for type, not for colour, not for plate text. Plate text comes only from `cpu_anpr.py`.
- Supported classes are `car/van/truck/bus` (+ `two_wheeler` from COCO id 3) and
  `white/gray/yellow/red/green/blue/black`. `suv`, `auto_rickshaw`, `taxi_cab`, `silver`,
  `brown` and `orange` have **no supporting weight** and must stay `unknown`. Do not map
  `gray`→`silver` or promote `car`→`suv`.
- Detection/type/colour must keep working when ANPR is off or fails. A plate failure may
  never discard a vehicle observation or alter its attributes.
- Prefer abstention. `unknown` is a decision the gate made, not a missing value, and an
  abstained record must report confidence `0.0`.
- No accuracy has been measured on Indian CCTV. Do not quote the vendor's barrier-dataset
  figures as if they described this deployment.

## Day-1 question

If authorised government feeds cannot be decoded on this host, the mandatory government-feed demo is blocked. Keep own-feed working. Put every camera in the ledger with an honest status.

## Host

Two hosts are valid:

- Original laptop: Windows, Python 3.14, GTX 1650, Tesseract at `C:\Program Files\Tesseract-OCR\tesseract.exe`.
- This GPU server: Linux, conda env `gujhac`, Python 3.11, RTX A4500 (20 GB). Tesseract from conda-forge. Local Ollama `qwen3-vl:8b` on `http://127.0.0.1:11434`.

Do not require an FFmpeg or Node **executable** for the P0. OpenCV may still use its bundled FFmpeg backend.

## Layout

```
app/           FastAPI backend + static UI
scripts/       seed, own-feed generator, host check
data/          sqlite, cameras, watchlist, frames, evidence (generated)
docs/          brief, plan, literature, deck
papers/        evidence pack (dropsonde SciData is in papers/_excluded/)
```

## Gates

Stop UI/infra if no persistent sighting by end of Day 1. Strip fuzzy/tracking/bonus if no end-to-end alert by midday Day 2. Feature freeze Day 4 midday.
