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

## Day-1 question

If authorised government feeds cannot be decoded on this host, the mandatory government-feed demo is blocked. Keep own-feed working. Put every camera in the ledger with an honest status.

## Host

Windows, Python 3.14, GTX 1650, Tesseract at `C:\Program Files\Tesseract-OCR\tesseract.exe`. Do not require FFmpeg or Node for the P0.

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
