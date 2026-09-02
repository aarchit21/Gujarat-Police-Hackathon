# Phase 2 — Literature, patents, datasets, tools

Gujarat Police Hackathon CCTV hybrid. Research only; nothing here is a measured result on portal feeds.

**Execution lock:** four-person P0, four days. The 14-day plan in `Gujarat_CCTV_Hybrid_Team_Plan.pptx` (slide 22) is useful evidence that four days is a substantial compression. Compress that deck’s Day 1–3 critical path (slide 26) into the first two days.

**Phase 1 locks:** live ~50-feed demo · vehicle ANPR + watchlist only · two-stage detector + specialised OCR · observed GIS sightings + inferred path · exact match + separate fuzzy review queue (fuzzy is droppable).

**Submission story (P0):** onboard heterogeneous government feeds into a vendor-neutral registry, continuously generate reviewable ANPR sightings, correlate them with an authorised watchlist, and show timestamped evidence and observed movement on GIS, while retaining existing departmental recording infrastructure.

The decisive Day-1 question is **feed access**, not model selection.

---

## Recommended prototype stack

```
RTSP / ONVIF / file
  → MediaMTX (one pull; WebRTC view + analytics tap)
  → FFmpeg decode at calibrated analysis FPS  (2 FPS is a hypothesis, not a lock)
  → ONNX plate detector (candidate: FastALPR YOLOv9-t or exported YOLO)
  → SORT default / ByteTrack if in-camera grouping needs it
  → candidate Indian OCR on crops (Awiros PP-OCRv5 fine-tune is first benchmark, not locked)
  → independently designed multi-frame consistency vote; keep raw OCR
  → exact watchlist match | fuzzy → review queue only (drop fuzzy if midday Day 2 missed)
  → PostGIS sighting + inferred GIS path
```

Do **not** ship `ultralytics` as a networked service (AGPL-3.0 is a **verification task**, not a settled licence fact until re-checked). Prefer MIT ONNX + Apache OCR **after** licence cards are re-verified. Vanilla PP-OCRv5 is **not** an Indian two-row OCR.

Avoid Kafka / Kubernetes / a full central VMS during the prototype.

---

## Hard 4-day checkpoints

| When | If this fails | Action |
|---|---|---|
| Midday Day 1 | No government feed decoded on the actual host | Escalate access immediately. Mandatory government-feed demo is blocked until it works. |
| End of Day 1 | No readable persistent sighting | Stop UI and infrastructure work. |
| Midday Day 2 | No real end-to-end alert | Remove fuzzy matching, advanced tracking, relay refinements, and bonus features. |
| Midday Day 3 | No stable required concurrency | Reduce inference sampling from measured plate dwell time; disclose actual analytics coverage. |
| Day 4 midday | — | Feature freeze. No new features after this. |

---

## 50-camera scope

“Two source types” is useful evidence of heterogeneity. It does **not** replace onboarding approximately 50 supplied cameras.

- Every accessible camera must appear in the ledger (connected, blocked, or deferred — with the reason).
- Report `analytics_active_count` versus onboarded count. Do not mark a camera healthy if analytics stopped.
- Hardware benchmarking of the actual host remains a critical unknown.

---

## 1. Papers — disposition

Local PDFs in `papers/`. Use the **Disposition** column; do not upgrade a paper beyond it.

### Detection / OCR / Indian plates

| Paper | Local file | Disposition |
|---|---|---|
| Laroca et al., YOLO ALPR, IJCNN 2018 | `2018_Laroca_YOLO_ALPR_IJCNN.pdf` | Strongly supports **video-level** evaluation and temporal redundancy. Brazilian results are **not** Gujarat accuracy. |
| Silva & Jung, WPOD-NET, ECCV 2018 | `2018_Silva_Jung_WPOD-NET_ECCV.pdf` | Supports perspective rectification. Reported complete pipeline was comparatively slow. Add **only** when skew is a measured failure. |
| Zherzdev & Gruzdev, LPRNet | `2018_Zherzdev_LPRNet.pdf` | Lightweight but Chinese-focused. Tanwar shows the Indian two-row limitation. Not the OCR path. |
| Laroca et al., layout-independent YOLO ALPR | `2019_Laroca_LayoutIndependent_YOLO_ALPR.pdf` | Supports explicit one-row / two-row classification and layout-aware validation. Copy the method, not Brazilian syntax. |
| Tanwar et al., Indian-LPR in the wild, 2021 | `2021_Tanwar_Indian_LPR_in_the_wild.pdf` | Strongest local open Indian detector-dataset evidence. Four-point warping improved character accuracy from 66.3% to 75.6%. **75% remains inadequate for deployment.** |
| Indian truck plates, YOLOv7 + PARSeq, 2022 | `2022_Indian_Truck_Plates_YOLOv7_PARSeq.pdf` | Relevant for difficult commercial / two-line plates. Validation set is only 239 images. **Retain as a fallback, do not dismiss completely.** |
| Nadiminti et al., Indian E2E ANPR, 2022 | `2022_Nadiminti_Indian_E2E_ANPR.pdf` | Supports COCO-pretrained detector adaptation. Do **not** transfer a Chinese CCPD model. |
| Laroca et al., AOLP/CCPD near-duplicates | `2023_Laroca_AOLP_CCPD_NearDuplicates.pdf` | Requires video-level or vehicle-level splits. Adjacent frames must **never** be divided between training and testing. |
| Sensors 2024, real-time LPR unconstrained | `2024_Sensors_Realtime_LPR_Unconstrained.pdf` | Strong Chinese CCPD results; explicitly acknowledges restricted plate coverage. **Not Indian validation.** |
| Awiros ANPR-OCR technical report | `2025_Awiros_ANPR_OCR_TechnicalReport.pdf` | **Candidate, not a locked recogniser.** Reports 98.42% overall and 96.91% two-row at ~5 ms/crop on an RTX 3090, on pre-cropped plates. Does not state the held-out validation-set size for those scores, does not provide an external Gujarat-feed test, and evaluates OCR rather than the full detector-to-alert pipeline. Promising first benchmark; **not** expected portal-feed accuracy. |
| Cui et al., PaddleOCR 3.0 | `2025_Cui_PaddleOCR_3.0.pdf` | Suitable runtime / toolkit. The general pretrained model is **not** by itself an Indian ANPR solution. |
| IJARSCT 2025, YOLOv8 + PaddleOCR (Indian) | `2025_IJARSCT_YOLOv8_PaddleOCR_Indian.pdf` | Supports the broad YOLO → PaddleOCR pipeline. Insufficient quantitative evidence for model selection. |
| Islas-Yañez et al., modular YOLO + PaddleOCR, *Sensors* **2026**, 26, 2785 | `2025_PMC_YOLO_PaddleOCR_Modular.pdf` | Filename says 2025; the article is *Sensors* **2026**. Closest published stack shape. Copy pipeline order only. |
| Korean YOLOv12 + OCR comparison, *Sensors* 2026 | `2026_Sensors_YOLOv12_OCR_Korean_plates.pdf` | Useful mainly as a warning: tracking errors and integrated-pipeline overhead dominate isolated OCR benchmarks. Wrong plates for Gujarat training. |

### Tracking (in-camera only)

| Paper | Local file | Disposition |
|---|---|---|
| Bewley et al., SORT, 2016 | `2016_Bewley_SORT.pdf` | Lightweight fallback for **per-camera passage grouping**. Detection quality dominates tracking quality. Default tracker for P0. |
| Zhang et al., ByteTrack, ECCV 2022 | `2021_Zhang_ByteTrack.pdf` | Useful inside one camera for grouping detections and selecting the best crop. **Not needed for cross-camera identity.** Drop if midday Day 2 is missed. |

### GIS / multi-camera (mostly out of P0)

| Paper | Local file | Disposition |
|---|---|---|
| Xie et al., multi-camera travel-time, 2022 | `2022_Xie_MultiCamera_TravelTime.pdf` | Visual-ReID travel-time system. **Future-roadmap material**, not necessary for registration-number tracking. |
| SAE EASE-MCVT, 2025 | `2025_SAE_MCVT_Edge_MultiCamera.pdf` | Supports regional edge processing and central metadata. Visual ReID and its engineering framework are **outside the four-day scope**. |

### Matching / alerts (practice)

| Source | Local file | Disposition |
|---|---|---|
| CRS R48160, ALPR + hot lists | `2024_CRS_R48160_ALPR_Hotlists.pdf` | Supports hotlist comparison, real-time alerts, privacy controls, and human review. Operational analogue for the live test. |

### Removed from the CCTV evidence pack

| File | Why |
|---|---|
| `2023_CityScale_Vehicle_Trajectories_SciData.pdf` | **Incorrect local paper.** It is a meteorological **dropsonde** dataset (*Scientific Data* 2023, ACTIVATE / western North Atlantic). It is **not** city-scale vehicle-trajectory research. Do not cite it as CCTV evidence. Moved to `papers/_excluded/`. |

---

## 2. Datasets

| Dataset | Link | Size | Role |
|---|---|---|---|
| Indian-LPR (Tanwar 2021) | [arXiv](https://arxiv.org/abs/2111.06054) · [GitHub](https://github.com/sanchit2843/Indian_LPR) | 16,192 img / 21,683 plates, 4-point | Best open Indian in-the-wild detector set. 75.6% char accuracy is still not deployable. |
| Awiros Indian OCR corpus | weights: [HF](https://huggingface.co/Awiros/anpr-ocr) | 558k (mostly private) | Use **weights as a candidate**, not their images. Licence on the model card is a **verification task**. |
| Datacluster Indian plates | [GitHub](https://github.com/datacluster-labs/Indian-Licence-Plate-Image-Dataset) | ~6,000+, 20+ states | Extra detector diversity. Confirm licence (verification task). |
| Kaggle Indian plate boxes | [Kaggle](https://www.kaggle.com/datasets/sunrajbishnolia/license-plate-detection) | 15,338 YOLO | Detector only. |
| Kaggle Gujarat plates | [Kaggle](https://www.kaggle.com/datasets/paneraghanshyam/gujarat-vehicle-number-plates-yolo-ready) | 355 | GJ smoke test only. Too small to train. |
| Indian truck plates | [arXiv:2211.13194](https://arxiv.org/pdf/2211.13194) | commercial/two-line; val n=239 | Optional hard-set fallback. |
| UFPR-ALPR | [UFPR](https://web.inf.ufpr.br/vri/databases/ufpr-alpr/) | 4,500 video frames | **Protocol** (passages), not Indian fonts. |
| CCPD (China) | [GitHub](https://github.com/detectRecog/CCPD) | ~250k–355k | Method benchmark only. Wrong script. |
| AOLP (Taiwan) | Hsu 2012 | 2,049 | Near-duplicates; not Indian. |

**Watchlist for the demo:** synthetic plates that also appear in authorised own-feed video. Do not harvest real private GJ plates as “stolen.”

---

## 3. Indian plate syntax (plausibility, not identity)

Source: [Vehicle registration plates of India](https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_India) · MoRTH BH series 2021. Treat the public syntax tables as **working hypotheses** until the cited instruments are re-verified.

| Family | Pattern | Example |
|---|---|---|
| Standard (MV Act 1988) | `SS DD XX NNNN` (I/O unused in series) | `GJ 01 AB 1234` |
| Some states omit leading 0 | `SS D XX NNNN` | `GJ 1 AB 1234` |
| Bharat series | `YY BH #### XX` (I/O unused) | `26BH4567AB` |
| Gujarat government | `GJ 18 G ####` / `GJ 18 GJ ####` | not a private series |
| Two-wheeler two-line | state+RTO / series+number | main OCR failure mode |

Invalid syntax → keep raw read, no auto-alert. BH must be in the grammar.

---

## 4. Patents (cite in HLD; do **not** copy claims)

**Do not make a patent the implementation specification.** Multi-frame OCR consensus is useful, but implement a **simple independently designed** voting / consistency method. Obtain IP review before making patent-derived functionality a production commitment.

| Patent | Link | Our posture |
|---|---|---|
| US20230394850A1 voting multi-frame LPR | [Google Patents](https://patents.google.com/patent/US20230394850A1/en) | Cite as related art. Implement an independent character-consistency vote. IP review before any production commitment. Patent **status** is a verification task. |
| US20240355130A1 list match with char-overlap | [Google Patents](https://patents.google.com/patent/US20240355130A1/en) | Fuzzy **review queue** only. Droppable at midday Day 2. |
| US10839303B2 auto-correct LPR via neighbours (SAP) | [Google Patents](https://patents.google.com/patent/US10839303B2/en) | Cite; **do not** silently rewrite OCR. Flag implausible skips. |
| US20200074211A1 ALPR + hotlist real-time alert | [Google Patents](https://patents.google.com/patent/US20200074211A1/en) | Same shape as the live test. Cite, do not copy claims. |
| US 7,711,150 B2 autonomous LPR + DB alert | [PDF](https://patentimages.storage.googleapis.com/df/09/41/f0eb9090834c08/US7711150.pdf) | Classic patrol-ALPR. Cite only. |
| US20240265704A1 VMS + GIS FOV | [Google Patents](https://patents.google.com/patent/US20240265704A1/en) | Model 1 coverage / gap analysis citation. |
| CN119540847A GIS urban video monitor | [Google Patents](https://patents.google.com/patent/CN119540847A/en) | Pixel→geo; we only plot camera points + inferred links. |
| KR101451115B1 multi-protocol ONVIF VMS | [Google Patents](https://patents.google.com/patent/KR101451115B1/en) | Adapter pattern citation. |

ONVIF versions and “no open VMS↔VMS federation profile” are **verification tasks**. Profile V (cloud VMS draft) is future, not a 50-feed tool — re-check the ONVIF site before asserting current profile status.

---

## 5. Tools

### Streaming / ONVIF

ONVIF Profile T / G / M availability on supplied cameras is a **verification task**, not a settled fact from the local paper collection.

| Tool | Link | Licence / note |
|---|---|---|
| ONVIF Profile T | [spec](https://www.onvif.org/profiles/profile-t/) | Verify whether supplied cameras implement it. |
| ONVIF Profile G | [spec](https://www.onvif.org/profiles/profile-g/) | Recording / playback **if** the NVR exposes it. |
| ONVIF Profile M | [spec](https://www.onvif.org/profiles/profile-m/) | Analytics metadata **if** the camera already has ANPR. |
| MediaMTX | [GitHub](https://github.com/bluenviron/mediamtx) | MIT (re-verify). One pull → WebRTC/HLS/RTSP. Drop relay refinements if midday Day 2 is missed. |
| gortsplib | [GitHub](https://github.com/bluenviron/gortsplib) | RTSP client. |
| python-onvif-zeep / onvif2_zeep | GitHub | GetProfiles / GetStreamUri; Media2 for HEVC. |
| FFmpeg | [ffmpeg.org](https://ffmpeg.org/) | Decode. Lowering inference FPS does not cut WAN if the source stream still moves. |

### Detector / OCR

| Tool | Link | Licence (verify) | Verdict |
|---|---|---|---|
| PaddleOCR / PP-OCRv5 | [GitHub](https://github.com/PaddlePaddle/PaddleOCR) | Apache 2.0 — **re-verify** | Runtime/toolkit. Crop-only rec; skip a second text detector. Not by itself an Indian ANPR solution. |
| Awiros-ANPR-OCR (Indian PP-OCRv5) | [Hugging Face](https://huggingface.co/Awiros/anpr-ocr) | Card says Apache 2.0 — **verification task** | **Candidate recogniser.** Pin hash if used. Do not treat 98.42% as portal-feed accuracy. |
| FastALPR | [GitHub](https://github.com/ankandrew/fast-alpr) | MIT — **re-verify** | Default YOLOv9-t 384 ONNX detector candidate. Swap OCR independently. |
| Ultralytics YOLO | [PyPI](https://pypi.org/project/ultralytics/) | **AGPL-3.0 — re-verify** | Local train/export only. Do not serve the package. |
| EasyOCR / Tesseract | — | Apache / Apache | Fallback only. |

### App / GIS / data

| Tool | Note |
|---|---|
| FastAPI + OpenAPI | Application API. |
| PostgreSQL + PostGIS | Registry, sightings, outbox. No Kafka on day one. |
| React + Leaflet | Map, alert queue, history. Stop UI work if no Day-1 sighting. |
| Docker Compose | Prototype deploy. Not Kubernetes. |
| ONNX Runtime | Detector inference without an AGPL runtime. |

---

## 6. Methodologies (what to implement in four days)

**ANPR worker**

1. Decode at a **calibrated** analysis FPS. **Do not hard-code 2 FPS.** Start from 2 FPS only as a hypothesis. Calibrate from vehicle speed, plate pixel width, dwell time, blur, and measured passage recall.
2. Detect plates (ONNX). Optional vehicle-first if plates are tiny.
3. Track with SORT (ByteTrack only if grouping quality needs it). Passage = track ID + short gap. Detection quality dominates tracking quality.
4. Score crop: det conf × sharpness × min plate width.
5. OCR top-K crops with the **candidate** recogniser. Keep raw strings.
6. Independently designed multi-frame character consistency / vote. **Not** a patent implementation. Keep raw OCR plus the voted plate. IP review before production commitment.
7. Normalise (strip space/hyphen, uppercase, NFKC). Syntax flag only.
8. Exact active watchlist → alert. Optional Levenshtein ≤ 1 or known confusions **and** syntax-ok → **review queue**, never rewrite OCR. Drop fuzzy if midday Day 2 is missed.
9. Persist every sighting (needed for designated-vehicle GIS search).

**Time:** store `source_time`, `ingest_time`, `clock_offset_ms`. Display IST. Flag large drift.

**Dedup key:** `(watchlist_id, camera_id, passage_id)`. New camera = new alert.

**GIS:** order by corrected UTC → camera points → dashed **inferred** links → flag implausible speed. Do not claim a verified road polyline.

**~50-feed:** GPU-batch OCR; drop analysis frames under backpressure; never mark a camera healthy if analytics stopped. UI tiles on demand (not 50 full-res WebRTC). Report actual analytics coverage.

**Eval:** label **passages** not frames; hold out by **video**; never split adjacent frames across train/test. Report alert precision, passage recall, and recall on independently readable plates. Output report: plate, camera, timestamp, crop, confidence, model hash, run ID.

---

## 7. Policy / database boundary (verification tasks)

These instruments are **asserted in earlier drafts**. They are **not established** by the local paper collection. Treat each as a verification task, not a settled fact.

| Instrument | Working implication (unverified) | Verify |
|---|---|---|
| DPDP Act 2023 + Rules 2025 | AI CCTV may be personal data; LE room still requires minimise / retain / secure. | Gazette / MeitY text and legal review. |
| MoRTH data-sharing / NAPIX | VAHAN/SARATHI via NAPIX for police if authorised. No public API. | Current MoRTH circular and whether access is granted. |
| NAFIS | Fingerprints. Out of ANPR path. | PIB / official description. |
| CCTNS / ICJS / eGujCop | Case/criminal records, not plate OCR. | Official scope. |
| Bharatiya Sakshya / eSakshya | Hash ≠ legal certificate. | Legal review. |

Prototype retention (engineering, not legal advice): short rolling buffer of all reads; persist watchlist hits, confirmed events, audit.

---

## 8. External claims that need re-verification

Do not treat the following as settled because they appeared in an earlier literature review:

- ONVIF profile versions actually implemented by supplied cameras.
- Model licences (PaddleOCR, FastALPR, Ultralytics AGPL, MediaMTX).
- Hugging Face model-card licences (Awiros and any other weights).
- MoRTH / NAPIX access for this team.
- DPDP interpretation for this prototype.
- Patent legal status (filed / granted / expired / applicable jurisdiction).

---

## 9. Do not cite as Gujarat accuracy

- Awiros 98.42% / 96.91% as expected portal-feed accuracy (OCR on pre-cropped plates, no Gujarat-feed test, full pipeline not measured).
- CCPD / AOLP headline recognition rates (duplicates + wrong fonts).
- Ultralytics mAP on contaminated Roboflow splits.
- Vanilla PP-OCRv5 as Indian two-row OCR (0.24% on Awiros two-row split).
- Tanwar 75.6% as a deployment-ready number.
- Sensors 2024 CCPD results as Indian validation.
- Sentinel-Hybrid — concurrent mapping of the same brief. Competition, not a kit.
- `2023_CityScale_Vehicle_Trajectories_SciData.pdf` as city-scale vehicle research.

---

## 10. HLD citation pack (P0)

ALPR: Laroca IJCNN 2018 / layout-independent; Tanwar 2021; Nadiminti 2022; Cui et al. PaddleOCR 3.0; Awiros report as **candidate OCR benchmark**; *Sensors* 2026 modular YOLO+PaddleOCR (pipeline shape only).

Tracking: SORT 2016 (default); ByteTrack only if needed in-camera.

Matching: CRS R48160; patents **cite only**.

GIS: observed sightings + inferred links. Xie 2022 and EASE-MCVT are roadmap, not P0.

India: MV Act plate format (re-verify); MoRTH BH / NAPIX and DPDP as verification tasks.

Do **not** cite the ACTIVATE dropsonde *Sci Data* 2023 paper.
