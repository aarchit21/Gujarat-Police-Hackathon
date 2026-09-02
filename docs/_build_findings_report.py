"""Generate the Gujarat CCTV research findings PDF."""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT = Path(__file__).resolve().parent / "Gujarat_CCTV_Research_Findings.pdf"

NAVY = colors.HexColor("#1B365D")
NAVY2 = colors.HexColor("#2C5282")
GOLD = colors.HexColor("#C4A35A")
INK = colors.HexColor("#1A1A1A")
MUTED = colors.HexColor("#4A5568")
RULE = colors.HexColor("#D6D3CD")
ROW = colors.HexColor("#F4F1EA")
HEAD_BG = colors.HexColor("#1B365D")
WHITE = colors.white

W, H = A4
LM = 18 * mm
RM = 18 * mm
TM = 22 * mm
BM = 18 * mm
CONTENT_W = W - LM - RM


def styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle(
        "CoverKicker", fontName="Times-Bold", fontSize=10, textColor=GOLD,
        tracking=1.2, alignment=TA_LEFT, spaceAfter=8,
    ))
    s.add(ParagraphStyle(
        "CoverTitle", fontName="Times-Bold", fontSize=26, leading=30,
        textColor=WHITE, alignment=TA_LEFT, spaceAfter=10,
    ))
    s.add(ParagraphStyle(
        "CoverSub", fontName="Times-Italic", fontSize=13, leading=17,
        textColor=colors.HexColor("#E8E4D9"), alignment=TA_LEFT, spaceAfter=6,
    ))
    s.add(ParagraphStyle(
        "CoverMeta", fontName="Helvetica", fontSize=9, leading=13,
        textColor=colors.HexColor("#D0D7E2"), alignment=TA_LEFT,
    ))
    s.add(ParagraphStyle(
        "H1", fontName="Times-Bold", fontSize=14, leading=18,
        textColor=NAVY, spaceBefore=14, spaceAfter=8,
        borderPadding=0,
    ))
    s.add(ParagraphStyle(
        "H2", fontName="Times-Bold", fontSize=11.5, leading=15,
        textColor=NAVY2, spaceBefore=10, spaceAfter=5,
    ))
    s.add(ParagraphStyle(
        "Body", fontName="Times-Roman", fontSize=10, leading=13.5,
        textColor=INK, alignment=TA_JUSTIFY, spaceAfter=7,
    ))
    s.add(ParagraphStyle(
        "BulletBody", fontName="Times-Roman", fontSize=10, leading=13.2,
        textColor=INK, leftIndent=12, spaceAfter=3,
    ))
    s.add(ParagraphStyle(
        "Cell", fontName="Helvetica", fontSize=7.6, leading=10.2,
        textColor=INK, alignment=TA_LEFT,
    ))
    s.add(ParagraphStyle(
        "CellHead", fontName="Helvetica-Bold", fontSize=7.6, leading=10.2,
        textColor=WHITE, alignment=TA_LEFT,
    ))
    s.add(ParagraphStyle(
        "Caption", fontName="Times-Italic", fontSize=8.5, leading=11,
        textColor=MUTED, spaceBefore=2, spaceAfter=8,
    ))
    s.add(ParagraphStyle(
        "Callout", fontName="Times-Roman", fontSize=10, leading=13.5,
        textColor=INK, leftIndent=8, rightIndent=8, spaceBefore=4, spaceAfter=8,
    ))
    s.add(ParagraphStyle(
        "Footer", fontName="Helvetica", fontSize=8, textColor=WHITE,
    ))
    s.add(ParagraphStyle(
        "TOC", fontName="Times-Roman", fontSize=10.5, leading=16,
        textColor=INK, leftIndent=4,
    ))
    return s


S = styles()


def p(text, style="Body"):
    return Paragraph(text, S[style])


def cell(text, head=False):
    return Paragraph(text, S["CellHead"] if head else S["Cell"])


def table(headers, rows, col_widths):
    data = [[cell(h, head=True) for h in headers]]
    for row in rows:
        data.append([cell(c) for c in row])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.3, RULE),
        ("BACKGROUND", (0, 1), (-1, -1), WHITE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, ROW]),
    ]))
    return t


def hrule():
    t = Table([[""]], colWidths=[CONTENT_W], rowHeights=[2])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), GOLD),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return t


def callout(text):
    inner = Paragraph(text, S["Callout"])
    t = Table([[inner]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7F3E8")),
        ("BOX", (0, 0), (-1, -1), 1.2, GOLD),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def cover_page(c, doc):
    c.saveState()
    c.setFillColor(NAVY)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(GOLD)
    c.rect(0, H - 8 * mm, W, 8 * mm, fill=1, stroke=0)
    c.rect(0, 0, W, 8 * mm, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Times-Bold", 11)
    c.drawString(LM, H - 28 * mm, "GUJARAT POLICE HACKATHON")
    c.setStrokeColor(GOLD)
    c.setLineWidth(1.2)
    c.line(LM, H - 31 * mm, LM + 55 * mm, H - 31 * mm)
    c.restoreState()


def later_page(c, doc):
    c.saveState()
    c.setFillColor(NAVY)
    c.rect(0, H - 12 * mm, W, 12 * mm, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Helvetica", 8)
    c.drawString(LM, H - 8 * mm, "Gujarat Police Hackathon  ·  CCTV Hybrid Research Report")
    c.drawRightString(W - RM, H - 8 * mm, "Corrected P0 findings")
    c.setFillColor(NAVY)
    c.rect(0, 0, W, 12 * mm, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Helvetica", 8)
    c.drawString(LM, 5 * mm, "1 September 2026  ·  Research, not a measured prototype")
    c.drawRightString(W - RM, 5 * mm, f"Page {doc.page}")
    c.setFillColor(GOLD)
    c.rect(0, 12 * mm, W, 1.2, fill=1, stroke=0)
    c.restoreState()


def build():
    story = []

    story.append(Spacer(1, 48 * mm))
    story.append(p("TECHNICAL RESEARCH REPORT  ·  CORRECTED P0", "CoverKicker"))
    story.append(p("Four days.<br/>Evidence-led alerts.", "CoverTitle"))
    story.append(p(
        "A tightly scoped four-person build: vendor-neutral registry, reviewable ANPR "
        "sightings, authorised watchlist alerts, and inferred GIS movement — with "
        "departmental recording left in place.",
        "CoverSub",
    ))
    story.append(Spacer(1, 14 * mm))
    story.append(p(
        "Sources: official brief, 31-slide team plan (31 August 2026), local papers folder "
        "after removing a misfiled dropsonde paper, and the four-day P0 correction.",
        "CoverMeta",
    ))
    story.append(p(
        "Date: 1 September 2026  ·  Status: research findings, not a working prototype claim.",
        "CoverMeta",
    ))
    story.append(p(
        "Execution: four people, four days. Decisive Day-1 question: government-feed access, "
        "not model selection.",
        "CoverMeta",
    ))
    story.append(PageBreak())

    story.append(p("Contents", "H1"))
    story.append(hrule())
    for line in [
        "1. Executive summary and corrections",
        "2. Problem statement and evaluation test",
        "3. Architecture decision and scope locks",
        "4. Four-day P0 plan and hard checkpoints",
        "5. Prototype pipeline to build",
        "6. Paper-by-paper disposition",
        "7. Datasets, syntax, tools",
        "8. Matching, GIS, ingest, and evidence",
        "9. Patents, policy, and verification tasks",
        "10. Remaining gaps and the Day-1 question",
        "11. Source index",
    ]:
        story.append(p(line, "TOC"))
    story.append(PageBreak())

    # 1
    story.append(p("1. Executive summary and corrections", "H1"))
    story.append(hrule())
    story.append(p(
        "Twenty-six Gujarat departments run independent CCTV estates, scattered up to about "
        "1,000 km, with different vendors, VMS platforms, storage, and retention. The scoring "
        "object is narrower: onboard about 50 heterogeneous government feeds, recognise a "
        "designated registration number, reconstruct timestamped GIS movement, and raise "
        "real-time watchlist alerts from a working backend."
    ))
    story.append(p(
        "This report remains research rather than a measured prototype. The architecture "
        "recommendation is unchanged in kind: a phased hybrid; departmental video retained "
        "locally; vehicle ANPR as the primary scoring path; raw OCR plus evidence retention; "
        "exact matching and human review; GIS sightings with inferred — not proven — links; "
        "and no Kafka, Kubernetes, or full central VMS during the prototype. Government-feed "
        "access and hardware benchmarking remain the critical unknowns."
    ))
    story.append(callout(
        "<b>Proceed only as a tightly scoped, four-person P0 build.</b> The strongest "
        "submission story is not “we built a statewide VMS.” It is: we onboard heterogeneous "
        "government feeds into a vendor-neutral registry, continuously generate reviewable "
        "ANPR sightings, correlate them with an authorised watchlist, and show timestamped "
        "evidence and observed movement on GIS, while retaining existing departmental "
        "recording infrastructure."
    ))
    story.append(p("1.1 Corrections to the previous draft", "H2"))
    story.append(table(
        ["Issue", "Correction"],
        [
            ["Schedule",
             "The 14-day / four-person plan in the team deck (slide 22) is not usable for this deadline. Four days needs a much smaller P0 and parallel ownership. Compress the deck’s Day 1–3 critical path (slide 26) into the first two days."],
            ["Fifty-camera scope",
             "“Two source types” is useful heterogeneity evidence. It does not replace onboarding approximately 50 supplied cameras. Every accessible camera must appear in the ledger. Actual analytics coverage must be reported."],
            ["Awiros recogniser",
             "Candidate, not locked. 98.42% overall / 96.91% two-row at ~5 ms/crop on an RTX 3090 is OCR on pre-cropped plates. The report does not state the held-out validation-set size for those scores, provides no external Gujarat-feed test, and does not measure the full detector-to-alert pipeline. Not expected portal-feed accuracy."],
            ["Misfiled paper",
             "2023_CityScale_Vehicle_Trajectories_SciData.pdf is a meteorological dropsonde dataset (ACTIVATE). Removed from the CCTV evidence pack. Do not cite it as city-scale vehicle research."],
            ["Publication year",
             "The modular YOLO / PaddleOCR file is a Sensors 2026 article (26, 2785), despite the 2025_ filename and the earlier “Sensors 2025” label."],
            ["External claims",
             "ONVIF versions, model licences, Hugging Face model-card licences, MoRTH/NAPIX access, DPDP interpretation, and patent status are verification tasks, not settled facts from the local papers."],
            ["Patents",
             "Do not make a patent the implementation specification. Implement a simple independently designed voting / consistency method. Obtain IP review before any patent-derived production commitment."],
            ["2 FPS",
             "Starting hypothesis only. Calibrate sampling from vehicle speed, plate pixel width, dwell time, blur, and measured passage recall."],
        ],
        [38 * mm, CONTENT_W - 38 * mm],
    ))

    # 2
    story.append(p("2. Problem statement and evaluation test", "H1"))
    story.append(hrule())
    story.append(p("2.1 Official problem", "H2"))
    story.append(p(
        "Departments operate standalone camera ecosystems. Some store in the cloud, some "
        "locally; retention is 7 or 15+ days. The brief also names VAHAN, SARATHI, "
        "eGujCop/CCTNS, AFIS, and NAFIS. AFIS/NAFIS are fingerprint systems and are out of "
        "the ANPR path. Live national-database access is not a four-day deliverable."
    ))
    story.append(p("2.2 Live test (the scoring object)", "H2"))
    story.append(table(
        ["Requirement", "Meaning for the four-day build"],
        [
            ["Onboard ~50 heterogeneous cameras",
             "Every accessible camera in the ledger (connected, blocked, or deferred, with reason). Two source types prove heterogeneity; they do not replace the ~50-camera scope. Report analytics_active_count versus onboarded count."],
            ["Integrate live or recorded feeds",
             "Working backend on the actual host. Mock UIs are disallowed. Recorded input must be labelled."],
            ["Designated registration number",
             "Identify, timestamp, GIS history, searchable events from persisted sightings."],
            ["Watchlist cross-reference",
             "Representative (synthetic) list is allowed. Exact match auto-alerts. Fuzzy is review-queue only and droppable."],
            ["Own-feed demo 2–3 min",
             "Onboard, then analytics, match, and alert."],
            ["Government-feed demo",
             "Separate deliverable plus plate/timestamp output report. Blocked if Day-1 decode fails."],
            ["Scale narrative to ~80,000",
             "Roadmap only. Not a fake 80k-stream benchmark from one host."],
        ],
        [52 * mm, CONTENT_W - 52 * mm],
    ))
    story.append(p(
        "Official submission pack: solution PPT/PDF, high-level design, own-feed video, "
        "government-feed video and output report, optional hosted URL and repository.",
        "Caption",
    ))

    # 3
    story.append(p("3. Architecture decision and scope locks", "H1"))
    story.append(hrule())
    story.append(p("3.1 Phased hybrid (Model 5)", "H2"))
    story.append(table(
        ["Model", "Role in P0", "Trade-off"],
        [
            ["1 Registry + GIS", "Always included: inventory, ownership, health, gap analysis.",
             "Inventory alone cannot detect vehicles."],
            ["2 Direct integration", "First working feed-to-alert path (RTSP / ONVIF / vendor API).",
             "Needs network access; loads the source."],
            ["3 VMS federation", "Out of four-day scope unless an API is already in hand.",
             "Vendor licence dependence."],
            ["4 Central VMS", "Not built. Selective incident crops only.",
             "Blanket central recording is costly."],
            ["5 Hybrid", "Site-by-site path from actual capability.",
             "Needs consistent contracts and ownership."],
        ],
        [38 * mm, 72 * mm, CONTENT_W - 110 * mm],
    ))
    story.append(p(
        "Target topology: departmental recording stays local, then a regional gateway (adapters, "
        "relay, ANPR, durable event queue), then central services (registry/GIS, watchlist, matching, "
        "alerts, audit). For the hackathon these layers may be co-located on one host. Do not "
        "claim edge or WAN savings if all feeds are pulled centrally.",
        "Caption",
    ))
    story.append(p("3.2 Locks that survive the four-day cut", "H2"))
    story.append(table(
        ["Lock", "Decision", "Four-day effect"],
        [
            ["Performance", "Live ~50-feed demo",
             "Report actual analytics coverage. Do not fake healthy cameras."],
            ["Analytics domain", "Vehicle ANPR + watchlist only",
             "No face, no person ReID, no FRS."],
            ["ANPR algorithm", "Two-stage detector + specialised OCR",
             "Awiros is a candidate, not the locked recogniser."],
            ["“Complete route”", "Observed sightings + inferred GIS path",
             "Dashed links, not verified roads."],
            ["Watchlist match", "Exact normalised match + human review",
             "Fuzzy review queue is droppable at midday Day 2."],
        ],
        [38 * mm, 58 * mm, CONTENT_W - 96 * mm],
    ))
    story.append(p("3.3 Build / drop / defer", "H2"))
    story.append(table(
        ["Build in four days", "Drop if a gate fails", "Defer"],
        [
            ["Registry + GIS ledger of all accessible cameras",
             "Fuzzy matching",
             "Facial recognition"],
            ["ANPR sightings + exact watchlist + evidence",
             "ByteTrack / WPOD-NET / relay refinements",
             "Visual ReID / Xie travel-time / EASE-MCVT"],
            ["RBAC, audit, secrets from day one",
             "Bonus features after a missed gate",
             "Live VAHAN / NAPIX / NAFIS; Kafka; Kubernetes; full central VMS"],
        ],
        [CONTENT_W / 3.0] * 3,
    ))

    # 4
    story.append(p("4. Four-day P0 plan and hard checkpoints", "H1"))
    story.append(hrule())
    story.append(p(
        "The existing deck’s 14-day plan for four contributors (slide 22) is planning evidence, "
        "not the execution schedule. Four experienced people still cannot deliver that scope in "
        "four days. Parallel ownership is mandatory: A integration, B ANPR, C backend, D UI/GIS."
    ))
    story.append(table(
        ["Window", "Work (compressed from deck Days 1–3 into Days 1–2)", "Owner"],
        [
            ["Day 1 morning",
             "Decode an authorised government feed on the actual host. Start the camera ledger. Freeze the event schema. Stand up detector + candidate OCR.",
             "A, C, B"],
            ["Day 1 afternoon",
             "Persist a readable plate as a real sighting (raw OCR, crop, camera, timestamp). Minimal GIS point only if that sighting exists.",
             "B, C, D"],
            ["Day 2 morning",
             "Exact watchlist match becomes an alert with evidence and GIS on one record. Two source types if possible; continue the ~50-camera ledger.",
             "A–D"],
            ["Day 2 afternoon",
             "Expand onboarding. Calibrate sampling from dwell time and passage recall. Exact-match only unless the midday gate passed.",
             "A, B, C"],
            ["Day 3",
             "Required concurrency. Report analytics_active_count versus onboarded count.",
             "A, B"],
            ["Day 4 to midday",
             "Feature freeze. Record both demos. Output report. HLD and links. Claims must match demonstrated behaviour.",
             "D + all"],
        ],
        [32 * mm, 98 * mm, CONTENT_W - 130 * mm],
    ))
    story.append(p("4.1 Hard checkpoints", "H2"))
    story.append(table(
        ["When", "Fail condition", "Immediate action"],
        [
            ["Midday Day 1",
             "No government feed decoded on the actual host",
             "Escalate access immediately. The mandatory government-feed demonstration is blocked until it works."],
            ["End of Day 1",
             "No readable persistent sighting",
             "Stop UI and infrastructure work."],
            ["Midday Day 2",
             "No real end-to-end alert",
             "Remove fuzzy matching, advanced tracking, relay refinements, and bonus features."],
            ["Midday Day 3",
             "No stable required concurrency",
             "Reduce inference sampling from measured plate dwell time. Disclose actual analytics coverage."],
            ["Day 4 midday",
             "—",
             "No new features after this."],
        ],
        [32 * mm, 55 * mm, CONTENT_W - 87 * mm],
    ))
    story.append(callout(
        "<b>Day-1 question.</b> If authorised government feeds cannot be decoded on the actual "
        "host during Day 1, the mandatory government-feed demonstration is blocked regardless of "
        "how polished the remaining system becomes."
    ))

    # 5
    story.append(p("5. Prototype pipeline to build", "H1"))
    story.append(hrule())
    story.append(p(
        "Licence-clean, real-time chain after the paper filter. Video stays on the analytics "
        "path even if nobody has the dashboard open. The UI must not decode 50 full-resolution "
        "browser tiles."
    ))
    pipe_style = ParagraphStyle(
        "PipeLineBody", parent=S["Cell"], fontName="Helvetica", fontSize=8, leading=11.2,
    )
    pipe = Paragraph(
        "1. Ingest RTSP / ONVIF / file<br/>"
        "2. MediaMTX — one pull; WebRTC view plus analytics tap "
        "(drop relay refinements if midday Day 2 is missed)<br/>"
        "3. FFmpeg decode at a <b>calibrated</b> analysis FPS — 2 FPS is a hypothesis, not a lock<br/>"
        "4. ONNX plate detector (FastALPR YOLOv9-t or exported YOLO)<br/>"
        "5. SORT passage grouping; ByteTrack only if needed; sharpest crop; WPOD only if skew is measured<br/>"
        "6. Candidate Indian OCR on crops (Awiros fine-tune is the first benchmark, not locked)<br/>"
        "7. Independently designed multi-frame consistency vote; keep raw OCR<br/>"
        "8. Exact watchlist match becomes an alert; fuzzy goes to a labelled review queue and is droppable<br/>"
        "9. PostGIS sighting plus inferred GIS path",
        pipe_style,
    )
    pipe_tbl = Table([[pipe]], colWidths=[CONTENT_W])
    pipe_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), ROW),
        ("BOX", (0, 0), (-1, -1), 0.4, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(pipe_tbl)
    story.append(Spacer(1, 4 * mm))
    story.append(p(
        "Do not ship the Ultralytics package as a networked service pending AGPL verification. "
        "Kubernetes, Kafka, and a full central VMS are not four-day proof of scale."
    ))

    # 6
    story.append(p("6. Paper-by-paper disposition", "H1"))
    story.append(hrule())
    story.append(p(
        "Open-access PDFs in papers/ were read against the locks and the four-day cut. "
        "The table is the source of truth for what a paper may be used for. Do not upgrade "
        "a candidate into a lock."
    ))
    story.append(p("6.1 Detection, OCR, Indian plates", "H2"))
    story.append(table(
        ["Paper", "Disposition"],
        [
            ["Laroca et al., IJCNN 2018",
             "Supports video-level evaluation and temporal redundancy. Brazilian results are not Gujarat accuracy."],
            ["Silva &amp; Jung, WPOD-NET, ECCV 2018",
             "Supports perspective rectification. Reported complete pipeline was comparatively slow. Add only when skew is a measured failure."],
            ["Zherzdev &amp; Gruzdev, LPRNet, 2018",
             "Lightweight but Chinese-focused. Tanwar shows the Indian two-row limitation. Not the OCR path."],
            ["Laroca et al., layout-independent YOLO, 2019",
             "Supports explicit one-row / two-row classification and layout-aware validation. Copy the method, not Brazilian syntax."],
            ["Tanwar et al., Indian-LPR, 2021",
             "Strongest local open Indian detector-dataset evidence. Four-point warping: 66.3% to 75.6% character accuracy. 75% remains inadequate for deployment."],
            ["Indian truck plates, YOLOv7 + PARSeq, 2022",
             "Relevant for difficult commercial / two-line plates. Validation set is only 239 images. Retain as a fallback; do not dismiss completely."],
            ["Nadiminti et al., Indian E2E ANPR, 2022",
             "Supports COCO-pretrained detector adaptation rather than transferring a Chinese CCPD model."],
            ["Laroca et al., AOLP/CCPD near-duplicates, 2023",
             "Requires video-level or vehicle-level splits. Adjacent frames must never be divided between training and testing."],
            ["Sensors 2024, unconstrained LPR",
             "Strong Chinese CCPD results; explicitly acknowledges restricted plate coverage. Not Indian validation."],
            ["Awiros ANPR-OCR technical report",
             "Candidate, not locked. 98.42% / 96.91% two-row at ~5 ms/crop (RTX 3090) is OCR on pre-cropped plates. Held-out validation-set size for those scores is not stated; no external Gujarat-feed test; not a full detector-to-alert pipeline. Promising first benchmark, not expected portal-feed accuracy."],
            ["Cui et al., PaddleOCR 3.0",
             "Suitable runtime / toolkit. The general pretrained model is not by itself an Indian ANPR solution."],
            ["IJARSCT 2025, YOLOv8 + PaddleOCR (Indian)",
             "Supports the broad YOLO to PaddleOCR pipeline. Insufficient quantitative evidence for model selection."],
            ["Islas-Yañez et al., Sensors 2026, 26, 2785",
             "Filename and earlier review said 2025; the article is Sensors 2026. Closest published stack shape. Copy pipeline order only."],
            ["Korean YOLOv12 + OCR, Sensors 2026",
             "Useful mainly as a warning: tracking errors and integrated-pipeline overhead dominate isolated OCR benchmarks."],
        ],
        [48 * mm, CONTENT_W - 48 * mm],
    ))
    story.append(p("6.2 Tracking, GIS, matching", "H2"))
    story.append(table(
        ["Paper", "Disposition"],
        [
            ["Bewley et al., SORT, 2016",
             "Lightweight fallback for per-camera passage grouping. Detection quality dominates tracking quality. Default tracker for P0."],
            ["Zhang et al., ByteTrack, 2021/22",
             "Useful inside one camera for grouping detections and selecting the best crop. Not needed for cross-camera identity. Drop if midday Day 2 is missed."],
            ["Xie et al., multi-camera travel-time, 2022",
             "Visual-ReID travel-time system. Future-roadmap material, not necessary for registration-number tracking."],
            ["SAE EASE-MCVT, 2025",
             "Supports regional edge processing and central metadata. Visual ReID and its engineering framework are outside the four-day scope."],
            ["CRS R48160, ALPR hotlists, 2024",
             "Supports hotlist comparison, real-time alerts, privacy controls, and human review."],
            ["Misfiled SciData 2023 PDF (removed)",
             "REMOVED. Meteorological dropsonde dataset (ACTIVATE / western North Atlantic), not city-scale vehicle research. File is in papers/_excluded/."],
        ],
        [48 * mm, CONTENT_W - 48 * mm],
    ))

    # 7
    story.append(p("7. Datasets, syntax, tools", "H1"))
    story.append(hrule())
    story.append(p("7.1 Data to use", "H2"))
    story.append(table(
        ["Asset", "Role", "Do not"],
        [
            ["Indian-LPR (Tanwar) — 16k images, 4-point",
             "Detector train/val; two-row split is the hard set.",
             "Treat 75.6% as deployment-ready, or still-image mAP as government-feed accuracy."],
            ["Awiros weights (candidate)",
             "First OCR benchmark if the model-card licence verifies. Pin hash.",
             "Claim 98% on Gujarat portal feeds. Treat as a locked recogniser."],
            ["Indian truck PARSeq set (val n=239)",
             "Fallback for commercial / two-line plates.",
             "Dismiss it completely, or treat n=239 as sufficient validation."],
            ["Authorised own-feed video + synthetic watchlist",
             "Mandatory own-feed demo. Plates in the video must be in the list.",
             "Harvest real private GJ plates as “stolen.”"],
            ["Government portal feeds",
             "Scoring object. Protocols unknown until they decode on the host.",
             "Pretend own-feed satisfies the government-feed test."],
            ["Kaggle Gujarat plates (355 images)",
             "GJ font smoke test only.",
             "Train a detector from this alone."],
        ],
        [52 * mm, 70 * mm, CONTENT_W - 122 * mm],
    ))
    story.append(p(
        "CCPD, AOLP, and UFPR-ALPR are the wrong fonts/layouts for training. UFPR is only a "
        "reminder that unconstrained video is much harder than stills. Adjacent frames must "
        "never be split between training and testing (Laroca et al., arXiv:2304.04653)."
    ))
    story.append(p("7.2 Indian plate syntax (plausibility filter)", "H2"))
    story.append(table(
        ["Family", "Pattern", "Note"],
        [
            ["Standard (Motor Vehicles Act 1988)", "SS DD XX NNNN (series 0–3 letters; I/O unused)",
             "Example: GJ 01 AB 1234 — re-verify cited instruments."],
            ["Some states omit leading district 0", "SS D XX NNNN", "Gujarat / Delhi style"],
            ["Bharat series (MoRTH 2021)", "YY BH #### XX (I/O unused)",
             "Example: 26BH4567AB — must not be dropped"],
            ["Gujarat government", "GJ 18 G #### / GJ 18 GJ ####", "Not a private series"],
            ["Two-wheeler two-line", "State+RTO on line 1; series+number on line 2",
             "Main OCR failure mode"],
        ],
        [48 * mm, 72 * mm, CONTENT_W - 120 * mm],
    ))
    story.append(p(
        "Invalid syntax: keep the raw read, no auto-alert. Valid syntax + exact watchlist becomes an alert. "
        "Do not aggressively substitute O/0 or I/1 on the raw string."
    ))
    story.append(p("7.3 Tools that belong in the prototype", "H2"))
    story.append(table(
        ["Tool", "Job", "Caveat"],
        [
            ["MediaMTX", "One RTSP pull, then WebRTC + analytics tap.",
             "Drop relay refinements if midday Day 2 is missed. Licence is a verification task."],
            ["ONVIF Profile T / S / G / M", "GetProfiles / GetStreamUri; G playback; M vendor metadata.",
             "Which profiles the supplied cameras implement is a verification task."],
            ["FFmpeg", "Decode.",
             "Lowering inference FPS does not cut WAN if the source stream still moves."],
            ["FastALPR YOLOv9-t ONNX", "Candidate plate detector.",
             "Swap OCR independently. Licence re-verify."],
            ["PaddleOCR runtime + Awiros weights", "Candidate OCR on crops.",
             "Toolkit is not an Indian ANPR solution. Awiros is not locked. HF card licence is a verification task."],
            ["SORT (ByteTrack optional)", "In-camera passage grouping and best crop.",
             "Not cross-camera identity. Detection quality dominates tracking quality."],
            ["FastAPI + PostgreSQL/PostGIS + React/Leaflet",
             "API, registry, GIS, alert queue.",
             "No Kafka. Stop UI work if no Day-1 sighting."],
            ["Docker Compose + ONNX Runtime", "Prototype deploy and inference.",
             "Not Kubernetes. Not an AGPL Ultralytics service."],
        ],
        [48 * mm, 52 * mm, CONTENT_W - 100 * mm],
    ))

    # 8
    story.append(p("8. Matching, GIS, ingest, and evidence", "H1"))
    story.append(hrule())
    story.append(p("8.1 ANPR worker (per feed)", "H2"))
    for b in [
        "Decode at a calibrated analysis FPS. Do not hard-code 2 FPS. Start from 2 FPS only as a hypothesis. Calibrate from vehicle speed, plate pixel width, dwell time, blur, and measured passage recall.",
        "Detect plates (ONNX). Optional vehicle-first if plates are tiny in 1080p.",
        "Track with SORT. Passage = track ID + short gap. ByteTrack only if grouping quality needs it.",
        "Score crop: detector confidence × sharpness × minimum plate width.",
        "OCR top-K crops with the candidate recogniser. Keep raw strings.",
        "Independently designed multi-frame character consistency / vote. Not a patent implementation. Keep raw OCR plus the voted plate.",
        "Normalise (strip space/hyphen, uppercase, NFKC). Syntax flag only.",
        "Exact active watchlist becomes an alert. Optional near-match and syntax-ok go to a review queue, never rewrite OCR. Drop fuzzy if midday Day 2 is missed.",
        "Persist every sighting so a designated-vehicle GIS search works even without a prior watchlist hit.",
    ]:
        story.append(p("• " + b, "BulletBody"))
    story.append(p("8.2 Time, GIS, viewing", "H2"))
    story.append(p(
        "Store source_time, ingest_time, and clock_offset_ms. Display IST. Flag large drift. "
        "GIS: order exact (or designated) plates by corrected UTC, plot camera coordinates, "
        "draw dashed inferred links, and flag implausible implied speed. Do not claim a verified "
        "road polyline. Absence of a sighting is not proof of absence."
    ))
    story.append(p(
        "Viewing path: one source pull, relay to clients, tiles on demand. Analytics path: "
        "process all required accessible feeds continuously. Report analytics_active_count "
        "versus onboarded count. Under backpressure, drop analysis frames; never mark a camera "
        "healthy if analytics stopped."
    ))
    story.append(p("8.3 Alert workflow (operational analogue: CRS R48160)", "H2"))
    story.append(p(
        "US Congressional Research Service R48160 describes the law-enforcement pattern: "
        "read plate, compare to a hotlist, raise a real-time alert, then officer review. That is the "
        "hackathon test. States: New, acknowledged, confirmed / rejected. No automatic "
        "enforcement. Human review is mandatory, not optional polish."
    ))
    story.append(p("8.4 Minimum event schema", "H2"))
    story.append(table(
        ["Record", "Minimum content"],
        [
            ["Camera",
             "ID, department, GIS, source, capabilities, health (reported vs probed). Every accessible camera, including blocked ones with reason. Secrets referenced, never exported."],
            ["Sighting",
             "ID, camera, event time, raw and normalised plate, confidence, model version, evidence reference."],
            ["Watchlist",
             "Record ID, plate, purpose, priority, validity, authority, version, expiry."],
            ["Alert",
             "Sighting IDs, watchlist ID, match type (exact / fuzzy), status, unique dedup key."],
            ["Evidence / audit",
             "Crop or clip URI, hash, actor, action, time, retention, export log."],
        ],
        [32 * mm, CONTENT_W - 32 * mm],
    ))
    story.append(p(
        "Every event also carries trace_id, schema_version, source_time, ingest_time, event_id. "
        "A file hash detects byte changes; it does not prove legal admissibility.",
        "Caption",
    ))

    # 9
    story.append(p("9. Patents, policy, and verification tasks", "H1"))
    story.append(hrule())
    story.append(p("9.1 Patents — cite; do not implement as spec", "H2"))
    story.append(p(
        "Multi-frame OCR consensus is useful. Implement a simple independently designed "
        "voting / consistency method. Obtain IP review before making patent-derived "
        "functionality a production commitment. Patent legal status (filed, granted, expired, "
        "applicable jurisdiction) is itself a verification task."
    ))
    story.append(table(
        ["Patent", "Idea", "Our posture"],
        [
            ["US20230394850A1", "Vote characters across frames.",
             "Cite as related art. Independent consistency method only."],
            ["US20200074211A1", "ALPR + hotlist, then real-time LE alert.",
             "Same shape as the live test; cite, do not copy claims."],
            ["US20240355130A1", "Match a list by character overlap.",
             "Fuzzy review queue only. Droppable at midday Day 2."],
            ["US10839303B2 (SAP)", "Rewrite a bad read from neighbouring cameras.",
             "Cite. Do not auto-correct identity."],
            ["US20240265704A1", "VMS + GIS field of view / coverage.",
             "Model 1 gap-analysis citation."],
            ["KR101451115B1", "ONVIF + non-ONVIF ingest.",
             "Adapter pattern citation."],
        ],
        [42 * mm, 62 * mm, CONTENT_W - 104 * mm],
    ))
    story.append(p("9.2 Policy (engineering notes, not a legal opinion)", "H2"))
    story.append(p(
        "DPDP interpretation, MoRTH/NAPIX access, and related instruments were asserted in "
        "the earlier literature review. They are not established by the local paper collection. "
        "Treat them as verification tasks. Do not scrape VAHAN. Do not fake a live national-database "
        "integration. Prototype retention remains: short rolling buffer of all reads; persist "
        "watchlist hits, operator-confirmed events, and audit."
    ))
    story.append(p("9.3 External claims that must be re-verified", "H2"))
    story.append(table(
        ["Claim class", "Why it is not settled"],
        [
            ["ONVIF versions",
             "Profile T/G/M text on onvif.org is not proof that supplied Gujarat cameras implement those profiles."],
            ["Model licences",
             "PaddleOCR, FastALPR, Ultralytics AGPL, MediaMTX — re-check the current licence files."],
            ["Hugging Face model-card licences",
             "Awiros and any other weights. Pin hash only after the card is re-read."],
            ["MoRTH / NAPIX access",
             "Police data-sharing circulars do not grant this team access."],
            ["DPDP interpretation",
             "Requires legal review, not a literature-review assertion."],
            ["Patent status",
             "Google Patents hits are not a freedom-to-operate opinion."],
        ],
        [48 * mm, CONTENT_W - 48 * mm],
    ))

    # 10
    story.append(p("10. Remaining gaps and the Day-1 question", "H1"))
    story.append(hrule())
    story.append(p(
        "Research cannot close access, hardware, or organiser questions. The workspace still "
        "contains documentation and papers only — no code, schema, or prototype."
    ))
    story.append(table(
        ["Gap", "Why it blocks"],
        [
            ["Portal feed protocols on the actual host",
             "RTSP vs player vs files decides the first adapter. Own-feed does not satisfy the government-feed test. Escalate at midday Day 1 if nothing decodes."],
            ["How many of ~50 cameras actually decode",
             "Every accessible camera still belongs in the ledger. Analytics coverage must be the measured number, not 50 by assertion."],
            ["Named GPU / host throughput",
             "Awiros ~5 ms/crop is OCR-only on an RTX 3090, not this host, and not the full pipeline. Do not promise one GPU without a soak."],
            ["Deadline is four days, not 14",
             "P0 only. Parallel ownership. Feature freeze at Day 4 midday."],
            ["Awiros as a government-submission dependency",
             "Candidate. Licence card, hash, and Gujarat-feed accuracy are all unverified."],
            ["Camera clock / NTP",
             "Wrong order of GIS sightings if ignored."],
        ],
        [52 * mm, CONTENT_W - 52 * mm],
    ))
    story.append(p(
        "Illustrative (unmeasured) capacity from the team deck, for HLD honesty only: at "
        "2 Mb/s per camera, 50 cameras about 100 Mb/s and 1.08 TB/day; 80,000 cameras about "
        "160 Gb/s and 1.73 PB/day. The hybrid model does not duplicate that centrally. "
        "50 cameras × 2 analysis FPS = 100 FPS is arithmetic on a hypothesis, not a design lock."
    ))

    # 11
    story.append(p("11. Source index", "H1"))
    story.append(hrule())
    story.append(p("11.1 Project documents", "H2"))
    story.append(p("• docs/GUJARAT POLICE HACKATHON.md — official brief, models 1–5, test case, deliverables, evaluation.", "BulletBody"))
    story.append(p("• docs/Gujarat_CCTV_Hybrid_Team_Plan.pptx — 31-slide team plan, 31 August 2026 (14-day scenario on slide 22; original 72-hour path on slide 26).", "BulletBody"))
    story.append(p("• docs/FOUR_DAY_P0_PLAN.md — compressed four-person execution plan and hard checkpoints.", "BulletBody"))
    story.append(p("• docs/PHASE2_LITERATURE_REVIEW.md — corrected literature list with paper-by-paper disposition.", "BulletBody"))
    story.append(p("• papers/ — local PDFs after removing the misfiled dropsonde paper to papers/_excluded/.", "BulletBody"))
    story.append(p("11.2 Core local PDFs (keep in the evidence pack)", "H2"))
    story.append(table(
        ["File in papers/", "Cite as"],
        [
            ["2025_Awiros_ANPR_OCR_TechnicalReport.pdf",
             "Awiros, Data-Intelligent ANPR (Indian PP-OCRv5). Candidate OCR benchmark only."],
            ["2021_Tanwar_Indian_LPR_in_the_wild.pdf",
             "Tanwar, Tiwari, Chowdhry, arXiv:2111.06054"],
            ["2025_Cui_PaddleOCR_3.0.pdf",
             "Cui et al., PaddleOCR 3.0, arXiv:2507.05595"],
            ["2018_Laroca_YOLO_ALPR_IJCNN.pdf",
             "Laroca et al., IJCNN 2018, arXiv:1802.09567"],
            ["2019_Laroca_LayoutIndependent_YOLO_ALPR.pdf",
             "Laroca et al., arXiv:1909.01754"],
            ["2018_Silva_Jung_WPOD-NET_ECCV.pdf",
             "Silva &amp; Jung, ECCV 2018 — optional if skew is measured"],
            ["2022_Nadiminti_Indian_E2E_ANPR.pdf",
             "Nadiminti, Gaur, Bhardwaj, arXiv:2207.06657"],
            ["2025_PMC_YOLO_PaddleOCR_Modular.pdf",
             "Islas-Yañez et al., Sensors 2026, 26(9), 2785 — not 2025"],
            ["2016_Bewley_SORT.pdf",
             "Bewley et al., SORT, 2016 — default in-camera tracker"],
            ["2021_Zhang_ByteTrack.pdf",
             "Zhang et al., ECCV 2022, arXiv:2110.06864 — optional in-camera only"],
            ["2024_CRS_R48160_ALPR_Hotlists.pdf",
             "CRS Report R48160, 19 August 2024"],
            ["2022_Indian_Truck_Plates_YOLOv7_PARSeq.pdf",
             "Fallback for commercial / two-line plates (val n=239)"],
        ],
        [72 * mm, CONTENT_W - 72 * mm],
    ))
    story.append(p(
        "Do not cite papers/_excluded/2023_CityScale_Vehicle_Trajectories_SciData.pdf. "
        "It is Vömel et al., Scientific Data (2023) 10:753, dropsonde observations during ACTIVATE.",
        "Caption",
    ))
    story.append(Spacer(1, 6 * mm))
    story.append(callout(
        "<b>Limits of this report.</b> No source establishes that the proposed prototype already "
        "works. No formal legal compliance assessment, live government-system connection, "
        "hardware benchmark, supplier quote, IP opinion, or submission has been performed. "
        "Numerical costs and capacity figures in the team deck are illustrative. Claims in any "
        "PPT or HLD must match demonstrated behaviour. The four-day plan is a scope cut, not a "
        "promise that the live test is already feasible."
    ))

    def on_first(c, doc):
        cover_page(c, doc)

    def on_later(c, doc):
        later_page(c, doc)

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=LM,
        rightMargin=RM,
        topMargin=TM,
        bottomMargin=BM,
        title="Gujarat CCTV Hybrid — Corrected P0 Research Findings",
        author="Hackathon team research",
        subject="Four-day P0, paper dispositions, and corrections to the research draft",
    )
    doc.build(story, onFirstPage=on_first, onLaterPages=on_later)
    print("Wrote", OUT, "pages", doc.page)


if __name__ == "__main__":
    build()
