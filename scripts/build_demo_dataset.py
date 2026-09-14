"""Build the dataset for the publicly hosted demonstration instance.

Writes a self-contained `data/demo/` -- database plus evidence imagery -- that
contains NOTHING from the government cameras. It never opens the working
database for writing and never reads `data/evidence/`.

Why this exists: the working corpus is 24,000 crops from live Ahmedabad CSITMS
cameras, with the site caption burned into the pixels and bystanders in frame,
captured under portal credentials that carry no permission to redistribute. It
cannot go on a cloud host. The synthetic own feed can, and it carries the part of
the story that actually demonstrates the product -- plate read -> watchlist match
-> alert -- which per README.md "is demonstrable on the own feed only".

The government feed contributes throughput, which a hosted demo cannot show in
any case.

    python scripts/build_demo_dataset.py
    python scripts/build_demo_dataset.py --observations 400

Then run against it with:

    DATABASE_URL=sqlite:///.../data/demo/cctv.db EVIDENCE_DIR=.../data/demo/evidence
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw  # noqa: E402

from scripts.generate_own_feed import _font, _plate, _scene  # noqa: E402

DEMO = ROOT / "data" / "demo"
DEMO_DB = DEMO / "cctv.db"
DEMO_EVIDENCE = DEMO / "evidence"
REAL_EVIDENCE = ROOT / "data" / "evidence"

PLATE = "GJ01AB1234"

CAMERAS = [
    dict(id="CAM-HOME-AHM-001", name="Ahmedabad Home junction 1", department="Home",
         city="Ahmedabad", lat=23.0225, lng=72.5714, label="CAM Home Ahmedabad · own-feed"),
    dict(id="CAM-RTO-SUR-002", name="Surat RTO junction 2", department="RTO",
         city="Surat", lat=21.1702, lng=72.8311, label="CAM RTO Surat · own-feed"),
]

# The demo has to show the vocabulary, not just the happy path. These weights
# put a realistic mix of states in front of a reviewer: mostly estimated colour
# awaiting review, a minority already confirmed by a person, and a tail the
# model abstained on entirely.
COLOURS = ["white", "silver", "black", "blue", "red", "grey", "yellow"]
COLOUR_STATES = (
    # (weight, colour_source, confidence range, verified)
    (58, "openvino:vehicle-attributes-recognition-barrier-0042", (0.56, 0.93), False),
    (22, "human_verified", (1.0, 1.0), True),
    (20, "abstained", (0.0, 0.0), False),
)
# Plate states, so PLATE NOT VISIBLE / UNREADABLE / NOT CHECKED all appear
# rather than only the successful read.
PLATE_STATES = ((30, "ok"), (25, "plate_not_visible"), (20, "plate_unreadable"), (25, "not_checked"))


def _pick(weighted, rng):
    total = sum(w for w, *_ in weighted)
    n = rng.uniform(0, total)
    for row in weighted:
        n -= row[0]
        if n <= 0:
            return row[1:] if len(row) > 2 else row[1]
    return weighted[-1][1:] if len(weighted[-1]) > 2 else weighted[-1][1]


def _crop(colour_name: str, idx: int) -> Image.Image:
    """A synthetic vehicle crop.

    Deliberately crude: it must be obvious at a glance that this is generated,
    so nobody mistakes the hosted demo for real footage.
    """
    img = Image.new("RGB", (320, 240), (52, 58, 55))
    draw = ImageDraw.Draw(img)
    body = {
        "white": (232, 232, 228), "silver": (186, 190, 193), "black": (38, 38, 40),
        "blue": (44, 84, 156), "red": (162, 44, 40), "grey": (118, 122, 124),
        "yellow": (214, 178, 44), "unknown": (96, 100, 102),
    }.get(colour_name, (96, 100, 102))
    draw.rectangle((40, 96, 280, 196), fill=body, outline=(20, 20, 20), width=3)
    draw.rectangle((70, 70, 250, 100), fill=body, outline=(20, 20, 20), width=3)
    draw.ellipse((70, 182, 110, 222), fill=(24, 24, 26))
    draw.ellipse((210, 182, 250, 222), fill=(24, 24, 26))
    draw.text((12, 10), "OWN-FEED · generated", fill=(200, 200, 180), font=_font(14))
    draw.text((12, 28), f"#{idx:04d}", fill=(150, 150, 150), font=_font(12))
    return img


def build(n_observations: int, seed: int) -> None:
    rng = random.Random(seed)

    if DEMO.exists():
        shutil.rmtree(DEMO)
    DEMO_EVIDENCE.mkdir(parents=True)

    # Point the app at the demo database BEFORE importing anything that binds an
    # engine at import time.
    import os

    os.environ["DATABASE_URL"] = f"sqlite:///{DEMO_DB.as_posix()}"
    os.environ["EVIDENCE_DIR"] = str(DEMO_EVIDENCE)

    from app.config import settings
    settings.database_url = f"sqlite:///{DEMO_DB.as_posix()}"
    settings.evidence_dir = DEMO_EVIDENCE

    from app.database import init_db, make_engine, make_session_factory
    from app.models import Alert, Camera, Sighting, VehicleObservation, WatchlistEntry
    from app.services.plates import normalize

    engine = make_engine(settings.database_url)
    init_db(engine)
    db = make_session_factory(engine)()

    # ---- cameras: the two synthetic ones, with every stream handle blank -----
    for spec in CAMERAS:
        db.add(Camera(
            id=spec["id"], name=spec["name"], department=spec["department"],
            city=spec["city"], lat=spec["lat"], lng=spec["lng"],
            source_type="image_dir",
            source_uri="",  # never an rtsp:// in this database
            status="onboarded", processing_mode="local_worker",
            analytics_policy="continuous", priority_class="A", network_class="good",
            analytics_active=False, decode_status="ok", coords_source="own_feed",
        ))
    db.commit()

    # ---- watchlist ----------------------------------------------------------
    watch = WatchlistEntry(
        plate_raw=PLATE, plate_norm=normalize(PLATE), purpose="stolen_vehicle",
        priority="high", authority="demo-synthetic",
        notes="Synthetic representative record. Not a real GJ plate.",
    )
    db.add(watch)
    db.commit()
    db.refresh(watch)

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(hours=24)

    # ---- observations -------------------------------------------------------
    made = 0
    for i in range(n_observations):
        cam = CAMERAS[i % len(CAMERAS)]
        seen = start + timedelta(seconds=rng.randint(0, 24 * 3600))
        colour = rng.choice(COLOURS)
        source, conf_range, verified = _pick(COLOUR_STATES, rng)
        conf = round(rng.uniform(*conf_range), 3)
        if source == "abstained":
            colour = "unknown"

        stamp = seen.strftime("%Y%m%dT%H%M%S%f")
        rel_dir = DEMO_EVIDENCE / cam["id"]
        rel_dir.mkdir(parents=True, exist_ok=True)
        vehicle_rel = f"{cam['id']}/{stamp}_demo_t{i}_vehicle.jpg"
        context_rel = f"{cam['id']}/{stamp}_demo_t{i}_context.jpg"
        _crop(colour, i).save(DEMO_EVIDENCE / vehicle_rel, quality=88)
        _scene(_plate(PLATE), 300 + (i % 9) * 40, 300, label=cam["label"]).save(
            DEMO_EVIDENCE / context_rel, quality=82)

        db.add(VehicleObservation(
            camera_id=cam["id"], run_id=f"demo-{seed}", track_id=f"t{i}",
            first_seen_at=seen, last_seen_at=seen + timedelta(seconds=rng.randint(2, 9)),
            detector="yolov8n", detector_confidence=round(rng.uniform(0.61, 0.95), 3),
            # The deployment gate is closed, so automatic type is 'unknown' here
            # exactly as it is in the working system. The demo must not imply a
            # capability the product withholds.
            vehicle_type="unknown", type_confidence=0.0,
            type_source="suppressed_pending_validation",
            vehicle_color=colour, color_confidence=conf, color_source=source,
            verified_vehicle_color=colour if verified else "",
            verified_by="operator" if verified else "",
            verified_at=seen + timedelta(minutes=12) if verified else None,
            review_status="verified" if verified else "unreviewed",
            evidence_path=vehicle_rel, context_evidence_path=context_rel,
            bbox_x=40, bbox_y=96, bbox_w=240, bbox_h=100,
            frame_width=1280, frame_height=720,
            metadata_json={"attributes": {
                "type_candidate": "", "type_candidate_confidence": 0.0,
                "color_source": source, "observations": rng.randint(2, 6),
            }, "demo": True},
        ))
        made += 1
        if made % 100 == 0:
            db.commit()
    db.commit()

    # ---- sightings and alerts: the plate -> watchlist -> alert path ----------
    # This is the sequence the whole demo exists to show, and the only one that
    # works at all: plate text is not recoverable from the government feeds.
    read_sightings: list[Sighting] = []
    for n, cam in enumerate(CAMERAS):
        for k in range(6):
            seen = start + timedelta(hours=4 * n + k, minutes=rng.randint(0, 50))
            read = _pick(PLATE_STATES, rng) == "ok" or k == 0  # guarantee one per camera
            # Write the image the row points at. An alert card with a broken
            # thumbnail reads as a bug in the product rather than a gap in the
            # fixture, and the alert card is the last thing a reviewer looks at.
            sighting_rel = f"{cam['id']}/demo_sighting_{n}{k}.jpg"
            _scene(_plate(PLATE), 360 + k * 30, 296, label=cam["label"]).save(
                DEMO_EVIDENCE / sighting_rel, quality=84)
            row = Sighting(
                camera_id=cam["id"],
                passage_id=f"demo-{cam['id']}-{n}{k}",
                source_time=seen,
                plate_raw=PLATE if read else "",
                plate_norm=normalize(PLATE) if read else "",
                syntax_ok=read,
                confidence=round(rng.uniform(0.72, 0.97), 3) if read else 0.0,
                model_id="demo-synthetic",
                evidence_path=sighting_rel,
                run_id=f"demo-{seed}",
            )
            db.add(row)
            if read:
                read_sightings.append(row)
    db.commit()

    for row in read_sightings[:2]:
        db.refresh(row)
        db.add(Alert(
            sighting_id=row.id, watchlist_id=watch.id, camera_id=row.camera_id,
            passage_id=row.passage_id, plate_norm=row.plate_norm,
            match_type="exact", status="new",
        ))
    db.commit()

    # ---- assertions: the whole point of the script ---------------------------
    from sqlalchemy import func, select

    rtsp = db.execute(select(func.count()).select_from(Camera).where(
        Camera.source_type == "rtsp")).scalar_one()
    assert rtsp == 0, f"{rtsp} rtsp cameras leaked into the demo database"

    gov = db.execute(select(func.count()).select_from(Camera).where(
        Camera.department == "government-catalogue")).scalar_one()
    assert gov == 0, f"{gov} government cameras leaked into the demo database"

    uris = db.execute(select(Camera.source_uri)).scalars().all()
    assert not any(u and "rtsp" in u.lower() for u in uris), "an rtsp uri survived"

    real = REAL_EVIDENCE.resolve()
    for path in DEMO_EVIDENCE.rglob("*.jpg"):
        assert real not in path.resolve().parents, f"{path} points into the real corpus"

    n_obs = db.execute(select(func.count()).select_from(VehicleObservation)).scalar_one()
    n_sight = db.execute(select(func.count()).select_from(Sighting)).scalar_one()
    n_alert = db.execute(select(func.count()).select_from(Alert)).scalar_one()
    n_img = sum(1 for _ in DEMO_EVIDENCE.rglob("*.jpg"))
    db.close()

    size = sum(p.stat().st_size for p in DEMO.rglob("*") if p.is_file()) / 1e6
    print("  cameras        2 (own feed only, 0 rtsp, 0 government)")
    print(f"  observations   {n_obs}")
    print(f"  sightings      {n_sight}  ·  alerts {n_alert}  ·  watchlist 1")
    print(f"  evidence       {n_img} generated images")
    print(f"  total size     {size:.1f} MB")
    print(f"\n  written to {DEMO}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--observations", type=int, default=250)
    ap.add_argument("--seed", type=int, default=20260915)
    args = ap.parse_args()
    build(args.observations, args.seed)
