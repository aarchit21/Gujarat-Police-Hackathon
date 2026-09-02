"""Seed own-feed cameras and synthetic watchlist.

Government cameras come from cameras.json, not from this seed.
This script does not create placeholder government cameras to reach 50.
Default: upsert. Does not delete sightings or alerts.
Use --reset to wipe operational rows (not the schema).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import delete, select  # noqa: E402

from app.database import SessionLocal, init_db  # noqa: E402
from app.models import Alert, AuditEvent, Camera, Sighting, WatchlistEntry  # noqa: E402
from app.services.plates import normalize  # noqa: E402
from scripts.generate_own_feed import generate  # noqa: E402


def cameras() -> list[Camera]:
    return [
        Camera(
            id="CAM-HOME-AHM-001",
            name="Ahmedabad Home junction 1",
            department="Home",
            city="Ahmedabad",
            lat=23.0225,
            lng=72.5714,
            source_type="image_dir",
            source_uri=str(ROOT / "data" / "frames" / "cam-ahmedabad"),
            status="onboarded",
            status_reason="own-feed image directory (Home / Ahmedabad)",
            analytics_active=False,
            processing_mode="local_worker",
            analytics_policy="continuous",
            priority_class="A",
            network_class="good",
            compute_target="local-host",
            decode_status="untested",
        ),
        Camera(
            id="CAM-RTO-SUR-002",
            name="Surat RTO junction 2",
            department="RTO",
            city="Surat",
            lat=21.1702,
            lng=72.8311,
            source_type="image_dir",
            source_uri=str(ROOT / "data" / "frames" / "cam-surat"),
            status="onboarded",
            status_reason="own-feed image directory (RTO / Surat) — second source type/location",
            analytics_active=False,
            processing_mode="local_worker",
            analytics_policy="continuous",
            priority_class="B",
            network_class="good",
            compute_target="local-host",
            decode_status="untested",
        ),
    ]


WATCHLIST = [
    ("GJ 01 AB 1234", "stolen_vehicle", "high", "Designated demo plate. Synthetic."),
    ("GJ 05 CD 9999", "suspect_watchlist", "medium", "Negative control — may appear, not auto-critical."),
    ("26BH4567AB", "blacklisted_vehicle", "high", "Bharat-series syntax must remain in the grammar."),
]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="delete sightings, alerts, and cameras before seed")
    args = parser.parse_args(argv)
    generate()
    init_db()
    db = SessionLocal()
    try:
        if args.reset:
            db.execute(delete(Alert))
            db.execute(delete(Sighting))
            db.execute(delete(Camera))
            db.execute(delete(WatchlistEntry))
        for cam in cameras():
            existing = db.get(Camera, cam.id)
            if existing is None:
                db.add(cam)
                continue
            existing.name = cam.name
            existing.department = cam.department
            existing.city = cam.city
            existing.lat = cam.lat
            existing.lng = cam.lng
            if not existing.source_uri:
                existing.source_type = cam.source_type
                existing.source_uri = cam.source_uri
            if cam.source_type == "image_dir" and (existing.processing_mode or "deferred") == "deferred":
                existing.processing_mode = cam.processing_mode
                existing.analytics_policy = cam.analytics_policy
                existing.priority_class = cam.priority_class
                existing.network_class = cam.network_class
                existing.compute_target = cam.compute_target
                existing.source_type = cam.source_type
                existing.source_uri = cam.source_uri
            existing.analytics_active = False
        existing_plates = {w.plate_norm for w in db.scalars(select(WatchlistEntry))}
        for raw, purpose, priority, notes in WATCHLIST:
            key = normalize(raw)
            if key in existing_plates:
                continue
            db.add(
                WatchlistEntry(
                    plate_raw=raw,
                    plate_norm=key,
                    purpose=purpose,
                    priority=priority,
                    notes=notes,
                )
            )
        db.add(AuditEvent(action="seed", detail="own-feed cameras + synthetic watchlist (government cameras come from catalogue)"))
        db.commit()
        print("seeded/updated", len(cameras()), "own-feed cameras and watchlist")
    finally:
        db.close()


if __name__ == "__main__":
    main()
