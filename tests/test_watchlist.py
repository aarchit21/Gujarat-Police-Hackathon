from app.services.match import observed_plates, rematch_watchlist_entry
from app.services.pipeline import persist_sighting
from tests.conftest import add_camera, add_watchlist


def test_rematch_creates_alert_from_existing_sighting(db):
    cam = add_camera(db, id="cam01", source_type="rtsp")
    persist_sighting(
        db,
        cam,
        plate_raw="CSITMS",
        plate_norm="CSITMS",
        plate_voted="CSITMS",
        syntax=False,
        confidence=0.4,
        model_id="tesseract-opencv-p0",
        model_hash="x",
        evidence_path="evidence/cam01/x.jpg",
        run_id="r",
        frame_index=0,
        passage_id="p-gov",
        source_pts_ms=100.0,
        provider="local",
    )
    db.commit()
    wl = add_watchlist(db, plate="CSITMS")
    out = rematch_watchlist_entry(db, wl)
    db.commit()
    assert out["alerts_created"] == 1
    again = rematch_watchlist_entry(db, wl)
    assert again["alerts_created"] == 0


def test_observed_plates_flag_watchlisted(db):
    cam = add_camera(db)
    add_watchlist(db, plate="GJ01AB1234")
    persist_sighting(
        db,
        cam,
        plate_raw="GJ01AB1234",
        plate_norm="GJ01AB1234",
        plate_voted="GJ01AB1234",
        syntax=True,
        confidence=0.9,
        model_id="tesseract-opencv-p0",
        model_hash="x",
        evidence_path="",
        run_id="r",
        frame_index=0,
        passage_id="p2",
        source_pts_ms=1.0,
        provider="local",
    )
    persist_sighting(
        db,
        cam,
        plate_raw="1306",
        plate_norm="1306",
        plate_voted="1306",
        syntax=False,
        confidence=0.2,
        model_id="tesseract-opencv-p0",
        model_hash="x",
        evidence_path="",
        run_id="r",
        frame_index=1,
        passage_id="p3",
        source_pts_ms=2.0,
        provider="local",
    )
    db.commit()
    rows = {r["plate_norm"]: r for r in observed_plates(db)}
    assert rows["GJ01AB1234"]["watchlisted"] is True
    assert rows["1306"]["watchlisted"] is False
    assert rows["1306"]["count"] == 1
