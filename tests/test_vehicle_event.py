import json

from app.services.pipeline import persist_sighting
from app.services.vehicle_event import (
    VEHICLE_PROMPT,
    build_vehicle_event,
    is_recordable_plate,
    parse_vehicle_payload,
)
from tests.conftest import add_camera, add_watchlist


def test_prepare_live_anpr_frame_masks_hud():
    import numpy as np

    from app.services.anpr import mask_hud, prepare_live_anpr_frame

    frame = np.full((200, 400, 3), 80, dtype=np.uint8)
    frame[:20] = 255
    masked = mask_hud(frame)
    assert int(masked[5, 10, 0]) == 0
    assert int(masked[100, 200, 0]) == 80
    zoomed = prepare_live_anpr_frame(frame)
    assert zoomed is not None and zoomed.size > 0


def test_empty_vehicle_json_loads(db):
    from sqlalchemy import text

    cam = add_camera(db)
    persist_sighting(
        db,
        cam,
        plate_raw="GJ01AB1234",
        plate_norm="GJ01AB1234",
        plate_voted="GJ01AB1234",
        syntax=True,
        confidence=0.9,
        model_id="ollama-vision-p0",
        model_hash="x",
        evidence_path="",
        run_id="r",
        frame_index=0,
        passage_id="p-empty-json",
        source_pts_ms=1.0,
        provider="ollama_vision",
    )
    db.commit()
    db.execute(text("UPDATE sightings SET vehicle_json = '' WHERE passage_id = 'p-empty-json'"))
    db.commit()
    db.expire_all()
    from sqlalchemy import select as sel

    from app.models import Sighting

    row = db.scalar(sel(Sighting).where(Sighting.passage_id == "p-empty-json"))
    assert row is not None
    assert row.vehicle_json is None or row.vehicle_json == {}


def test_recordable_plate_accepts_indian_syntax():
    assert is_recordable_plate("GJ01AB1234") is True
    assert is_recordable_plate("26BH4567AB") is True
    assert is_recordable_plate("GJG1AB1234") is True


def test_recordable_plate_rejects_overlay():
    assert is_recordable_plate("1306") is False
    assert is_recordable_plate("CSITMS") is False
    assert is_recordable_plate("PTZ2") is False
    assert is_recordable_plate("22561F") is False
    assert is_recordable_plate("O") is False
    assert is_recordable_plate("2275603") is False


def test_parse_vehicle_payload_and_prompt_has_no_watchlist():
    assert "GJ01" not in VEHICLE_PROMPT
    parsed = parse_vehicle_payload(
        '{"plate_text":"GJ05CD1234","vehicle_type":"car","make":"Maruti","model":"Swift","color":"white","confidence":0.8}'
    )
    assert parsed["plate_norm"] == "GJ05CD1234"
    assert parsed["vehicle_type"] == "car"
    assert parsed["vehicle_make"] == "Maruti"
    assert parsed["vehicle_model"] == "Swift"
    assert parsed["vehicle_color"] == "white"


def test_persist_stores_vehicle_json(db):
    cam = add_camera(db)
    add_watchlist(db)
    sighting, alert, created = persist_sighting(
        db,
        cam,
        plate_raw="GJ01AB1234",
        plate_norm="GJ01AB1234",
        plate_voted="GJ01AB1234",
        syntax=True,
        confidence=0.9,
        model_id="ollama:gemma4:e4b",
        model_hash="x",
        evidence_path="",
        run_id="r",
        frame_index=0,
        passage_id="p-json",
        source_pts_ms=1.0,
        provider="ollama_vision",
        vehicle_type="car",
        vehicle_make="Maruti",
        vehicle_model="Swift",
        vehicle_color="white",
    )
    db.commit()
    payload = sighting.vehicle_json if isinstance(sighting.vehicle_json, dict) else json.loads(sighting.vehicle_json)
    assert payload["vehicle"]["number"] == "GJ01AB1234"
    assert payload["observed_at_ist"]
    assert "IST" in payload["observed_at_ist"]
    assert payload["vehicle"]["type"] == "car"
    assert payload["vehicle"]["make"] == "Maruti"
    assert payload["camera_id"] == cam.id
    assert payload["watchlist_matched"] is True
    assert created is True
    assert "VAHAN" in payload["disclaimer"]
    rebuilt = build_vehicle_event(camera=cam, sighting=sighting, extras={"vehicle_type": "car"})
    assert rebuilt["schema_version"] == 1
    assert alert is not None
