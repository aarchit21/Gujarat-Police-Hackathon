"""Model 1 registry: manual entry, bulk import and gap analysis.

The behaviour under test is mostly about what onboarding refuses to do -- invent
coordinates, assert health it has not measured, or delete a camera that fell out
of an import file.
"""

from __future__ import annotations

import pytest

from app.models import Camera, VehicleObservation
from app.services.registry import (
    ImportError_,
    gap_analysis,
    gap_analysis_csv,
    normalise_row,
    onboard_cameras,
    parse_csv,
)
from tests.conftest import add_camera


# --- row validation -------------------------------------------------------


def test_row_without_id_is_rejected():
    with pytest.raises(ImportError_, match="id is required"):
        normalise_row({"name": "no id"})


def test_camera_id_alias_is_accepted():
    assert normalise_row({"camera_id": "CAM-9"})["id"] == "CAM-9"


def test_non_numeric_coordinate_is_rejected_not_guessed():
    with pytest.raises(ImportError_, match="lat is not a number"):
        normalise_row({"id": "CAM-9", "lat": "north", "lng": "72.5"})


def test_half_a_coordinate_pair_is_rejected():
    with pytest.raises(ImportError_, match="must be supplied together"):
        normalise_row({"id": "CAM-9", "lat": "23.0"})


def test_out_of_range_coordinate_is_rejected():
    with pytest.raises(ImportError_, match="out of range"):
        normalise_row({"id": "CAM-9", "lat": "991.0", "lng": "72.5"})


def test_unknown_source_type_is_rejected():
    with pytest.raises(ImportError_, match="source_type"):
        normalise_row({"id": "CAM-9", "source_type": "carrier-pigeon"})


def test_blank_fields_are_omitted_rather_than_written_as_empty():
    # A blank cell in a spreadsheet must not blank an existing value.
    assert "city" not in normalise_row({"id": "CAM-9", "city": "  "})["fields"]


def test_operational_state_cannot_be_set_by_an_import_file():
    # decode_status/analytics_active are measured by this host, never asserted.
    fields = normalise_row(
        {"id": "CAM-9", "decode_status": "ok", "analytics_active": "true", "status": "connected"}
    )["fields"]
    assert "decode_status" not in fields
    assert "analytics_active" not in fields
    assert "status" not in fields


# --- CSV parsing ----------------------------------------------------------


def test_csv_without_id_column_is_rejected():
    with pytest.raises(ImportError_, match="must contain an 'id'"):
        parse_csv("name,city\nfoo,Surat\n")


def test_csv_with_no_header_is_rejected():
    with pytest.raises(ImportError_, match="no header row"):
        parse_csv("")


def test_csv_preserves_row_order():
    rows = parse_csv("id,city\nB,Surat\nA,Rajkot\n")
    assert [r["id"] for r in rows] == ["B", "A"]


# --- onboarding -----------------------------------------------------------


def test_new_camera_starts_untested_and_inactive(db):
    onboard_cameras(db, [{"id": "CAM-NEW", "city": "Rajkot"}], actor="op")
    cam = db.get(Camera, "CAM-NEW")
    assert cam.decode_status == "untested"
    assert cam.analytics_active is False
    assert cam.status == "onboarded"


def test_camera_without_coordinates_is_marked_placeholder_not_invented(db):
    # Camera.lat/lng are non-nullable and default to 0.0, so the *marker* that a
    # position is unreal is coords_source -- exactly as the catalogue sync does
    # it. The number must never be presented as a surveyed position.
    onboard_cameras(db, [{"id": "CAM-NOGEO"}], actor="op")
    cam = db.get(Camera, "CAM-NOGEO")
    assert cam.coords_source == "placeholder"


def test_placeholder_camera_is_reported_as_a_coordinate_gap(db):
    onboard_cameras(db, [{"id": "CAM-NOGEO"}], actor="op")
    report = gap_analysis(db)
    assert [e["id"] for e in report["gaps"]["placeholder_coords"]] == ["CAM-NOGEO"]


def test_null_island_coordinates_are_never_reported_as_a_real_position(db):
    # 0,0 is the default, and is in the Atlantic. It must surface as a gap
    # whether it arrived as a placeholder or as an untagged default.
    add_camera(db, id="CAM-ZERO", lat=0.0, lng=0.0, coords_source="")
    report = gap_analysis(db)
    flagged = {e["id"] for e in report["gaps"]["placeholder_coords"]} | {
        e["id"] for e in report["gaps"]["coordinates_out_of_bounds"]
    }
    assert "CAM-ZERO" in flagged


def test_supplied_coordinates_are_marked_manual_entry(db):
    onboard_cameras(db, [{"id": "CAM-GEO", "lat": "23.02", "lng": "72.57"}], actor="op")
    cam = db.get(Camera, "CAM-GEO")
    assert cam.coords_source == "manual_entry"
    assert cam.lat == pytest.approx(23.02)


def test_import_updates_without_deleting_absent_cameras(db):
    add_camera(db, id="CAM-KEEP")
    onboard_cameras(db, [{"id": "CAM-OTHER"}], actor="op")
    # CAM-KEEP was not in the file and must survive untouched.
    assert db.get(Camera, "CAM-KEEP") is not None


def test_reimport_does_not_blank_fields_the_file_omits(db):
    onboard_cameras(db, [{"id": "CAM-X", "city": "Surat", "name": "Ring Road"}], actor="op")
    onboard_cameras(db, [{"id": "CAM-X", "name": "Ring Road North"}], actor="op")
    cam = db.get(Camera, "CAM-X")
    assert cam.name == "Ring Road North"
    assert cam.city == "Surat"


def test_bad_rows_are_reported_individually_and_good_rows_still_land(db):
    result = onboard_cameras(
        db, [{"id": "CAM-OK"}, {"name": "missing id"}, {"id": "CAM-BAD", "lat": "x", "lng": "1"}],
        actor="op",
    )
    assert result["created_count"] == 1
    assert result["error_count"] == 2
    assert {e["row"] for e in result["errors"]} == {2, 3}
    assert db.get(Camera, "CAM-OK") is not None
    assert db.get(Camera, "CAM-BAD") is None


def test_credentials_in_source_uri_are_not_echoed_in_the_gap_report(db):
    onboard_cameras(
        db,
        [{"id": "CAM-CRED", "source_type": "rtsp",
          "source_uri": "rtsp://user:hunter2@10.0.0.1:554/s1"}],
        actor="op",
    )
    report = gap_analysis(db)
    blob = gap_analysis_csv(report) + str(report)
    assert "hunter2" not in blob
    assert "***:***@10.0.0.1:554" in blob


# --- gap analysis ---------------------------------------------------------


def test_never_probed_camera_is_reported_as_untested_not_healthy(db):
    add_camera(db, id="CAM-UNTESTED", decode_status="untested")
    report = gap_analysis(db)
    assert report["totals"]["never_probed"] == 1
    assert [e["id"] for e in report["gaps"]["never_probed"]] == ["CAM-UNTESTED"]


def test_camera_with_no_observations_is_listed(db):
    add_camera(db, id="CAM-SILENT")
    report = gap_analysis(db)
    assert "CAM-SILENT" in [e["id"] for e in report["gaps"]["no_vehicle_observations"]]


def test_camera_with_observations_is_not_listed_as_silent(db):
    cam = add_camera(db, id="CAM-BUSY")
    db.add(VehicleObservation(camera_id=cam.id, run_id="r1", track_id="t1", detector="yolov8n"))
    db.commit()
    report = gap_analysis(db)
    silent = [e["id"] for e in report["gaps"]["no_vehicle_observations"]]
    assert "CAM-BUSY" not in silent
    assert report["totals"]["with_vehicle_observations"] == 1


def test_coordinates_outside_gujarat_are_flagged(db):
    add_camera(db, id="CAM-SEA", lat=-33.9, lng=151.2, coords_source="manual_entry")
    report = gap_analysis(db)
    assert [e["id"] for e in report["gaps"]["coordinates_out_of_bounds"]] == ["CAM-SEA"]


def test_city_with_no_decoding_camera_is_reported_uncovered(db):
    add_camera(db, id="CAM-A", city="Patan", decode_status="untested")
    add_camera(db, id="CAM-B", city="Surat", decode_status="ok")
    report = gap_analysis(db)
    assert "Patan" in report["uncovered_cities_no_working_camera"]
    assert "Surat" not in report["uncovered_cities_no_working_camera"]


def test_report_does_not_claim_road_coverage(db):
    add_camera(db, id="CAM-A")
    caveats = " ".join(gap_analysis(db)["caveats"]).lower()
    assert "not of road coverage" in caveats or "not a claim" in caveats
    assert "never_probed" in caveats


def test_gap_csv_has_a_row_per_camera_per_gap(db):
    add_camera(db, id="CAM-UNTESTED", decode_status="untested")
    csv_text = gap_analysis_csv(gap_analysis(db))
    lines = [ln for ln in csv_text.splitlines() if ln.strip()]
    assert lines[0].startswith("gap,camera_id")
    assert any(ln.startswith("never_probed,CAM-UNTESTED") for ln in lines)
