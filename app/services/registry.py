"""Camera registry onboarding and gap analysis (reference Model 1).

Model 1 asks for three onboarding routes -- bulk import, manual entry and API --
plus a gap-analysis report over the resulting inventory. The API route already
exists as the government catalogue sync (`app/services/catalogue.py`); this
module adds the other two and the report.

Two rules shape everything here:

* **Onboarding is additive.** An import never deletes a camera and never blanks
  a field it was not given. A camera that disappears from an import file keeps
  its row and its history, exactly as the catalogue sync behaves.
* **The report counts what was observed, not what was hoped for.** A camera that
  has never been probed is reported as never probed, not as healthy. "Uncovered"
  means no camera produced a vehicle observation there -- it does not claim the
  road itself is unmonitored, because the registry only knows about cameras it
  has been told about.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Camera, Sighting, VehicleObservation
from app.security import redact_url
from app.services.coverage import camera_origin

#: Columns a bulk import may set. `id` identifies the row; everything else is
#: optional and is only written when the file actually supplies it. Operational
#: state (decode_status, analytics_active, last_error, ...) is deliberately
#: absent: those are measured by this host, never asserted by an import file.
IMPORTABLE_FIELDS = (
    "name",
    "department",
    "city",
    "lat",
    "lng",
    "source_type",
    "source_uri",
    "substream_uri",
    "priority_class",
    "processing_mode",
    "analytics_policy",
    "network_class",
    "vendor",
    "model",
)

FLOAT_FIELDS = {"lat", "lng"}

VALID_SOURCE_TYPES = {"rtsp", "hls", "onvif", "image_dir", "file"}

# Gujarat's bounding box, generously padded. A camera outside it is almost
# certainly a bad coordinate (swapped lat/lng is the usual cause) and is
# reported rather than silently plotted in the sea.
GUJARAT_BOUNDS = {"lat_min": 19.5, "lat_max": 25.0, "lng_min": 68.0, "lng_max": 75.0}


class ImportError_(ValueError):
    """A row could not be onboarded. Carries the row number for the report."""


def _clean(value) -> str:
    return str(value if value is not None else "").strip()


def _parse_float(raw: str, field: str) -> float | None:
    if raw == "":
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ImportError_(f"{field} is not a number: {raw!r}") from exc


def normalise_row(raw: dict) -> dict:
    """Validate one onboarding row and return only the fields it actually sets.

    Raises ``ImportError_`` with a specific reason rather than guessing a value.
    """
    row = { _clean(k).lower(): v for k, v in (raw or {}).items() }
    camera_id = _clean(row.get("id") or row.get("camera_id"))
    if not camera_id:
        raise ImportError_("id is required")
    if len(camera_id) > 128:
        raise ImportError_("id is longer than 128 characters")

    out: dict = {}
    for field in IMPORTABLE_FIELDS:
        if field not in row:
            continue
        value = _clean(row[field])
        if field in FLOAT_FIELDS:
            parsed = _parse_float(value, field)
            if parsed is not None:
                out[field] = parsed
            continue
        if value == "":
            continue
        out[field] = value

    source_type = out.get("source_type", "")
    if source_type and source_type not in VALID_SOURCE_TYPES:
        raise ImportError_(
            f"source_type {source_type!r} is not one of {sorted(VALID_SOURCE_TYPES)}"
        )

    lat, lng = out.get("lat"), out.get("lng")
    if (lat is None) != (lng is None):
        raise ImportError_("lat and lng must be supplied together")
    if lat is not None:
        if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lng <= 180.0):
            raise ImportError_(f"coordinates out of range: {lat},{lng}")

    return {"id": camera_id, "fields": out}


def parse_csv(text: str) -> list[dict]:
    """Parse an uploaded CSV into raw row dicts, preserving file order."""
    stream = io.StringIO(text)
    reader = csv.DictReader(stream)
    if not reader.fieldnames:
        raise ImportError_("CSV has no header row")
    header = {(h or "").strip().lower() for h in reader.fieldnames}
    if "id" not in header and "camera_id" not in header:
        raise ImportError_("CSV header must contain an 'id' (or 'camera_id') column")
    return [dict(r) for r in reader]


def onboard_cameras(
    db: Session,
    rows: list[dict],
    *,
    actor: str,
    origin: str = "manual_entry",
) -> dict:
    """Create or update registry rows. Never deletes, never blanks.

    Returns a per-row outcome so the operator can see exactly which lines were
    rejected and why, instead of a single pass/fail.
    """
    created: list[str] = []
    updated: list[str] = []
    errors: list[dict] = []
    now = datetime.now(timezone.utc)

    for index, raw in enumerate(rows, start=1):
        try:
            parsed = normalise_row(raw)
        except ImportError_ as exc:
            errors.append({"row": index, "id": _clean((raw or {}).get("id")), "error": str(exc)})
            continue

        camera_id = parsed["id"]
        fields = parsed["fields"]
        camera = db.get(Camera, camera_id)

        if camera is None:
            camera = Camera(
                id=camera_id,
                name=fields.get("name") or camera_id,
                department=fields.get("department") or "local_registry",
                city=fields.get("city") or "",
                source_type=fields.get("source_type") or "rtsp",
                source_uri=fields.get("source_uri") or "",
                # A newly onboarded camera has never been contacted by this
                # host. Saying "untested" is the whole point of the ledger.
                status="onboarded",
                status_reason="onboarded via " + origin,
                decode_status="untested",
                processing_mode=fields.get("processing_mode") or "local_worker",
                analytics_policy=fields.get("analytics_policy") or "continuous",
                priority_class=fields.get("priority_class") or "B",
                network_class=fields.get("network_class") or "unknown",
                analytics_active=False,
                coords_source="manual_entry" if fields.get("lat") is not None else "placeholder",
            )
            for field, value in fields.items():
                setattr(camera, field, value)
            if fields.get("source_uri"):
                camera.protected_rtsp_url_or_reference = fields["source_uri"]
            db.add(camera)
            created.append(camera_id)
        else:
            for field, value in fields.items():
                setattr(camera, field, value)
            if fields.get("source_uri"):
                camera.protected_rtsp_url_or_reference = fields["source_uri"]
            if fields.get("lat") is not None and (camera.coords_source or "") in {"", "placeholder"}:
                camera.coords_source = "manual_entry"
            updated.append(camera_id)

        camera.catalogue_synced_at = camera.catalogue_synced_at or now

    db.commit()
    return {
        "ok": not errors,
        "created": created,
        "updated": updated,
        "created_count": len(created),
        "updated_count": len(updated),
        "error_count": len(errors),
        "errors": errors,
        "origin": origin,
        "actor": actor,
        "note": (
            "Import is additive. Cameras absent from this file were left untouched, "
            "and every new camera starts as decode_status=untested until this host "
            "actually opens the stream."
        ),
    }


def _observation_counts(db: Session) -> dict[str, int]:
    rows = db.execute(
        select(VehicleObservation.camera_id, func.count(VehicleObservation.id))
        .group_by(VehicleObservation.camera_id)
    ).all()
    return {cam_id: int(n) for cam_id, n in rows}


def _sighting_counts(db: Session) -> dict[str, int]:
    rows = db.execute(
        select(Sighting.camera_id, func.count(Sighting.id))
        .where(Sighting.plate_norm.isnot(None), Sighting.plate_norm != "")
        .group_by(Sighting.camera_id)
    ).all()
    return {cam_id: int(n) for cam_id, n in rows}


def gap_analysis(db: Session) -> dict:
    """Where the fleet is blind, and why.

    Every bucket is derived from persisted state. Nothing here is an estimate of
    road coverage -- the registry cannot know about a road that has no camera in
    it. It reports gaps in the *inventory it holds*.
    """
    cameras = list(db.scalars(select(Camera).order_by(Camera.id)))
    obs = _observation_counts(db)
    plates = _sighting_counts(db)

    never_probed: list[dict] = []
    decode_failed: list[dict] = []
    placeholder_coords: list[dict] = []
    out_of_bounds: list[dict] = []
    no_observations: list[dict] = []
    no_plate_reads: list[dict] = []

    by_city: dict[str, dict] = {}

    for c in cameras:
        n_obs = obs.get(c.id, 0)
        n_plates = plates.get(c.id, 0)
        city = (c.city or "").strip() or "(city not supplied)"
        bucket = by_city.setdefault(
            city,
            {
                "city": city,
                "cameras": 0,
                "decode_ok": 0,
                "analytics_active": 0,
                "vehicle_observations": 0,
                "plate_reads": 0,
                "placeholder_coords": 0,
            },
        )
        bucket["cameras"] += 1
        bucket["vehicle_observations"] += n_obs
        bucket["plate_reads"] += n_plates
        if c.decode_status == "ok":
            bucket["decode_ok"] += 1
        if c.analytics_active:
            bucket["analytics_active"] += 1

        entry = {
            "id": c.id,
            "name": c.name or "",
            "city": c.city or "",
            "department": c.department or "",
            "origin": camera_origin(c),
            "status": c.status or "",
            "decode_status": c.decode_status or "",
            # Never emit a camera URL that could carry credentials.
            "source": redact_url(c.source_uri),
            "vehicle_observations": n_obs,
            "plate_reads": n_plates,
        }

        if (c.decode_status or "untested") == "untested":
            never_probed.append(entry)
        if c.decode_status == "failed":
            decode_failed.append({**entry, "last_error": (c.last_error or "")[:300]})
        if (c.coords_source or "") == "placeholder" or c.lat is None or c.lng is None:
            placeholder_coords.append(entry)
            bucket["placeholder_coords"] += 1
        elif not (
            GUJARAT_BOUNDS["lat_min"] <= c.lat <= GUJARAT_BOUNDS["lat_max"]
            and GUJARAT_BOUNDS["lng_min"] <= c.lng <= GUJARAT_BOUNDS["lng_max"]
        ):
            out_of_bounds.append({**entry, "lat": c.lat, "lng": c.lng})
        if n_obs == 0:
            no_observations.append(entry)
        if n_plates == 0:
            no_plate_reads.append(entry)

    cities = sorted(by_city.values(), key=lambda b: (-b["cameras"], b["city"]))
    uncovered_cities = [b["city"] for b in cities if b["decode_ok"] == 0]
    silent_cities = [b["city"] for b in cities if b["vehicle_observations"] == 0]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "totals": {
            "cameras": len(cameras),
            "decode_ok": sum(1 for c in cameras if c.decode_status == "ok"),
            "decode_failed": len(decode_failed),
            "never_probed": len(never_probed),
            "analytics_active": sum(1 for c in cameras if c.analytics_active),
            "with_vehicle_observations": sum(1 for c in cameras if obs.get(c.id, 0) > 0),
            "with_plate_reads": sum(1 for c in cameras if plates.get(c.id, 0) > 0),
            "placeholder_coords": len(placeholder_coords),
            "coordinates_out_of_bounds": len(out_of_bounds),
        },
        "by_city": cities,
        "gaps": {
            "never_probed": never_probed,
            "decode_failed": decode_failed,
            "placeholder_coords": placeholder_coords,
            "coordinates_out_of_bounds": out_of_bounds,
            "no_vehicle_observations": no_observations,
            "no_plate_reads": no_plate_reads,
        },
        "uncovered_cities_no_working_camera": uncovered_cities,
        "cities_with_no_vehicle_observations": silent_cities,
        "caveats": [
            "This is a gap analysis of the camera inventory this host holds, not of "
            "road coverage. A location with no registered camera does not appear here.",
            "'never_probed' means this host has not yet opened the stream. It is not a "
            "claim that the camera is down.",
            "'placeholder_coords' cameras are drawn at a fallback map position because "
            "the source catalogue supplied no lat/lng. Their map position is not survey data.",
            "'no_plate_reads' is expected on most cameras in this deployment: plate text "
            "is not reliably recoverable from the current feeds, while vehicle detection "
            "and tracking still work. A camera with observations but no plate reads is "
            "working as measured, not broken.",
            "Camera URLs are redacted; credentials are never included in this report.",
        ],
    }


def gap_analysis_csv(report: dict) -> str:
    """Flat per-camera CSV of every gap bucket, for attaching to a submission."""
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(
        ["gap", "camera_id", "name", "city", "department", "origin",
         "status", "decode_status", "vehicle_observations", "plate_reads", "source"]
    )
    for gap_name, entries in (report.get("gaps") or {}).items():
        for e in entries:
            writer.writerow([
                gap_name, e.get("id", ""), e.get("name", ""), e.get("city", ""),
                e.get("department", ""), e.get("origin", ""), e.get("status", ""),
                e.get("decode_status", ""), e.get("vehicle_observations", 0),
                e.get("plate_reads", 0), e.get("source", ""),
            ])
    return out.getvalue()
