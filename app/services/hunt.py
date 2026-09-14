"""Time-multiplex analytics across all live catalogue cameras.

Concurrent RTSP slots are operator-chosen (default 4). Hunt rotates those
slots so every live government feed is visited. It does not open every
catalogue stream at once and is not a central VMS.
"""
from __future__ import annotations

import random
import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Alert, AuditEvent, Camera, Sighting
from app.services.coverage import camera_origin
from app.services.ingest import rtsp_url_for
from app.services.vehicle_event import is_recordable_plate


PIN_DEFAULT = ("cam01", "cam02", "cam03", "cam05")


def slot_count() -> int:
    """Configured live RTSP slots. Operator UI/env can change this; it is not locked to 30."""
    return max(1, int(settings.max_concurrent_workers or 4))


def _rotation_key(cam: Camera, rng: random.Random) -> tuple:
    """Order within a decode tier: least-recently-hunted first, random tiebreak.

    Selection used to be `sorted by id`, with PIN_DEFAULT forced to the front,
    so every click pinned cam01/02/03/05 and the rest of the catalogue was
    never visited. Ordering by ``last_hunted_at`` (never-hunted first) rotates
    fairly across clicks, and the random tiebreak stops equal timestamps from
    collapsing back to alphabetical order.
    """
    never_hunted = cam.last_hunted_at is None
    last = cam.last_hunted_at.timestamp() if cam.last_hunted_at else 0.0
    return (0 if never_hunted else 1, last, rng.random())


def decode_ok_pin_ids(db: Session, n: int | None = None, *, rng: random.Random | None = None) -> list[str]:
    """Fill the capture slots, preferring decode-ok and least-recently-hunted.

    Decode tiers are kept because pinning a known-dead camera wastes a slot:
    ok -> untested -> failed. Rotation happens *within* each tier.
    """
    count = max(1, int(n if n is not None else slot_count()))
    cams = hunt_targets(db, pinned_only=False)
    mode = str(getattr(settings, "hunt_rotation", "least_recent") or "least_recent").lower()
    if mode == "fixed":
        ok = [c.id for c in cams if c.decode_status == "ok"]
        rest = [c.id for c in cams if c.id not in ok]
        return (ok + rest)[:count]

    seed = int(getattr(settings, "hunt_pin_seed", 0) or 0)
    rng = rng or (random.Random(seed) if seed else random.Random())

    tiers: dict[int, list[Camera]] = {0: [], 1: [], 2: []}
    for cam in cams:
        status = (cam.decode_status or "untested").lower()
        tiers[0 if status == "ok" else 1 if status == "untested" else 2].append(cam)

    for tier in (0, 1, 2):
        if mode == "random":
            rng.shuffle(tiers[tier])
        else:
            tiers[tier].sort(key=lambda c: _rotation_key(c, rng))

    # Explore / exploit. Strict tiering alone is a trap: an untested camera is
    # never given a slot, so it stays untested forever and the same handful of
    # decode-ok cameras is pinned on every click. Reserving part of the slots
    # for untested cameras lets repeated clicks work through the catalogue and
    # actually fill in decode_status.
    explore = float(getattr(settings, "hunt_explore_fraction", 0.5) or 0.0)
    explore_slots = min(len(tiers[1]), int(round(count * max(0.0, min(1.0, explore)))))
    exploit_slots = count - explore_slots

    chosen = [c.id for c in tiers[0][:exploit_slots]]
    chosen += [c.id for c in tiers[1][:explore_slots]]

    # Backfill if either pool ran short, so slots are never left idle.
    if len(chosen) < count:
        taken = set(chosen)
        for tier in (0, 1, 2):
            for cam in tiers[tier]:
                if len(chosen) >= count:
                    break
                if cam.id not in taken:
                    chosen.append(cam.id)
                    taken.add(cam.id)
    return chosen[:count]


def hunt_targets(
    db: Session,
    *,
    pinned_only: bool = False,
    pin_ids: list[str] | None = None,
    rng: random.Random | None = None,
) -> list[Camera]:
    rows = list(db.scalars(select(Camera).order_by(Camera.id)))
    explicit = [str(x).strip() for x in (pin_ids or []) if str(x).strip()]
    allow = set(explicit) or {str(x).strip() for x in PIN_DEFAULT}
    out = []
    for cam in rows:
        if not cam.catalogue_live and camera_origin(cam) != "government_catalogue":
            continue
        if not (rtsp_url_for(cam) or cam.hls_url):
            continue
        if pinned_only and cam.id not in allow:
            continue
        out.append(cam)

    def tier(cam: Camera) -> int:
        status = (cam.decode_status or "untested").lower()
        return 0 if status == "ok" else 1 if status == "untested" else 2

    if explicit:
        # An operator named these cameras; honour that order exactly.
        rank = {cid: i for i, cid in enumerate(explicit)}
        out.sort(key=lambda c: (rank.get(c.id, len(rank)), tier(c), c.id))
        return out

    mode = str(getattr(settings, "hunt_rotation", "least_recent") or "least_recent").lower()
    if mode == "fixed":
        # Legacy behaviour: PIN_DEFAULT first, then alphabetical. This is what
        # made every hunt start at cam01-cam05 and never reach the rest.
        out.sort(key=lambda c: (0 if c.id in allow else 1, tier(c), c.id))
        return out

    seed = int(getattr(settings, "hunt_pin_seed", 0) or 0)
    rng = rng or (random.Random(seed) if seed else random.Random())
    if mode == "random":
        rng.shuffle(out)
        out.sort(key=tier)
    else:
        out.sort(key=lambda c: (tier(c), *_rotation_key(c, rng)))
    return out


def promote_hunt_camera(cam: Camera) -> bool:
    if cam.processing_mode in {"deferred", "", None}:
        cam.processing_mode = "local_worker"
        cam.analytics_policy = "continuous"
        return True
    return False


def start_hunt(
    manager,
    db: Session,
    *,
    actor: str = "operator",
    pinned_only: bool = False,
    pin_ids: list[str] | None = None,
    vision_only: bool = False,
    max_concurrent: int | None = None,
) -> dict:
    from app.services.workers import persist_concurrency

    if max_concurrent is not None:
        slots = persist_concurrency(db, max_concurrent, mgr=manager)
    else:
        slots = max(1, int(getattr(manager, "max_workers", None) or slot_count()))
    if pinned_only and not pin_ids:
        pin_ids = decode_ok_pin_ids(db, slots)
    targets = hunt_targets(db, pinned_only=pinned_only, pin_ids=pin_ids)
    promoted = []
    for cam in targets:
        if promote_hunt_camera(cam):
            promoted.append(cam.id)
    db.commit()

    prev = set(getattr(manager, "hunt_target_ids", set()) or set()) | set(getattr(manager, "pin_hold_ids", set()) or set())
    manager.end_hunt()
    if hasattr(manager, "end_pin"):
        manager.end_pin()
    for camera_id in prev:
        manager.stop(db, camera_id, actor=actor)

    manager.vision_only = bool(vision_only)
    target_ids = [c.id for c in targets]
    if pinned_only:
        manager.begin_pin(target_ids)
    else:
        manager.begin_hunt(target_ids)

    running = [
        w.get("camera_id")
        for w in manager.snapshot().get("workers", [])
        if w.get("camera_id") in set(target_ids)
    ]
    for camera_id in running:
        manager.stop(db, camera_id, actor=actor)

    started, queued, failed = [], [], []
    for cam in targets:
        out = manager.start(db, cam.id, actor=actor)
        state = out.get("state") or ""
        if not out.get("ok"):
            failed.append({"id": cam.id, "error": out.get("error")})
        elif state == "queued":
            queued.append(cam.id)
        else:
            started.append(cam.id)

    db.add(
        AuditEvent(
            actor=actor,
            action="hunt_start",
            detail=f"targets={len(targets)} started={len(started)} queued={len(queued)} dwell={settings.hunt_dwell_seconds}s",
        )
    )
    db.commit()
    status = hunt_status(manager, db)
    status.update(
        {
            "ok": True,
            "promoted": promoted,
            "started": started,
            "queued": queued,
            "failed": failed,
            "pinned_only": pinned_only,
            "vision_only": bool(getattr(manager, "vision_only", False)),
            "disclaimer": (
                f"Vision-only A/B on {slots} pinned live cameras. YOLO still runs."
                if vision_only
                else (
                    f"Pinned {len(targets)} working cameras ({slots} concurrent slots). "
                    "These stay open and reconnect; they do not rotate after 28s. "
                    "Change Max concurrent cameras to pin a different count."
                )
                if pinned_only
                else (
                    f"This host hunts {slots} government streams at a time and visits all live catalogue "
                    "cameras each cycle. Not every catalogue camera at once. Not a central VMS."
                )
            ),
        }
    )
    return status


def stop_hunt(manager, db: Session, *, actor: str = "operator") -> dict:
    ids = list(set(manager.hunt_target_ids) | set(getattr(manager, "pin_hold_ids", set()) or set()))
    manager.end_hunt()
    if hasattr(manager, "end_pin"):
        manager.end_pin()
    for camera_id in ids:
        manager.stop(db, camera_id, actor=actor)
    db.add(AuditEvent(actor=actor, action="hunt_stop", detail=f"stopped={len(ids)}"))
    db.commit()
    return {"ok": True, "stopped": ids, **hunt_status(manager, db)}


def hunt_status(manager, db: Session) -> dict:
    snap = manager.snapshot()
    pinned_ids = list(getattr(manager, "pin_hold_ids", set()) or set())
    targets = list(manager.hunt_target_ids) or pinned_ids
    visited = sorted(manager.hunt_visited)
    gov_ids = targets or [c.id for c in hunt_targets(db)]
    vehicles = 0
    plates = 0
    alerts = 0
    if gov_ids:
        rows = list(db.scalars(select(Sighting).where(Sighting.camera_id.in_(gov_ids))))
        vehicles = len(rows)
        plates = sum(1 for s in rows if is_recordable_plate(s.plate_norm))
        alerts = db.scalar(select(func.count(Alert.id)).where(Alert.camera_id.in_(gov_ids))) or 0
    hunting = [
        w.get("camera_id")
        for w in snap.get("workers", [])
        if w.get("status") in {"running", "starting"}
    ]
    last_hunted = [
        {"id": c.id, "last_hunted_at": c.last_hunted_at.isoformat() if c.last_hunted_at else None}
        for c in db.scalars(select(Camera).where(Camera.last_hunted_at.isnot(None)).order_by(Camera.last_hunted_at.desc()).limit(12))
    ]
    total = len(gov_ids)
    vo = ""
    if getattr(manager, "vision_only", False):
        names = [m.strip() for m in (settings.vision_only_models or "").split(",") if m.strip()]
        vo = " · vision-only " + (" then ".join(names) if names else "on")
    return {
        "enabled": bool(manager.hunt_enabled or pinned_ids),
        "pinned": bool(pinned_ids),
        "cycle_id": manager.hunt_cycle_id,
        "cycle": manager.hunt_cycle,
        "hunting": hunting,
        "hunting_count": len(hunting),
        "max_concurrent": snap.get("max_concurrent") or settings.max_concurrent_workers,
        "visited": visited,
        "visited_count": len(visited),
        "total": total,
        "queued": snap.get("queued") or [],
        "queued_count": snap.get("queued_count") or 0,
        "vehicles_seen": vehicles,
        "plates_read": plates,
        "alerts": alerts,
        "dwell_seconds": settings.hunt_dwell_seconds,
        "max_frames": settings.hunt_max_frames,
        "last_hunted": last_hunted,
        "label": (
            f"Pinned {len(hunting)}/{len(pinned_ids)} working cameras (held open, not rotating) · "
            f"{vehicles} vehicles · {plates} plates"
            if pinned_ids and not manager.hunt_enabled
            else f"Hunting {len(hunting)}/{total} · visited {len(visited)}/{total} this cycle · "
            f"{vehicles} vehicles · {plates} plates{vo}"
            if manager.hunt_enabled
            else (
                f"Hunt idle — {snap.get('max_concurrent') or slot_count()} live streams at a time. "
                "Set Max concurrent cameras, then pin working cameras or hunt all live feeds."
            )
        ),
    }


def mark_camera_hunted(db: Session, camera_id: str) -> None:
    cam = db.get(Camera, camera_id)
    if cam is None:
        return
    cam.last_hunted_at = datetime.now(timezone.utc)
    db.add(cam)
