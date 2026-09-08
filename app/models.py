from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    department: Mapped[str] = mapped_column(String(80))
    city: Mapped[str] = mapped_column(String(80), default="")
    lat: Mapped[float] = mapped_column(Float, default=0.0)
    lng: Mapped[float] = mapped_column(Float, default=0.0)
    source_type: Mapped[str] = mapped_column(String(32), default="blocked")
    source_uri: Mapped[str] = mapped_column(Text, default="")
    substream_uri: Mapped[str] = mapped_column(Text, default="")
    priority_class: Mapped[str] = mapped_column(String(8), default="D")
    processing_mode: Mapped[str] = mapped_column(String(32), default="deferred")
    analytics_policy: Mapped[str] = mapped_column(String(32), default="on_demand")
    compute_target: Mapped[str] = mapped_column(String(64), default="")
    network_class: Mapped[str] = mapped_column(String(32), default="offline")
    target_analysis_fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="deferred")
    status_reason: Mapped[str] = mapped_column(Text, default="")
    analytics_active: Mapped[bool] = mapped_column(Boolean, default=False)
    last_frame_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str] = mapped_column(Text, default="")
    capabilities: Mapped[str] = mapped_column(Text, default="")
    vendor: Mapped[str] = mapped_column(String(80), default="")
    model: Mapped[str] = mapped_column(String(80), default="")
    clock_offset_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    catalogue_camera_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    catalogue_live: Mapped[bool] = mapped_column(Boolean, default=False)
    codec: Mapped[str] = mapped_column(String(32), default="")
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reported_fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    bitrate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    protected_rtsp_url_or_reference: Mapped[str] = mapped_column(Text, default="")
    whep_url: Mapped[str] = mapped_column(Text, default="")
    hls_url: Mapped[str] = mapped_column(Text, default="")
    catalogue_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    decode_tested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    decode_status: Mapped[str] = mapped_column(String(32), default="untested")
    source_pts_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_pts_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    reconnect_count: Mapped[int] = mapped_column(Integer, default=0)
    active_protocol: Mapped[str] = mapped_column(String(16), default="")
    measured_worker_fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    measured_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    coords_source: Mapped[str] = mapped_column(String(24), default="")
    last_hunted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # inherit follows the system ANPR safety switch.  The switch itself is
    # deliberately persisted in SystemState so an operator choice survives a
    # restart without being baked into a camera feed definition.
    plate_recognition_mode: Mapped[str] = mapped_column(String(12), default="inherit")

    sightings: Mapped[list["Sighting"]] = relationship(back_populates="camera")
    vehicle_observations: Mapped[list["VehicleObservation"]] = relationship(back_populates="camera")

    @property
    def latitude(self) -> float:
        return self.lat

    @property
    def longitude(self) -> float:
        return self.lng


class WatchlistEntry(Base):
    __tablename__ = "watchlist"
    __table_args__ = (Index("ix_watchlist_active_plate", "active", "plate_norm"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    plate_raw: Mapped[str] = mapped_column(String(32))
    plate_norm: Mapped[str] = mapped_column(String(32), index=True)
    purpose: Mapped[str] = mapped_column(String(80), default="stolen_vehicle")
    priority: Mapped[str] = mapped_column(String(16), default="high")
    authority: Mapped[str] = mapped_column(String(80), default="demo-synthetic")
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    notes: Mapped[str] = mapped_column(Text, default="Synthetic representative record. Not a real GJ plate.")


class Sighting(Base):
    __tablename__ = "sightings"
    __table_args__ = (
        Index("ix_sightings_camera_time", "camera_id", "source_time"),
        UniqueConstraint("vendor_event_id", name="uq_sightings_vendor_event"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id"), index=True)
    passage_id: Mapped[str] = mapped_column(String(64), index=True)
    source_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    ingest_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    source_pts_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    plate_raw: Mapped[str] = mapped_column(String(64))
    plate_norm: Mapped[str] = mapped_column(String(32), index=True)
    plate_voted: Mapped[str] = mapped_column(String(32), default="")
    syntax_ok: Mapped[bool] = mapped_column(Boolean, default=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    model_id: Mapped[str] = mapped_column(String(80), default="ollama-vision-p0")
    model_hash: Mapped[str] = mapped_column(String(64), default="unpinned")
    evidence_path: Mapped[str] = mapped_column(Text, default="")
    run_id: Mapped[str] = mapped_column(String(64), default="")
    frame_index: Mapped[int] = mapped_column(Integer, default=0)
    provider: Mapped[str] = mapped_column(String(32), default="local")
    vendor_event_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    vendor_payload_hash: Mapped[str] = mapped_column(String(64), default="")
    bbox_x: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bbox_y: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bbox_w: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bbox_h: Mapped[int | None] = mapped_column(Integer, nullable=True)
    frame_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    frame_height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    vehicle_type: Mapped[str] = mapped_column(String(32), default="")
    vehicle_make: Mapped[str] = mapped_column(String(40), default="")
    vehicle_model: Mapped[str] = mapped_column(String(40), default="")
    vehicle_color: Mapped[str] = mapped_column(String(40), default="")
    vehicle_json: Mapped[dict | None] = mapped_column(JSON(none_as_null=True), nullable=True)
    vehicle_observation_id: Mapped[int | None] = mapped_column(
        ForeignKey("vehicle_observations.id"), nullable=True, index=True
    )

    camera: Mapped[Camera] = relationship(back_populates="sightings")
    vehicle_observation: Mapped["VehicleObservation | None"] = relationship(back_populates="plate_sightings")


class VehicleObservation(Base):
    """One representative vehicle record for a local camera track.

    This is intentionally not a vehicle identity or cross-camera ReID record.
    It stores the best available evidence for a single observed passage.
    """

    __tablename__ = "vehicle_observations"
    __table_args__ = (
        UniqueConstraint("camera_id", "run_id", "track_id", name="uq_vehicle_observation_track"),
        Index("ix_vehicle_observation_camera_time", "camera_id", "first_seen_at"),
        Index("ix_vehicle_observation_type_color_time", "vehicle_type", "vehicle_color", "first_seen_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id"), index=True)
    run_id: Mapped[str] = mapped_column(String(64), default="", index=True)
    track_id: Mapped[str] = mapped_column(String(96), default="", index=True)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    first_pts_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_pts_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_frame_index: Mapped[int] = mapped_column(Integer, default=0)
    bbox_x: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bbox_y: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bbox_w: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bbox_h: Mapped[int | None] = mapped_column(Integer, nullable=True)
    frame_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    frame_height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    detector: Mapped[str] = mapped_column(String(80), default="")
    detector_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    vehicle_type: Mapped[str] = mapped_column(String(32), default="unknown", index=True)
    type_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    type_source: Mapped[str] = mapped_column(String(48), default="local")
    vehicle_color: Mapped[str] = mapped_column(String(32), default="unknown", index=True)
    color_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    color_source: Mapped[str] = mapped_column(String(48), default="local")
    evidence_path: Mapped[str] = mapped_column(Text, default="")
    context_evidence_path: Mapped[str] = mapped_column(Text, default="")
    metadata_json: Mapped[dict | None] = mapped_column(JSON(none_as_null=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    camera: Mapped[Camera] = relationship(back_populates="vehicle_observations")
    plate_sightings: Mapped[list[Sighting]] = relationship(back_populates="vehicle_observation")


class Alert(Base):
    __tablename__ = "alerts"
    __table_args__ = (
        UniqueConstraint("watchlist_id", "camera_id", "passage_id", name="uq_alert_dedup"),
        Index("ix_alerts_status", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sighting_id: Mapped[int] = mapped_column(ForeignKey("sightings.id"))
    watchlist_id: Mapped[int] = mapped_column(ForeignKey("watchlist.id"))
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id"))
    passage_id: Mapped[str] = mapped_column(String(64))
    plate_norm: Mapped[str] = mapped_column(String(32), index=True)
    match_type: Mapped[str] = mapped_column(String(16), default="exact")
    status: Mapped[str] = mapped_column(String(24), default="new")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    sighting: Mapped[Sighting] = relationship()
    watchlist: Mapped[WatchlistEntry] = relationship()


class AuditEvent(Base):
    __tablename__ = "audit"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    actor: Mapped[str] = mapped_column(String(80), default="operator")
    action: Mapped[str] = mapped_column(String(80), index=True)
    detail: Mapped[str] = mapped_column(Text, default="")


class SystemState(Base):
    __tablename__ = "system_state"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class CameraActivity(Base):
    """Analytics-active window on this host. Not a VMS recording archive."""

    __tablename__ = "camera_activity"
    __table_args__ = (Index("ix_activity_camera_start", "camera_id", "started_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id"), index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    stopped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    run_id: Mapped[str] = mapped_column(String(64), default="")
    protocol: Mapped[str] = mapped_column(String(16), default="")
    reason: Mapped[str] = mapped_column(String(80), default="worker")


class RecognitionAttempt(Base):
    """One auditable plate candidate/recognition decision, including failures."""

    __tablename__ = "recognition_attempts"
    __table_args__ = (
        Index("ix_recognition_attempt_camera_time", "camera_id", "created_at"),
        Index("ix_recognition_attempt_reason", "reason_code"),
        Index("ix_recognition_attempt_track", "camera_id", "track_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id"), index=True)
    run_id: Mapped[str] = mapped_column(String(64), default="")
    track_id: Mapped[str] = mapped_column(String(64), default="", index=True)
    frame_index: Mapped[int] = mapped_column(Integer, default=0)
    source_pts_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    stage: Mapped[str] = mapped_column(String(32), default="recognition")
    reason_code: Mapped[str] = mapped_column(String(48), default="")
    detector: Mapped[str] = mapped_column(String(80), default="")
    recognizer: Mapped[str] = mapped_column(String(80), default="")
    model_id: Mapped[str] = mapped_column(String(120), default="")
    model_hash: Mapped[str] = mapped_column(String(64), default="")
    bbox_x: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bbox_y: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bbox_w: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bbox_h: Mapped[int | None] = mapped_column(Integer, nullable=True)
    native_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    native_height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    enhanced_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    enhanced_height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    quality_json: Mapped[dict | None] = mapped_column(JSON(none_as_null=True), nullable=True)
    raw_output: Mapped[str] = mapped_column(Text, default="")
    plate_norm: Mapped[str] = mapped_column(String(32), default="", index=True)
    syntax_ok: Mapped[bool] = mapped_column(Boolean, default=False)
    character_confidences: Mapped[list[float] | None] = mapped_column(JSON(none_as_null=True), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    accepted: Mapped[bool] = mapped_column(Boolean, default=False)
    evidence_path: Mapped[str] = mapped_column(Text, default="")
