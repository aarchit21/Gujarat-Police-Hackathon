from pathlib import Path
from urllib.parse import urlparse

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore", populate_by_name=True)

    app_name: str = "Gujarat CCTV Hybrid P0"
    architecture: str = "Customised Model 5 Hybrid (Model 1 registry+GIS + Model 2 first feed-to-alert)"

    # ---- Production / developer split -----------------------------------
    # The dashboard at "/" is the production investigation UI. Every developer
    # surface -- raw JSON, probability vectors, model ids and hashes, decode and
    # FPS panels, detector/OCR diagnostics, threshold controls, the worker
    # manager and the capacity estimator -- lives behind these two flags and is
    # OFF by default. Nothing is deleted: the console still exists at /dev and
    # the diagnostic APIs still exist, they are simply not served in production.
    #
    # `app_env` is descriptive (shown in the UI footer); `enable_developer_ui`
    # is the switch that actually gates routes. Turning the switch on requires
    # a non-production app_env as well, so a production deployment cannot
    # expose the console by setting one variable alone.
    app_env: str = "production"
    enable_developer_ui: bool = False
    # Set on the publicly hosted instance. It changes nothing about behaviour --
    # it makes the console say what it is, so a reviewer looking at generated
    # footage is told so rather than left to work it out. The nav rail already
    # promises "full video stays in departmental stores"; on a public URL that
    # sentence needs the accompanying one.
    demo_instance: bool = False

    def is_production(self) -> bool:
        return (self.app_env or "production").strip().lower() in {"production", "prod"}

    def developer_ui_enabled(self) -> bool:
        return bool(self.enable_developer_ui) and not self.is_production()

    database_url: str = f"sqlite:///{(ROOT / 'data' / 'cctv.db').as_posix()}"
    # 0 means "size the pool from the worker count" -- see effective_db_pool_size().
    # Every camera worker thread holds its own Session for the life of the run, so
    # a pool smaller than the worker count makes workers block on each other
    # rather than on the cameras.
    db_pool_size: int = 0
    db_max_overflow: int = 0
    db_busy_timeout_ms: int = 10_000  # SQLite fallback only.

    evidence_dir: Path = ROOT / "data" / "evidence"
    frames_dir: Path = ROOT / "data" / "frames"
    tesseract_cmd: str = ""
    admin_token: str = "p0-operator"
    vendor_ingest_token: str = "p0-vendor"
    # `require_auth` used to live here. It made `require_operator` accept a
    # request carrying NO token at all, which turned every operator route
    # anonymous from one environment variable. A switch whose only function is to
    # disable authentication is not worth the convenience it buys, so it is gone
    # rather than merely defaulted safe. Unknown keys in .env are ignored
    # (`extra="ignore"`), so an existing REQUIRE_AUTH line is harmless.

    # Hypothesis default only. Per-camera target_analysis_fps overrides. Do not treat as measured capacity.
    analysis_fps: float = 2.0
    min_plate_width_px: int = 24
    min_vision_box_px: int = 24
    cpu_anpr_enabled: bool = True
    # Master safety default for the optional plate-recognition branch.  The
    # persisted runtime value in system_state takes precedence after first use.
    plate_recognition_enabled: bool = False
    cpu_anpr_models_ready: bool = False
    cpu_anpr_tesseract_enabled: bool = True
    cpu_anpr_secondary_tesseract: bool = True
    cpu_anpr_secondary_trigger_confidence: float = 0.98
    cpu_anpr_detector_model: str = "yolo-v9-t-384-license-plate-end2end"
    cpu_anpr_ocr_model: str = "cct-s-v2-global-model"
    cpu_anpr_min_plate_width_px: int = 48
    cpu_anpr_min_plate_height_px: int = 14
    # A local plate detector must identify a plate with this confidence before a
    # live crop can be OCRed or sent to the optional cloud verifier.
    cpu_anpr_min_detector_confidence: float = 0.35
    cpu_anpr_workers: int = 1
    # auto: CUDA ONNX if the runtime has it, else CPU. Laptop P0 stays CPU.
    cpu_anpr_device: str = "auto"
    cpu_anpr_ocr_min_width: int = 400
    plate_track_iou_threshold: float = 0.25
    plate_track_max_gap_seconds: float = 1.5
    plate_track_best_crops: int = 5
    active_track_fps_multiplier: float = 2.0
    active_track_max_fps: float = 6.0
    active_track_hold_seconds: float = 1.5
    plate_confirmation_min_frames: int = 2
    plate_confirmation_min_gap_ms: float = 100.0
    plate_confirmation_median_confidence: float = 0.75
    plate_confirmation_min_char_confidence: float = 0.55
    recognition_attempt_retention_per_track: int = 20
    passage_gap_ms: float = 3000.0
    pts_jump_reset_ms: float = 5000.0
    inference_max_width: int = 1280
    own_feed_synthetic_frame_interval_ms: float = 100.0

    host: str = "127.0.0.1"
    port: int = 8000

    ingest_catalogue_url: str = "https://cctv.corp8.cloud/cameras.json"
    catalogue_sync_timeout_seconds: float = 15.0
    cctv_auth_mode: str = "none"
    cctv_access_username: str = ""
    cctv_access_token: str = ""
    cctv_auth_header_name: str = ""
    cctv_login_url: str = ""
    rtsp_transport: str = "tcp"

    remote_inference_url: str = ""
    remote_inference_token: str = ""
    remote_inference_timeout_seconds: float = 8.0
    remote_inference_allowed_hosts: str = ""
    remote_fallback_local: bool = True

    ollama_url: str = "https://ollama.com"
    ollama_api_key: str = ""
    ollama_vision_model: str = "gemma4:31b"
    # Off by default. A VLM must never infer vehicle type, colour or plate text:
    # those come from vehicle_attributes.py (OpenVINO) and cpu_anpr.py.
    ollama_vision_enabled: bool = False
    ollama_vision_on_own_feed: bool = True
    ollama_vision_timeout_seconds: float = 90.0
    ollama_vision_max_width: int = 768
    ollama_live_interval_seconds: float = 0.4
    ollama_lock_wait_seconds: float = 25.0
    cloud_verifier_queue_enabled: bool = True
    cloud_verifier_queue_size: int = 8
    cloud_verifier_max_age_seconds: float = 12.0
    vision_enhancement_enabled: bool = True
    # LPDGAN is generative and off until trained LPBlur weights are on disk.
    lpdgan_enabled: bool = False
    lpdgan_weights: str = "data/models/lpdgan_generator.pth"
    lpdgan_device: str = "auto"
    vision_only_enabled: bool = False
    vision_only_interval_seconds: float = 8.0
    vision_only_models: str = "gemma4:31b"

    yolo_enabled: bool = True
    yolo_weights: str = "yolov8n.pt"
    yolo_device: str = "auto"
    yolo_conf: float = 0.35
    yolo_max_crops: int = 2
    vehicle_crop_pad_x: float = 0.20
    vehicle_crop_pad_top: float = 0.15
    vehicle_crop_pad_bottom: float = 0.45
    vehicle_max_detections: int = 8
    lpdnet_enabled: bool = True
    lpdnet_weights_dir: str = "data/models/lpdnet"
    lpdnet_onnx: str = "LPDNet_usa_pruned_tao5.onnx"
    lpdnet_conf: float = 0.30
    lpdnet_device: str = "auto"
    yolo_iou: float = 0.45
    yolo_imgsz: int = 640
    yolo_half: bool = True  # FP16 on CUDA only; ignored on CPU.
    # Retired: vehicle attributes came from an Ollama VLM. Replaced by the
    # deterministic OpenVINO path below. Left at False so nothing re-enables it.
    vehicle_attribute_refinement_enabled: bool = False
    vehicle_attribute_model: str = "gemma4:31b"
    vehicle_attribute_queue_size: int = 16
    vehicle_attribute_max_age_seconds: float = 20.0
    vehicle_attribute_min_confidence: float = 0.65

    # ---- Deterministic vehicle attributes (OpenVINO OMZ vabr-0042) --------
    # Supported types:  car, van, truck, bus  (+ two_wheeler from COCO id 3)
    # Supported colours: white, gray, yellow, red, green, blue, black
    # suv / auto_rickshaw / taxi_cab / silver / brown / orange are NOT supported
    # by any weight on disk and are never emitted.
    #
    # Every threshold below is a conservative starting value chosen from the
    # vendor's per-class accuracies. NONE of them are calibrated against
    # labelled data for this deployment.
    vattr_enabled: bool = True
    vattr_weights: str = "data/models/vehicle-attributes-recognition-barrier-0042/FP16"
    vattr_device: str = "CPU"  # OpenVINO device; keeps the NVIDIA GPU for YOLO.
    vattr_extend: float = 0.30  # symmetric box growth, per side, before 115->72.
    vattr_best_crops_per_track: int = 3
    vattr_min_observations: int = 2
    vattr_type_min_prob: float = 0.60
    vattr_type_min_margin: float = 0.15
    vattr_type_min_agreement: float = 0.60
    # Detector/classifier conflict on bus/truck (classes COCO handles well).
    # "detector"   -> keep the COCO class (default).
    # "abstain"    -> unknown.
    # "classifier" -> let the classifier win above vattr_type_conflict_min_prob.
    #
    # Measured on 20 hand-labelled supported-class tracks from cam01 (night):
    #   policy       coverage  selective precision
    #   detector       90.0%        50.0%
    #   abstain        55.0%        36.4%
    #   classifier     85.0%        29.4%
    # `detector` dominates on both axes, because COCO YOLO genuinely separates
    # bus from truck while this barrier-trained model calls Indian city buses
    # `truck` at 0.99. Abstaining discarded the cases the detector got right.
    #
    # WARNING: 50% is still not usable. Vehicle TYPE from this pretrained model
    # must not be relied on for this camera at any threshold -- raising the
    # probability floor makes precision WORSE (50% -> 30% at 0.95), because the
    # errors are confident and systematic, not noisy. Fine-tuning is required.
    # Colour is the usable signal at ~83%. See README "Measured accuracy".
    vattr_conflict_policy: str = "detector"
    vattr_type_conflict_min_prob: float = 0.75
    vattr_color_min_prob: float = 0.55
    vattr_color_min_margin: float = 0.12
    vattr_color_min_agreement: float = 0.60
    # A barrier-trained car model is out of distribution on two-wheelers.
    vattr_two_wheeler_color_enabled: bool = False
    # Crop-quality gate.
    vattr_min_crop_px: int = 40
    vattr_min_sharpness: float = 25.0
    vattr_sharpness_reference: float = 300.0
    vattr_min_visible_fraction: float = 0.70
    vattr_max_dark_fraction: float = 0.60
    vattr_max_clipped_fraction: float = 0.35
    vattr_min_detector_confidence: float = 0.30
    # Reject a crop when this much of it belongs to a DIFFERENT vehicle.
    # Measured on cam01 night traffic: 243 of 276 crops came from multi-vehicle
    # frames, 20% held >25% of a neighbour and 5% held more neighbour than
    # subject. Feeding those to the classifier is what let one vehicle's type
    # sit beside another's colour.
    vattr_max_foreign_fraction: float = 0.25
    # Tracking.
    vattr_tracker: str = "bytetrack.yaml"
    vattr_force_iou_tracker: bool = False
    vattr_track_high_conf: float = 0.50
    vattr_track_low_iou: float = 0.20
    vattr_track_timeout_seconds: float = 3.0
    # ANPR stays optional and must never gate a vehicle observation.
    vattr_anpr_enabled: bool = False

    # ---- Deployment gate for automatic vehicle TYPE ----------------------
    # The operational vehicle_type is forced to "unknown" unless the running
    # attribute model is named here, or a human has verified the record.
    # Empty means no model is validated: that is the correct state today.
    #
    # Do NOT populate this with the current OMZ model. It measured 50%
    # selective precision at 64% coverage on 28 hand-labelled cam01 tracks,
    # confuses Indian buses for trucks, and gets *less* precise as the
    # confidence floor rises. The `detector` conflict policy scored best of
    # three options, but winning a three-way comparison at 50% precision does
    # not make it a production classifier.
    vattr_type_validated_model_id: str = ""
    # Precision a model must demonstrate on held-out labelled tracks before an
    # operator sets the id above. Recorded for documentation; not self-checked.
    vattr_type_required_precision: float = 0.95
    # Colour stays enabled but is always presented as an estimate that needs
    # review. Measured: 83% selective precision at 71% coverage.
    vattr_color_enabled: bool = True
    # Hunt capture slots rotate across the catalogue. Pin holds slots without rotating.
    #
    # How cameras are chosen for those slots:
    #   least_recent  never-hunted first, then oldest last_hunted_at, random
    #                 tiebreak. Rotates fairly, so repeated clicks reach the
    #                 whole catalogue instead of re-pinning the same few.
    #   random        shuffle within each decode tier.
    #   fixed         legacy: PIN_DEFAULT first, then alphabetical -- this is
    #                 what made every hunt start at cam01-cam05 and never get
    #                 past cam06.
    # Decode tiers (ok -> untested -> failed) are always honoured first, so a
    # known-dead camera never takes a slot from a working one.
    hunt_rotation: str = "least_recent"
    hunt_pin_seed: int = 0  # non-zero makes selection reproducible, for tests
    # Share of capture slots reserved for UNTESTED cameras. Without this,
    # decode-ok cameras take every slot, untested ones are never opened, and so
    # they stay untested -- on this catalogue 25 of 32 cameras were stuck that
    # way while the same 6 were pinned every time. 0.0 restores that behaviour.
    hunt_explore_fraction: float = 0.5
    hunt_dwell_seconds: float = 28.0
    hunt_max_frames: int = 40
    rtsp_open_wait_seconds: float = 6.0
    # Sequential decode probe. Independent of max concurrent workers — do not walk 30 RTSP URLs per click.
    measure_batch_size: int = 4
    measure_probe_timeout_seconds: float = 6.0

    vendor_max_payload_bytes: int = 64_000
    max_upload_bytes: int = 8_000_000

    # Concurrent live camera workers. Raised from 4 to cover a whole catalogue at
    # once: the old value was chosen because SQLite's write lock could not survive
    # more, not because the host could not decode more. Decoding is the real
    # ceiling now -- each RTSP worker decodes every frame even though analytics
    # samples a subset -- so measure on the host before raising further.
    max_concurrent_workers: int = Field(
        default=32,
        validation_alias=AliasChoices("MAX_CONCURRENT_WORKERS", "max_concurrent_workers"),
    )
    max_open_captures: int = Field(
        default=32,
        validation_alias=AliasChoices("MAX_CONCURRENT_CAPTURES", "MAX_OPEN_CAPTURES", "max_open_captures"),
    )
    reconnect_start_seconds: float = Field(
        default=2.0,
        validation_alias=AliasChoices("RTSP_RECONNECT_INITIAL_SECONDS", "RECONNECT_START_SECONDS", "reconnect_start_seconds"),
    )
    reconnect_max_seconds: float = Field(
        default=30.0,
        validation_alias=AliasChoices("RTSP_RECONNECT_MAX_SECONDS", "RECONNECT_MAX_SECONDS", "reconnect_max_seconds"),
    )
    keyframe_wait_seconds: float = 8.0
    live_analyze_max_frames: int = 24
    live_analyze_max_seconds: float = 20.0
    # Snap-to-road. Default public OSRM Match — no key, no credit card. Google Directions is not used.
    map_match_provider: str = "osrm"
    osrm_match_url: str = "http://router.project-osrm.org"
    map_match_radius_m: float = 25.0
    mapbox_access_token: str = ""
    geoapify_api_key: str = ""
    google_maps_api_key: str = ""
    demo_autostart_workers: bool = False
    demo_decode_ok_only: bool = True

    def catalogue_host(self) -> str:
        return (urlparse(self.ingest_catalogue_url).hostname or "").lower()

    def catalogue_origin(self) -> str:
        parsed = urlparse(self.ingest_catalogue_url)
        if not parsed.scheme or not parsed.netloc:
            return ""
        return f"{parsed.scheme}://{parsed.netloc}"

    def effective_db_pool_size(self) -> int:
        """Enough pooled connections for every worker plus API traffic.

        Each live camera worker holds one Session for the duration of its run.
        With the old fixed pool of 5 (+10 overflow), raising the worker count
        would have made workers queue for connections instead of reading frames.
        An explicit DB_POOL_SIZE still wins.
        """
        if self.db_pool_size and self.db_pool_size > 0:
            return int(self.db_pool_size)
        return max(10, int(self.max_concurrent_workers) + 12)

    def effective_db_max_overflow(self) -> int:
        if self.db_max_overflow and self.db_max_overflow > 0:
            return int(self.db_max_overflow)
        return max(10, int(self.max_concurrent_workers) // 2)

    def database_kind(self) -> str:
        scheme = (urlparse(self.database_url).scheme or "").lower()
        if scheme.startswith("postgres"):
            return "postgresql"
        if scheme.startswith("sqlite"):
            return "sqlite"
        return scheme or "unknown"

    def remote_allowed_hosts(self) -> set[str]:
        hosts: set[str] = set()
        if self.remote_inference_url:
            host = urlparse(self.remote_inference_url).hostname
            if host:
                hosts.add(host.lower())
        for part in self.remote_inference_allowed_hosts.split(","):
            item = part.strip().lower()
            if item:
                hosts.add(item)
        return hosts


settings = Settings()
settings.evidence_dir.mkdir(parents=True, exist_ok=True)
settings.frames_dir.mkdir(parents=True, exist_ok=True)
(ROOT / "data").mkdir(parents=True, exist_ok=True)
