from pathlib import Path
from urllib.parse import urlparse

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore", populate_by_name=True)

    app_name: str = "Gujarat CCTV Hybrid P0"
    architecture: str = "Customised Model 5 Hybrid (Model 1 registry+GIS + Model 2 first feed-to-alert)"

    database_url: str = f"sqlite:///{(ROOT / 'data' / 'cctv.db').as_posix()}"
    db_pool_size: int = 5
    db_max_overflow: int = 10

    evidence_dir: Path = ROOT / "data" / "evidence"
    frames_dir: Path = ROOT / "data" / "frames"
    tesseract_cmd: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    admin_token: str = "p0-operator"
    vendor_ingest_token: str = "p0-vendor"
    require_auth: bool = True

    # Hypothesis default only. Per-camera target_analysis_fps overrides. Do not treat as measured capacity.
    analysis_fps: float = 2.0
    min_plate_width_px: int = 80
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

    vendor_max_payload_bytes: int = 64_000
    max_upload_bytes: int = 8_000_000

    max_concurrent_workers: int = 4
    max_open_captures: int = Field(
        default=4,
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

    def catalogue_host(self) -> str:
        return (urlparse(self.ingest_catalogue_url).hostname or "").lower()

    def catalogue_origin(self) -> str:
        parsed = urlparse(self.ingest_catalogue_url)
        if not parsed.scheme or not parsed.netloc:
            return ""
        return f"{parsed.scheme}://{parsed.netloc}"

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
