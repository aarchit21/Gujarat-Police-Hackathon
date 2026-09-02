from types import SimpleNamespace

from app.services.processing import select_processing_route


def cam(**kwargs):
    base = dict(
        processing_mode="deferred",
        network_class="good",
        source_type="rtsp",
        source_uri="rtsp://example/x",
        substream_uri="",
        protected_rtsp_url_or_reference="",
        hls_url="",
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_vendor_metadata_preferred_when_selected():
    route = select_processing_route(cam(processing_mode="vendor_metadata"))
    assert route["worker_kind"] == "vendor_metadata"
    assert route["analytics_may_start"] is False


def test_local_worker_when_source_present():
    route = select_processing_route(cam(processing_mode="local_worker", source_type="image_dir", source_uri="C:/frames"))
    assert route["worker_kind"] == "local_worker"


def test_remote_gpu_requires_url():
    route = select_processing_route(cam(processing_mode="remote_gpu"), remote_url="")
    assert route["worker_kind"] in {"deferred", "local_worker"}


def test_remote_gpu_when_url_configured():
    route = select_processing_route(cam(processing_mode="remote_gpu"), remote_url="https://gpu.example/infer")
    assert route["selected"] == "remote_gpu"


def test_deferred_when_offline_without_local_source():
    route = select_processing_route(
        cam(processing_mode="remote_gpu", network_class="offline", source_uri="", source_type="rtsp", hls_url=""),
        remote_url="https://gpu.example/infer",
    )
    assert route["worker_kind"] == "deferred"


def test_central_on_demand():
    route = select_processing_route(cam(processing_mode="central_on_demand"))
    assert route["selected"] == "central_on_demand"


def test_shared_regional_maps_to_local_worker():
    route = select_processing_route(cam(processing_mode="shared_regional", source_type="rtsp"))
    assert route["worker_kind"] == "local_worker"
