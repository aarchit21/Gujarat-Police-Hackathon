import httpx
import pytest

from app.security import assert_http_url_allowed
from app.services.remote import RemoteInferenceError, infer_jpeg


def test_remote_success_with_mock():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("authorization") == "Bearer secret"
        assert request.headers.get("content-type") == "image/jpeg"
        return httpx.Response(
            200,
            json={
                "plate_text": "GJ01AB1234",
                "confidence": 0.88,
                "model_id": "unit-ocr",
                "model_hash": "hash1",
                "bbox": [1, 2, 3, 4],
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    read = infer_jpeg(
        b"jpeg-bytes",
        camera_id="CAM-1",
        url="https://gpu.example/infer",
        token="secret",
        allowed_hosts={"gpu.example"},
        client=client,
    )
    assert read.plate_raw == "GJ01AB1234"
    assert read.model_id == "unit-ocr"
    assert read.box == (1, 2, 3, 4)


def test_remote_timeout():
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("slow")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(RemoteInferenceError):
        infer_jpeg(
            b"x",
            camera_id="CAM-1",
            url="https://gpu.example/infer",
            allowed_hosts={"gpu.example"},
            client=client,
            timeout_seconds=0.01,
        )


def test_remote_allowlist_enforced():
    with pytest.raises(RemoteInferenceError):
        infer_jpeg(
            b"x",
            camera_id="CAM-1",
            url="https://evil.example/infer",
            allowed_hosts={"gpu.example"},
        )


def test_assert_http_url_allowed_rejects_file():
    with pytest.raises(ValueError):
        assert_http_url_allowed("file:///etc/passwd", {"gpu.example"})
