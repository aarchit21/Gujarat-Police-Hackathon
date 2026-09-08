import numpy as np

from app.config import settings
from app.services.cloud_verifier_queue import CloudJob, CloudVerifierQueue, _failure_reason


class _NoopThread:
    def __init__(self, **_kwargs):
        self.started = False

    def is_alive(self):
        return self.started

    def start(self):
        self.started = True


def _job(frame, priority):
    return CloudJob(
        kind="vehicle", camera_id="cam", run_id="run", track_id=f"t{frame}",
        frame_index=frame, source_pts_ms=float(frame * 100),
        image=np.zeros((20, 60, 3), np.uint8), box=(0, 0, 60, 20),
        frame_shape=(100, 200), priority=priority,
    )


def test_cloud_queue_is_bounded_and_keeps_best_candidates(monkeypatch):
    monkeypatch.setattr(settings, "cloud_verifier_queue_enabled", True)
    monkeypatch.setattr(settings, "ollama_vision_enabled", True)
    monkeypatch.setattr(settings, "cloud_verifier_queue_size", 2)
    monkeypatch.setattr("app.services.cloud_verifier_queue.threading.Thread", _NoopThread)
    queue = CloudVerifierQueue()
    assert queue.enqueue(_job(1, 0.1)) is True
    assert queue.enqueue(_job(2, 0.9)) is True
    assert queue.enqueue(_job(3, 0.5)) is True
    assert queue.status()["depth"] == 2
    assert queue.status()["dropped"] == 1
    assert [job.priority for job in queue._jobs] == [0.9, 0.5]


def test_cloud_failure_reasons_are_specific():
    assert _failure_reason(RuntimeError("ollama vision timeout")) == "timeout"
    assert _failure_reason(RuntimeError("HTTP 401 check API_KEY")) == "authentication_failed"
    assert _failure_reason(RuntimeError("model unavailable 404")) == "model_unavailable"
    assert _failure_reason(RuntimeError("socket failed")) == "cloud_error"
