from pathlib import Path

import numpy as np

from app.services.snapshot import encode_jpeg, grab_snapshot
from tests.conftest import add_camera


def test_snapshot_from_image_dir(db, tmp_path: Path):
    frame = np.zeros((24, 40, 3), dtype=np.uint8)
    jpeg = encode_jpeg(frame)
    (tmp_path / "plate.jpg").write_bytes(jpeg)
    cam = add_camera(db, source_uri=str(tmp_path), source_type="image_dir")
    out = grab_snapshot(cam)
    assert out["ok"] is True
    assert out["source"] == "own_feed_file"
    assert out["jpeg"][:2] == b"\xff\xd8"
