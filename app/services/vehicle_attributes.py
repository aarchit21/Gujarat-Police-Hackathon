"""Deterministic vehicle type + colour from the Intel OMZ attribute model.

Model: ``vehicle-attributes-recognition-barrier-0042`` (Apache-2.0, OpenVINO
Open Model Zoo), run on CPU through the OpenVINO runtime.  No LLM, no VLM, no
network call.

What is *verified* against the downloaded IR and the official model README:

* input ``input`` ``[1, 3, 72, 72]`` NCHW float32, **BGR** channel order,
  raw 0-255 range with **no mean/scale normalization**;
* output ``color`` ``[1, 7]`` and ``type`` ``[1, 4]``, both already softmaxed.

What is *interpreted* rather than verified: the accuracy-check preprocessing
chain is ``extend_around_rect 0.3 -> crop_rect -> resize 115 -> crop 72``.  The
exact extend semantics are not published with the model, so the per-side extend
factor is configurable (``vattr_extend``).  At the default 0.30 per side the
box grows 1.6x and the central 72/115 = 0.626 is kept, so the net region is
~1.0x the detector box.

Honesty constraints enforced here:

* the class orders below are the model's *output index* order.  The
  ``label_map`` in the published accuracy-check.yml is alphabetical
  (``bus,car,truck,van``) and is NOT the output order -- using it silently
  swaps ``car`` and ``bus``.  ``tests/test_vehicle_attributes.py`` pins this.
* this model has no ``suv``, ``auto_rickshaw``, ``taxi_cab``, ``silver``,
  ``brown`` or ``orange`` class.  Those are never emitted from this source.
* ``gray`` is never rewritten to ``silver``.
"""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.config import ROOT, settings

MODEL_NAME = "vehicle-attributes-recognition-barrier-0042"

# Authoritative output-index order (model README). Do not reorder.
COLOR_CLASSES: tuple[str, ...] = ("white", "gray", "yellow", "red", "green", "blue", "black")
TYPE_CLASSES: tuple[str, ...] = ("car", "van", "truck", "bus")

# Canonical vocabulary members this model can actually produce.
SUPPORTED_TYPES = frozenset(TYPE_CLASSES)
SUPPORTED_COLORS = frozenset(COLOR_CLASSES)

INPUT_SIZE = 72
RESIZE_SIZE = 115

_lock = threading.Lock()
_compiled = None
_load_error = ""
_model_hash = ""


class AttributeModelUnavailable(RuntimeError):
    """Raised when the attribute model cannot be loaded. Never swallowed silently."""


@dataclass
class AttributeResult:
    """One attribute inference over one crop. Carries full probability vectors."""

    type_probs: np.ndarray
    color_probs: np.ndarray
    model_id: str
    model_hash: str = ""

    @property
    def type_top(self) -> tuple[str, float]:
        idx = int(np.argmax(self.type_probs))
        return TYPE_CLASSES[idx], float(self.type_probs[idx])

    @property
    def color_top(self) -> tuple[str, float]:
        idx = int(np.argmax(self.color_probs))
        return COLOR_CLASSES[idx], float(self.color_probs[idx])

    def as_dict(self) -> dict:
        return {
            "type_probs": {c: round(float(p), 6) for c, p in zip(TYPE_CLASSES, self.type_probs)},
            "color_probs": {c: round(float(p), 6) for c, p in zip(COLOR_CLASSES, self.color_probs)},
            "model_id": self.model_id,
            "model_hash": self.model_hash,
        }


@dataclass
class AttributeBatchResult:
    """Per-crop results for one track, kept separate from aggregation."""

    results: list[AttributeResult] = field(default_factory=list)


def weights_dir() -> Path:
    raw = (getattr(settings, "vattr_weights", "") or "").strip()
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else ROOT / path
    return ROOT / "data" / "models" / MODEL_NAME / "FP16"


def _xml_path() -> Path:
    return weights_dir() / f"{MODEL_NAME}.xml"


def _bin_path() -> Path:
    return weights_dir() / f"{MODEL_NAME}.bin"


def missing_weight_files() -> list[str]:
    return [str(p) for p in (_xml_path(), _bin_path()) if not p.is_file()]


def _compute_model_hash() -> str:
    binary = _bin_path()
    if not binary.is_file():
        return ""
    manifest = weights_dir() / "manifest.json"
    if manifest.is_file():
        try:
            recorded = json.loads(manifest.read_text()).get("sha256", {})
            value = recorded.get(binary.name)
            if value:
                return str(value)[:16]
        except (OSError, ValueError):
            pass
    digest = hashlib.sha256()
    with binary.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def model_id() -> str:
    return f"openvino:{MODEL_NAME}"


def load_model():
    """Load and compile the IR once. Raises AttributeModelUnavailable, never returns None."""
    global _compiled, _load_error, _model_hash
    if _compiled is not None:
        return _compiled
    with _lock:
        if _compiled is not None:
            return _compiled
        missing = missing_weight_files()
        if missing:
            _load_error = (
                "vehicle attribute weights missing: "
                + ", ".join(missing)
                + ". Run: python scripts/pull_vehicle_attributes.py"
            )
            raise AttributeModelUnavailable(_load_error)
        try:
            import openvino as ov
        except ImportError as exc:  # pragma: no cover - environment dependent
            _load_error = f"openvino runtime not installed: {exc}. Run: pip install openvino"
            raise AttributeModelUnavailable(_load_error) from exc
        try:
            core = ov.Core()
            model = core.read_model(str(_xml_path()))
            _verify_signature(model)
            device = (getattr(settings, "vattr_device", "CPU") or "CPU").upper()
            _compiled = core.compile_model(model, device)
            _model_hash = _compute_model_hash()
            _load_error = ""
        except AttributeModelUnavailable:
            raise
        except Exception as exc:
            _load_error = f"failed to compile {MODEL_NAME}: {exc}"
            raise AttributeModelUnavailable(_load_error) from exc
    return _compiled


def _verify_signature(model) -> None:
    """Fail loudly if the IR is not the model this code was written against."""
    out_names: set[str] = set()
    for out in model.outputs:
        out_names |= set(out.get_names())
    if not {"color", "type"} <= out_names:
        raise AttributeModelUnavailable(
            f"{MODEL_NAME}: expected outputs 'color' and 'type', found {sorted(out_names)}"
        )
    shapes = {}
    for out in model.outputs:
        for name in out.get_names():
            shapes[name] = list(out.partial_shape.to_shape())
    if shapes.get("color") != [1, len(COLOR_CLASSES)] or shapes.get("type") != [1, len(TYPE_CLASSES)]:
        raise AttributeModelUnavailable(
            f"{MODEL_NAME}: output shapes {shapes} do not match the pinned class orders "
            f"(color={len(COLOR_CLASSES)}, type={len(TYPE_CLASSES)})"
        )


def type_deployment_gate() -> tuple[bool, str]:
    """May the automatic vehicle TYPE be exposed as an operational value?

    Fails closed. A model passes only when an operator has explicitly named it
    in ``vattr_type_validated_model_id`` after measuring it on held-out
    labelled tracks. Nothing here inspects accuracy itself -- it cannot, and a
    gate that trusted a model's own confidence would be exactly the failure
    this exists to prevent.

    The model currently shipped is deliberately NOT listed: it measured 50%
    selective precision on 28 hand-labelled cam01 tracks, and precision fell as
    the confidence floor rose, so its errors are systematic rather than noisy.
    Until a model clears the gate, the operational ``vehicle_type`` is
    ``unknown`` and the raw candidate lives in metadata for diagnostics only.
    """
    validated = (getattr(settings, "vattr_type_validated_model_id", "") or "").strip()
    if not validated:
        return False, "no vehicle-type model has been validated for deployment"
    current = model_id()
    if validated != current:
        return False, f"validated model is {validated!r}, but {current!r} is running"
    return True, f"{current} is declared validated by configuration"


def attributes_status() -> dict:
    """Status for /api/status. Never raises."""
    missing = missing_weight_files()
    try:
        import openvino  # noqa: F401

        runtime = True
    except ImportError:
        runtime = False
    return {
        "enabled": bool(getattr(settings, "vattr_enabled", True)),
        "model": MODEL_NAME,
        "model_id": model_id(),
        "runtime_available": runtime,
        "weights_present": not missing,
        "missing_files": missing,
        "loaded": _compiled is not None,
        "device": (getattr(settings, "vattr_device", "CPU") or "CPU").upper(),
        "model_hash": _model_hash or (_compute_model_hash() if not missing else ""),
        "supported_types": list(TYPE_CLASSES),
        "supported_colors": list(COLOR_CLASSES),
        "unsupported_types": ["suv", "auto_rickshaw", "taxi_cab"],
        "unsupported_colors": ["silver", "brown", "orange", "other"],
        "error": _load_error,
        "type_gate_passed": type_deployment_gate()[0],
        "type_gate_reason": type_deployment_gate()[1],
        "licence": "Apache-2.0 (openvinotoolkit/open_model_zoo)",
        "accuracy_note": (
            "Vendor figures are from a barrier/toll dataset, not Indian CCTV. "
            "No accuracy has been measured for this deployment."
        ),
    }


def extend_box(
    box: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
    extend: float | None = None,
) -> tuple[int, int, int, int]:
    """Symmetrically grow an (x, y, w, h) detector box, clipped to the frame.

    This deliberately does NOT reuse ``yolo_detect.expand_vehicle_box``: that one
    pads the bottom by 0.45 so bumper plates survive, which would push the body
    off-centre for a classifier trained on centred vehicles.
    """
    if extend is None:
        extend = float(getattr(settings, "vattr_extend", 0.30) or 0.0)
    height, width = int(frame_shape[0]), int(frame_shape[1])
    x, y, w, h = (int(v) for v in box)
    dx, dy = w * extend, h * extend
    x1 = max(0, int(round(x - dx)))
    y1 = max(0, int(round(y - dy)))
    x2 = min(width, int(round(x + w + dx)))
    y2 = min(height, int(round(y + h + dy)))
    if x2 <= x1 or y2 <= y1:
        return max(0, x), max(0, y), min(width, x + w), min(height, y + h)
    return x1, y1, x2, y2


def preprocess(crop_bgr: np.ndarray) -> np.ndarray:
    """BGR crop -> [1, 3, 72, 72] float32, OMZ-faithful, no mean/scale.

    Channel order is preserved as BGR because the model expects BGR. Pixel
    values stay in the raw 0-255 range.
    """
    if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
        raise ValueError("empty crop passed to vehicle attribute preprocess")
    image = crop_bgr
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    resized = cv2.resize(image, (RESIZE_SIZE, RESIZE_SIZE), interpolation=cv2.INTER_LINEAR)
    offset = (RESIZE_SIZE - INPUT_SIZE) // 2
    cropped = resized[offset : offset + INPUT_SIZE, offset : offset + INPUT_SIZE]
    tensor = cropped.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
    return np.ascontiguousarray(tensor)


def classify_crop(crop_bgr: np.ndarray, *, infer_fn=None) -> AttributeResult:
    """Run the attribute model on one already-cropped vehicle body."""
    tensor = preprocess(crop_bgr)
    if infer_fn is None:
        compiled = load_model()

        def infer_fn(batch):  # noqa: ANN001
            with _lock:
                return compiled(batch)

    raw = infer_fn(tensor)
    type_probs, color_probs = _extract(raw)
    return AttributeResult(
        type_probs=type_probs,
        color_probs=color_probs,
        model_id=model_id(),
        model_hash=_model_hash,
    )


def _extract(raw) -> tuple[np.ndarray, np.ndarray]:
    """Pull the 'type' and 'color' vectors out of an OpenVINO result mapping."""
    found: dict[str, np.ndarray] = {}
    try:
        items = raw.items()
    except AttributeError as exc:
        raise AttributeModelUnavailable(f"unexpected inference result type {type(raw)!r}") from exc
    for key, value in items:
        names = set(getattr(key, "get_names", lambda: set())()) if not isinstance(key, str) else {key}
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        for name in names:
            if name in {"type", "color"}:
                found[name] = arr
    if "type" not in found or "color" not in found:
        raise AttributeModelUnavailable(
            f"inference result missing 'type'/'color' outputs; got {sorted(found)}"
        )
    if found["type"].size != len(TYPE_CLASSES) or found["color"].size != len(COLOR_CLASSES):
        raise AttributeModelUnavailable(
            f"output sizes type={found['type'].size} color={found['color'].size} "
            f"do not match pinned class orders"
        )
    return found["type"], found["color"]
