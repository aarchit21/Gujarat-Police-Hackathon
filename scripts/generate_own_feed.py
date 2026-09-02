"""Render two short own-feed frame folders with a moving Indian plate.

No FFmpeg required. High-contrast plates so Tesseract can actually read them.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FONT = Path(r"C:\Windows\Fonts\consola.ttf")
FALLBACK = Path(r"C:\Windows\Fonts\arial.ttf")


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (FONT, FALLBACK):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _plate(text: str, width: int = 520, height: int = 140) -> Image.Image:
    img = Image.new("RGB", (width, height), (248, 236, 70))
    draw = ImageDraw.Draw(img)
    draw.rectangle((2, 2, width - 3, height - 3), outline=(10, 10, 10), width=6)
    font = _font(64)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = max(16, (width - tw) // 2)
    y = max(10, (height - th) // 2 - 8)
    draw.text((x, y), text, fill=(0, 0, 0), font=font)
    return img


def _scene(plate: Image.Image, x: int, y: int, size=(1280, 720), label: str = "") -> Image.Image:
    w, h = size
    img = Image.new("RGB", size, (55, 62, 58))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, int(h * 0.62), w, h), fill=(40, 42, 40))
    draw.rectangle((0, int(h * 0.74), w, int(h * 0.78)), fill=(210, 210, 210))
    img.paste(plate, (x, y))
    caption = _font(22)
    draw.text((24, 20), label, fill=(230, 230, 230), font=caption)
    draw.text((24, 50), "OWN-FEED · recorded/generated · not a government portal stream", fill=(180, 180, 160), font=_font(16))
    return img


def generate(n_frames: int = 24) -> dict[str, Path]:
    out_a = ROOT / "data" / "frames" / "cam-ahmedabad"
    out_b = ROOT / "data" / "frames" / "cam-surat"
    out_a.mkdir(parents=True, exist_ok=True)
    out_b.mkdir(parents=True, exist_ok=True)
    plate_a = _plate("GJ01AB1234")
    plate_b = _plate("GJ01AB1234")
    for i in range(n_frames):
        xa = 40 + i * 18
        xb = 700 - i * 16
        _scene(plate_a, xa, 290, label="CAM Home Ahmedabad · own-feed").save(out_a / f"{i:04d}.jpg", quality=95)
        _scene(plate_b, xb, 270, label="CAM RTO Surat · own-feed").save(out_b / f"{i:04d}.jpg", quality=95)
    return {"ahmedabad": out_a, "surat": out_b}


if __name__ == "__main__":
    paths = generate()
    print("wrote", paths)
