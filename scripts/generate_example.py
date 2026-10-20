#!/usr/bin/env python3
"""Generate a small corresponding pair of explicitly synthetic geometric images.

These patterns demonstrate software operations. They are not micrographs,
mineralogical reference data, a scientific benchmark, or an acquisition model.
Only NumPy and Pillow from the application's requirements are needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin


WIDTH = 320
HEIGHT = 256
SHIFT_X = 4
SHIFT_Y = -3
BACKGROUND = (229, 235, 240)
TONE_SCALE_PERCENT = (90, 94, 88)
TONE_OFFSET = (16, 10, 21)


def make_fields() -> tuple[np.ndarray, np.ndarray]:
    """Return RGB arrays using integer geometry and no random sampling."""
    y, x = np.indices((HEIGHT, WIDTH), dtype=np.int64)
    field = np.empty((HEIGHT, WIDTH, 3), dtype=np.int64)
    field[:] = BACKGROUND

    rectangle = ((x >= WIDTH // 8) & (x < WIDTH * 2 // 5)
                 & (y >= HEIGHT // 6) & (y < HEIGHT * 9 // 20))
    field[rectangle] = (75, 135, 180)

    ellipse_x = x - WIDTH * 7 // 10
    ellipse_y = y - HEIGHT // 3
    axis_x, axis_y = WIDTH // 6, HEIGHT // 7
    ellipse = (ellipse_x**2 * axis_y**2 + ellipse_y**2 * axis_x**2
               <= axis_x**2 * axis_y**2)
    field[ellipse] = (195, 146, 65)

    circle = ((x - WIDTH * 7 // 25)**2 + (y - HEIGHT * 3 // 4)**2
              <= (HEIGHT // 8)**2)
    field[circle] = (165, 79, 135)

    stripe = ((x >= WIDTH // 2) & (x < WIDTH * 9 // 10)
              & (np.abs(5 * y - x - HEIGHT * 3) <= HEIGHT // 3))
    field[stripe] = (65, 157, 116)

    # Asymmetric geometric markers make the imposed translation identifiable.
    margin_x, margin_y = WIDTH // 20, HEIGHT // 20
    marker = (((x >= margin_x) & (x < margin_x + 3)
               & (y >= margin_y) & (y < margin_y + 22))
              | ((x >= margin_x) & (x < margin_x + 19)
                 & (y >= margin_y + 19) & (y < margin_y + 22)))
    for bar in range(4):
        start_x = WIDTH * 4 // 5 + bar * 7
        marker |= ((x >= start_x) & (x < start_x + 3)
                   & (y >= margin_y) & (y < margin_y + 8 + 3 * bar))
    field[marker] = (33, 41, 52)

    # Low-amplitude integer patterns are visual texture, not optical simulation.
    texture = ((x + 2 * y) % 11) - 5
    field += texture[..., None]
    field_a = np.clip(field, 0, 255).astype(np.uint8)

    scale = np.array(TONE_SCALE_PERCENT, dtype=np.int64)
    offset = np.array(TONE_OFFSET, dtype=np.int64)
    toned = np.clip((field_a.astype(np.int64) * scale + 50) // 100 + offset,
                    0, 255).astype(np.uint8)
    fill = np.clip((np.array(BACKGROUND) * scale + 50) // 100 + offset,
                   0, 255).astype(np.uint8)
    field_b = np.empty_like(field_a)
    field_b[:] = fill
    # b[y-3, x+4] = tone(a[y, x]) where both positions are inside the image.
    source_x, source_y = max(0, -SHIFT_X), max(0, -SHIFT_Y)
    target_x, target_y = max(0, SHIFT_X), max(0, SHIFT_Y)
    copied_width, copied_height = WIDTH - abs(SHIFT_X), HEIGHT - abs(SHIFT_Y)
    field_b[target_y:target_y + copied_height, target_x:target_x + copied_width] = (
        toned[source_y:source_y + copied_height, source_x:source_x + copied_width]
    )
    return field_a, field_b


def write_fields(output: Path) -> dict:
    """Write the generated PNG pair and return parameters and content hashes."""
    output.mkdir(parents=True, exist_ok=True)
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Description", "Synthetic geometric example; not a micrograph.")
    metadata.add_text("Software", "RIIMA geometric example generator")
    result = {
        "description": "Synthetic geometric example; not a micrograph or benchmark.",
        "width": WIDTH,
        "height": HEIGHT,
        "channels": "RGB uint8",
        "field_b_translation_from_a_pixels": [SHIFT_X, SHIFT_Y],
        "tone_scale_percent": list(TONE_SCALE_PERCENT),
        "tone_offset": list(TONE_OFFSET),
        "random_sampling": False,
        "images": [],
    }
    for name, pixels in zip(("field_a.png", "field_b.png"), make_fields()):
        target = output / name
        Image.fromarray(pixels).save(target, format="PNG", pnginfo=metadata,
                                     optimize=False, compress_level=9)
        result["images"].append({
            "filename": name,
            "bytes": target.stat().st_size,
            "file_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "rgb_pixels_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="Destination for field_a.png and field_b.png (overwrites this pair).")
    args = parser.parse_args()
    print(json.dumps(write_fields(args.output), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
