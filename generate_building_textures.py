#!/usr/bin/env python3
"""
Generate seamless procedural building placeholder textures.

Usage:
    python generate_building_textures.py -o assets/buildings
    python generate_building_textures.py -o assets/buildings --overwrite

Arguments:
    -o, --output-dir  Output folder.
    --size            Texture size; default 512, power of two.
    --seed            Repeatable seed; default 12345.
    --overwrite       Allow replacement of generated files.

Outputs:
    textures/*.png    For GLB/material preparation.
    textures/*.tga    For the legacy OpenGL viewer.
    materials.json   Texture paths and real-world repeat dimensions.

These are visual placeholders, not accurate building-specific façades.
Requires Pillow and NumPy.
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# ============================================================================
# EDITABLE MATERIAL SETTINGS
# ============================================================================

CONFIG = {
    "wall_default": {
        "kind": "window_wall",
        "base_color": [209, 204, 190],
        "repeat_width_metres": 3.0,
        "repeat_height_metres": 3.0,

        # Window dimensions within a 3 x 3 m repeating façade panel.
        "window_width_metres": 1.35,
        "window_height_metres": 1.45,
        "window_bottom_metres": 0.75,
        
        "ventilation_ratio": 0.15,
        "ventilation_side": "left",
    },

    "wall_plain": {
        "kind": "plain_wall",
        "base_color": [209, 204, 190],
        "repeat_width_metres": 3.0,
        "repeat_height_metres": 3.0,
    },

    "roof_default": {
        "kind": "metal_roof",
        "base_color": [92, 96, 95],
        "repeat_width_metres": 2.0,
        "repeat_height_metres": 2.0,
        "seam_spacing_metres": 0.50,
    },

    "roof_red_tile": {
        "kind": "tile_roof",
        "base_color": [150, 77, 53],
        "repeat_width_metres": 2.0,
        "repeat_height_metres": 2.0,
        "columns": 8,
        "rows": 6,
    },
}


def noisy_background(size, color, rng, strength=2.0):
    """
    Create restrained surface noise.

    Independent per-pixel noise has no prominent large-scale boundary,
    so repetition is less obvious than with a nonseamless photograph.
    """
    noise = rng.normal(0, strength, (size, size, 1))
    rgb = np.asarray(color, dtype=float)[None, None, :] + noise
    return Image.fromarray(
        np.clip(rgb, 0, 255).astype(np.uint8)
    ).convert("RGBA")


def window_wall(size, config, rng):
    """Draw one repeatable wall panel with a central window."""
    image = noisy_background(
        size, config["base_color"], rng
    )
    draw = ImageDraw.Draw(image)

    width_m = config["repeat_width_metres"]
    height_m = config["repeat_height_metres"]

    window_width = config["window_width_metres"] / width_m * size
    window_height = config["window_height_metres"] / height_m * size
    bottom = config["window_bottom_metres"] / height_m * size

    left = (size - window_width) / 2
    right = left + window_width
    lower = size - bottom
    upper = lower - window_height

    frame = max(2, round(size * 0.016))

    # Slight shadow and sill give basic depth cues.
    draw.rectangle(
        [left - frame, upper - frame, right + frame, lower + frame],
        fill=(116, 113, 105, 255),
    )

    draw.rectangle(
        [left, upper, right, lower],
        fill=(225, 224, 211, 255),
    )

    glass = [
        left + frame,
        upper + frame,
        right - frame,
        lower - frame,
    ]
    draw.rectangle(glass, fill=(69, 91, 103, 255))

    # A muted reflection gradient—not a strong photographic reflection.
    glass_top = int(upper + frame)
    glass_bottom = int(lower - frame)
    for y in range(glass_top, glass_bottom):
        fraction = (y - glass_top) / max(1, glass_bottom - glass_top)
        color = (
            round(90 - 28 * fraction),
            round(113 - 30 * fraction),
            round(124 - 28 * fraction),
            255,
        )
        draw.line(
            [(left + frame, y), (right - frame, y)],
            fill=color,
        )

    # Narrow ventilation pane on the left and a larger main pane.
    # Measure the ratio within the glass area, excluding the outer frame.
    ventilation_ratio = config.get("ventilation_ratio", 0.10)
    ventilation_side = config.get("ventilation_side", "left")

    glass_left = left + frame
    glass_right = right - frame
    glass_width = glass_right - glass_left

    if ventilation_side == "right":
        divider_x = glass_right - glass_width * ventilation_ratio
    else:
        divider_x = glass_left + glass_width * ventilation_ratio

    draw.rectangle(
        [
            divider_x - frame / 2,
            upper + frame,
            divider_x + frame / 2,
            lower - frame,
        ],
        fill=(225, 224, 211, 255),
    )


    # Window sill.
    draw.rectangle(
        [
            left - frame,
            lower,
            right + frame,
            lower + frame * 1.5,
        ],
        fill=(178, 175, 164, 255),
    )

    return image


def metal_roof(size, config, rng):
    """Draw a repeating standing-seam metal pattern."""
    image = noisy_background(
        size, config["base_color"], rng, strength=1.8
    )
    draw = ImageDraw.Draw(image)

    repeats = max(
        1,
        round(
            config["repeat_width_metres"]
            / config["seam_spacing_metres"]
        ),
    )
    seam_width = max(1, round(size * 0.004))

    # Seams align with texture repetition at the left/right boundaries.
    for number in range(repeats):
        x = round(number * size / repeats)

        draw.line(
            [(x, 0), (x, size)],
            fill=(58, 62, 62, 255),
            width=seam_width,
        )
        draw.line(
            [(x + seam_width, 0), (x + seam_width, size)],
            fill=(126, 129, 126, 255),
            width=seam_width,
        )

    return image


def tile_roof(size, config, rng):
    """Draw a seamless approximate red roof-tile pattern."""
    image = noisy_background(
        size, config["base_color"], rng, strength=1.8
    )
    draw = ImageDraw.Draw(image)

    columns = config["columns"]
    rows = config["rows"]
    tile_width = size / columns
    tile_height = size / rows
    seam = max(1, round(size * 0.003))

    for row in range(rows):
        y = round(row * tile_height)

        draw.line(
            [(0, y), (size, y)],
            fill=(99, 48, 35, 255),
            width=seam,
        )

        # Offset alternating rows. Extra columns wrap across boundaries.
        offset = tile_width / 2 if row % 2 else 0

        for column in range(-1, columns + 1):
            x = column * tile_width + offset

            draw.line(
                [(x, y), (x, y + tile_height)],
                fill=(111, 53, 37, 255),
                width=seam,
            )

            draw.line(
                [
                    (x + tile_width * 0.22, y + seam),
                    (x + tile_width * 0.22, y + tile_height - seam),
                ],
                fill=(179, 98, 68, 255),
                width=seam,
            )

    return image


def write_tga(image, destination):
    """Write opaque uncompressed BGRA TGA with a bottom-left origin."""
    image = image.convert("RGBA")
    width, height = image.size

    header = struct.pack(
        "<BBBHHBHHHHBB",
        0, 0, 2,
        0, 0, 0,
        0, 0,
        width, height,
        32, 8,
    )

    pixels = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    with destination.open("wb") as handle:
        handle.write(header)
        handle.write(pixels.tobytes("raw", "BGRA"))


def main():
    parser = argparse.ArgumentParser(
        description="Generate procedural placeholder building textures."
    )
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if (
        args.size < 128
        or args.size > 2048
        or args.size & (args.size - 1)
    ):
        parser.error("--size must be a power of two from 128 to 2048.")

    output = Path(args.output_dir).resolve()

    if output.exists() and not output.is_dir():
        parser.error("Output path is not a folder.")

    if output.exists() and any(output.iterdir()) and not args.overwrite:
        parser.error("Output folder is not empty; use --overwrite.")

    textures = output / "textures"
    textures.mkdir(parents=True, exist_ok=True)

    materials = {}

    for number, (name, config) in enumerate(CONFIG.items()):
        rng = np.random.default_rng(args.seed + number * 101)
        kind = config["kind"]

        if kind == "window_wall":
            image = window_wall(args.size, config, rng)
        elif kind == "plain_wall":
            image = noisy_background(
                args.size, config["base_color"], rng
            )
        elif kind == "metal_roof":
            image = metal_roof(args.size, config, rng)
        elif kind == "tile_roof":
            image = tile_roof(args.size, config, rng)
        else:
            raise ValueError(f"Unknown material kind: {kind}")

        image.save(textures / f"{name}.png")
        write_tga(image, textures / f"{name}.tga")

        materials[name] = {
            "texture": f"textures/{name}.png",
            "viewer_texture": f"textures/{name}.tga",
            "repeat_width_metres": config["repeat_width_metres"],
            "repeat_height_metres": config["repeat_height_metres"],
        }

        print(f"Generated {name}")

    definition = {
        "default": {
            "wall": "wall_default",
            "roof": "roof_default",
        },
        "materials": materials,

        # Add building-specific choices here later, for example:
        # "building_000003": {
        #     "wall": "wall_plain",
        #     "roof": "roof_red_tile"
        # }
        "buildings": {},
    }

    (output / "materials.json").write_text(
        json.dumps(definition, indent=2),
        encoding="utf-8",
    )

    print(f"\nFinished: {output}")


if __name__ == "__main__":
    main()
