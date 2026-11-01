#!/usr/bin/env python3
"""
Generate procedural road textures and material definitions.

Usage:
    python generate_road_textures.py -o assets/roads
    python generate_road_textures.py -o assets/roads --size 1024 --overwrite

Outputs:
    textures/*.png
    textures/*.tga
    materials.json

Texture strategy:
    - Base surface textures are seamless-ish and repeatable.
    - Marking materials are also produced separately.
    - Combined prototype textures are produced for simple road meshes:
        asphalt_center_dashed
        asphalt_edge_lines
        asphalt_edge_center_dashed
        asphalt_edge_double_yellow

Later, road markings can be generated as separate geometry. For now these
textures allow quick visual testing.

Requires:
    pip install pillow numpy
"""

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# ============================================================================
# EDITABLE SETTINGS
# ============================================================================

CONFIG = {
    # Main texture resolution. Must be power of two for older GPUs.
    "default_size": 512,

    # Physical repeat dimensions. The road mesh generator should use these
    # to assign UVs in metres.
    "repeat_length_metres": 12.0,
    "repeat_width_metres": 8.0,

    # Road marking dimensions in metres, converted into pixels using
    # repeat_width/repeat_length below.
    "marking": {
        "edge_line_offset_from_edge_metres": 0.35,
        "edge_line_width_metres": 0.12,

        "center_white_width_metres": 0.12,

        # Double yellow means two continuous parallel yellow lines.
        "double_yellow_line_width_metres": 0.10,
        "double_yellow_gap_metres": 0.18,

        # White dashed center marking.
        "dash_length_metres": 3.0,
        "dash_gap_metres": 6.0,

        "white": [225, 225, 215],
        "yellow": [210, 170, 42],
    },
}


MATERIALS = {
    # Base surfaces.
    "asphalt_faded": {
        "kind": "asphalt_faded",
        "repeat_length_metres": 12.0,
        "repeat_width_metres": 8.0,
    },
    "asphalt_fresh": {
        "kind": "asphalt_fresh",
        "repeat_length_metres": 12.0,
        "repeat_width_metres": 8.0,
    },
    "gravel": {
        "kind": "gravel",
        "repeat_length_metres": 8.0,
        "repeat_width_metres": 4.0,
    },
    "paved_footpath": {
        "kind": "paved_footpath",
        "repeat_length_metres": 6.0,
        "repeat_width_metres": 3.0,
    },

    # Simple marking-only materials for later geometry-based markings.
    "marking_white": {
        "kind": "solid_color",
        "color": [225, 225, 215],
        "repeat_length_metres": 1.0,
        "repeat_width_metres": 1.0,
    },
    "marking_yellow": {
        "kind": "solid_color",
        "color": [210, 170, 42],
        "repeat_length_metres": 1.0,
        "repeat_width_metres": 1.0,
    },

    # Combined prototype road materials.
    "asphalt_center_dashed": {
        "kind": "asphalt_faded",
        "markings": ["center_dashed_white"],
        "repeat_length_metres": 12.0,
        "repeat_width_metres": 8.0,
    },
    "asphalt_edge_lines": {
        "kind": "asphalt_faded",
        "markings": ["edge_lines_white"],
        "repeat_length_metres": 12.0,
        "repeat_width_metres": 8.0,
    },
    "asphalt_edge_center_dashed": {
        "kind": "asphalt_faded",
        "markings": ["edge_lines_white", "center_dashed_white"],
        "repeat_length_metres": 12.0,
        "repeat_width_metres": 8.0,
    },
    "asphalt_edge_double_yellow": {
        "kind": "asphalt_faded",
        "markings": ["edge_lines_white", "center_double_yellow"],
        "repeat_length_metres": 12.0,
        "repeat_width_metres": 8.0,
    },
}


# Road-class material suggestions. These are not used by this texture generator
# directly, but the upcoming road mesh generator can consume them.
ROAD_CLASS_DEFAULTS = {
    "12111": {
        "description": "large motorway or arterial road",
        "surface_material": "asphalt_edge_double_yellow",
    },
    "12112": {
        "description": "large motorway or arterial road",
        "surface_material": "asphalt_edge_double_yellow",
    },
    "12121": {
        "description": "large main road",
        "surface_material": "asphalt_edge_double_yellow",
    },
    "12122": {
        "description": "road with center marking and edge lines",
        "surface_material": "asphalt_edge_double_yellow",
    },
    "12131": {
        "description": "regular street",
        "surface_material": "asphalt_center_dashed",
    },
    "12141": {
        "description": "narrow asphalt street, no markings",
        "surface_material": "asphalt_faded",
    },
    "12313": {
        "description": "gravel path",
        "surface_material": "gravel",
    },
    "12314": {
        "description": "paved footpath",
        "surface_material": "paved_footpath",
    },
    "12316": {
        "description": "narrow gravel path",
        "surface_material": "gravel",
    },
    "fallback": {
        "description": "unknown road class",
        "surface_material": "asphalt_faded",
    },
}


# ============================================================================
# LOW-LEVEL IMAGE HELPERS
# ============================================================================

def write_tga(image, destination):
    """Write uncompressed 32-bit TGA with bottom-left origin."""
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


def tileable_noise(size, rng, strength=1.0, blur=0.0):
    """
    Create simple seamless-ish noise by generating a larger random field,
    blurring it, and wrapping opposite edges by averaging.

    This is not mathematically perfect noise, but is adequate for prototype
    road textures.
    """
    noise = rng.normal(0, 1, (size, size)).astype(np.float32)

    image = Image.fromarray(
        np.clip((noise * 32 + 128), 0, 255).astype(np.uint8)
    )

    if blur > 0:
        image = image.filter(ImageFilter.GaussianBlur(blur))

    array = np.asarray(image).astype(np.float32)
    array -= array.mean()
    array /= max(array.std(), 1e-6)
    return array * strength


def make_rgba(rgb):
    """Convert an RGB NumPy image into a PIL RGBA image."""
    alpha = np.full(rgb.shape[:2] + (1,), 255, dtype=np.uint8)
    return Image.fromarray(np.concatenate((rgb, alpha), axis=2))


def clamp_rgb(array):
    """Clamp floating-point RGB data into uint8."""
    return np.clip(array, 0, 255).astype(np.uint8)


def metres_to_x(metres, width, repeat_width):
    """Road cross-section coordinate to pixel x."""
    return metres / repeat_width * width


def metres_to_y(metres, height, repeat_length):
    """Road length coordinate to pixel y."""
    return metres / repeat_length * height


# ============================================================================
# SURFACE TEXTURES
# ============================================================================

def asphalt_faded(size, rng):
    """
    Older faded grey asphalt.

    Fine, low-contrast grain. Large-scale variation is intentionally subtle
    so the texture does not look oversized compared with road markings.
    """
    base = np.array([103, 104, 101], dtype=np.float32)

    # Fine aggregate noise.
    fine = tileable_noise(size, rng, strength=4.0, blur=0.25)

    # Very subtle broad variation only.
    broad = tileable_noise(size, rng, strength=2.0, blur=18.0)

    # Extra tiny speckle.
    micro = rng.normal(0, 2.0, (size, size)).astype(np.float32)

    rgb = (
        base[None, None, :]
        + fine[:, :, None]
        + broad[:, :, None]
        + micro[:, :, None]
    )

    # Very faint longitudinal wear streaks.
    y = np.linspace(0, 2 * math.pi, size, endpoint=False)
    streak = np.sin(y * 18 + rng.uniform(0, 2 * math.pi)) * 1.2
    rgb += streak[:, None, None]

    # Sparse small aggregate specks, restrained.
    specks = rng.random((size, size))
    rgb[specks < 0.006] += rng.uniform(4, 9)
    rgb[specks > 0.995] -= rng.uniform(4, 8)

    return make_rgba(clamp_rgb(rgb))


def asphalt_fresh(size, rng):
    """Darker, newer asphalt with fine restrained grain."""
    base = np.array([45, 47, 46], dtype=np.float32)

    fine = tileable_noise(size, rng, strength=3.5, blur=0.25)
    broad = tileable_noise(size, rng, strength=1.8, blur=20.0)
    micro = rng.normal(0, 1.6, (size, size)).astype(np.float32)

    rgb = (
        base[None, None, :]
        + fine[:, :, None]
        + broad[:, :, None]
        + micro[:, :, None]
    )

    specks = rng.random((size, size))
    rgb[specks < 0.004] += rng.uniform(3, 7)
    rgb[specks > 0.997] -= rng.uniform(3, 7)

    return make_rgba(clamp_rgb(rgb))


def gravel(size, rng):
    """
    Grey-brown gravel texture.

    Stone colors are generated along a controlled neutral grey/brown ramp
    to avoid unnatural cyan, green, or magenta artifacts.
    """
    base = np.array([132, 126, 112], dtype=np.float32)

    broad = tileable_noise(size, rng, strength=7.0, blur=8.0)
    fine = tileable_noise(size, rng, strength=8.0, blur=0.35)
    micro = rng.normal(0, 3.0, (size, size)).astype(np.float32)

    rgb = (
        base[None, None, :]
        + broad[:, :, None]
        + fine[:, :, None]
        + micro[:, :, None]
    )

    # Controlled stone palette: grey/tan/brown only.
    palette = np.array([
        [105, 102, 94],
        [122, 117, 103],
        [139, 132, 114],
        [151, 143, 121],
        [117, 111, 96],
        [160, 153, 134],
    ], dtype=np.float32)

    yy, xx = np.ogrid[:size, :size]

    # Small stones.
    stone_count = size * size // 120

    for _ in range(stone_count):
        cx = int(rng.integers(0, size))
        cy = int(rng.integers(0, size))

        # Mostly tiny stones.
        radius = int(rng.integers(1, max(2, size // 110)))

        color = palette[int(rng.integers(0, len(palette)))].copy()

        # Brightness variation only, not arbitrary RGB variation.
        color += rng.normal(0, 8)

        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2

        # Blend stone into base instead of hard replacement.
        rgb[mask] = rgb[mask] * 0.45 + color * 0.55

    return make_rgba(clamp_rgb(rgb))



def paved_footpath(size, rng):
    """Plain paved footpath material, slightly lighter than fresh asphalt."""
    base = np.array([118, 119, 114], dtype=np.float32)

    fine = tileable_noise(size, rng, strength=6.0, blur=0.5)
    broad = tileable_noise(size, rng, strength=5.0, blur=6.0)

    rgb = base[None, None, :] + fine[:, :, None] + broad[:, :, None]

    return make_rgba(clamp_rgb(rgb))
    

def solid_color(size, color):
    """Flat opaque material for future separate marking geometry."""
    image = np.zeros((size, size, 4), dtype=np.uint8)
    image[:, :, :3] = np.array(color, dtype=np.uint8)
    image[:, :, 3] = 255
    return Image.fromarray(image)


# ============================================================================
# ROAD MARKINGS
# ============================================================================

def draw_edge_lines(draw, width, height, repeat_width):
    """Draw continuous white edge lines near the left and right road edges."""
    settings = CONFIG["marking"]

    offset = settings["edge_line_offset_from_edge_metres"]
    line_width = settings["edge_line_width_metres"]

    left_x = metres_to_x(offset, width, repeat_width)
    right_x = width - left_x

    half = metres_to_x(line_width, width, repeat_width) / 2
    color = tuple(settings["white"]) + (255,)

    for x in (left_x, right_x):
        draw.rectangle(
            [x - half, 0, x + half, height],
            fill=color,
        )


def draw_center_dashed_white(draw, width, height, repeat_width, repeat_length):
    """Draw a dashed white centerline."""
    settings = CONFIG["marking"]

    line_width = settings["center_white_width_metres"]
    dash_length = settings["dash_length_metres"]
    dash_gap = settings["dash_gap_metres"]

    x = width / 2
    half = metres_to_x(line_width, width, repeat_width) / 2
    color = tuple(settings["white"]) + (255,)

    period = dash_length + dash_gap

    # Draw enough dashes to wrap cleanly at texture boundaries.
    start = -period
    end = repeat_length + period
    position = start

    while position < end:
        y0 = metres_to_y(position, height, repeat_length)
        y1 = metres_to_y(position + dash_length, height, repeat_length)

        draw.rectangle(
            [x - half, y0, x + half, y1],
            fill=color,
        )
        position += period


def draw_center_double_yellow(draw, width, height, repeat_width):
    """Draw two continuous parallel yellow center lines."""
    settings = CONFIG["marking"]

    line_width = settings["double_yellow_line_width_metres"]
    gap = settings["double_yellow_gap_metres"]

    line_width_px = metres_to_x(line_width, width, repeat_width)
    gap_px = metres_to_x(gap, width, repeat_width)

    center = width / 2
    color = tuple(settings["yellow"]) + (255,)

    # Centers of the two yellow lines.
    left_center = center - gap_px / 2 - line_width_px / 2
    right_center = center + gap_px / 2 + line_width_px / 2

    for x in (left_center, right_center):
        draw.rectangle(
            [
                x - line_width_px / 2,
                0,
                x + line_width_px / 2,
                height,
            ],
            fill=color,
        )


def apply_markings(image, material):
    """Draw markings requested by a material definition."""
    markings = material.get("markings", [])

    if not markings:
        return image

    image = image.copy()
    draw = ImageDraw.Draw(image, "RGBA")

    width, height = image.size
    repeat_width = material["repeat_width_metres"]
    repeat_length = material["repeat_length_metres"]

    for marking in markings:
        if marking == "edge_lines_white":
            draw_edge_lines(draw, width, height, repeat_width)

        elif marking == "center_dashed_white":
            draw_center_dashed_white(
                draw,
                width,
                height,
                repeat_width,
                repeat_length,
            )

        elif marking == "center_double_yellow":
            draw_center_double_yellow(
                draw,
                width,
                height,
                repeat_width,
            )

        else:
            raise ValueError(f"Unknown marking type: {marking}")

    return image


# ============================================================================
# MATERIAL GENERATION
# ============================================================================

def create_base(kind, size, rng, material):
    """Create the base texture for one material."""
    if kind == "asphalt_faded":
        return asphalt_faded(size, rng)
    if kind == "asphalt_fresh":
        return asphalt_fresh(size, rng)
    if kind == "gravel":
        return gravel(size, rng)
    if kind == "paved_footpath":
        return paved_footpath(size, rng)
    if kind == "solid_color":
        return solid_color(size, material["color"])

    raise ValueError(f"Unsupported material kind: {kind}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate procedural road textures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--size", type=int, default=CONFIG["default_size"])
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
        parser.error("Output path exists and is not a folder.")

    if output.exists() and any(output.iterdir()) and not args.overwrite:
        parser.error("Output folder is not empty. Use --overwrite.")

    textures = output / "textures"
    textures.mkdir(parents=True, exist_ok=True)

    rng_master = np.random.default_rng(args.seed)

    exported_materials = {}

    for index, (name, material) in enumerate(MATERIALS.items()):
        rng = np.random.default_rng(
            int(rng_master.integers(0, 2**31 - 1)) + index
        )

        kind = material["kind"]

        image = create_base(kind, args.size, rng, material)
        image = apply_markings(image, material)

        png_name = f"{name}.png"
        tga_name = f"{name}.tga"

        image.save(textures / png_name)
        write_tga(image, textures / tga_name)

        exported_materials[name] = {
            "texture": f"textures/{png_name}",
            "viewer_texture": f"textures/{tga_name}",
            "repeat_length_metres": material["repeat_length_metres"],
            "repeat_width_metres": material["repeat_width_metres"],
            "kind": material["kind"],
            "markings": material.get("markings", []),
        }

        print(f"Generated {name}")

    material_config = {
        "format": "road_materials_v1",
        "default": {
            "surface": "asphalt_faded",
            "marking_white": "marking_white",
            "marking_yellow": "marking_yellow",
        },
        "materials": exported_materials,

        # Intended default material per MTK tieviiva LUOKKA.
        # The future road mesh generator can use and override these.
        "class_assignments": ROAD_CLASS_DEFAULTS,

        # Optional future per-road overrides:
        # "road_000123": {
        #     "surface_material": "asphalt_fresh",
        #     "width": 7.0
        # }
        "feature_assignments": {},
        "notes": [
            "This is a prototype texture set.",
            "Combined marking textures are convenient but inflexible.",
            "Later, markings should preferably be separate geometry.",
            "White center marking is dashed.",
            "Yellow center marking is continuous double yellow.",
            "Fresh asphalt is generated for future overrides.",
        ],
    }

    (output / "materials.json").write_text(
        json.dumps(material_config, indent=2),
        encoding="utf-8",
    )

    print(f"\nFinished: {output}")
    print("Materials written to materials.json")


if __name__ == "__main__":
    main()
