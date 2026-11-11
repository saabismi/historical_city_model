#!/usr/bin/env python3
"""Generate procedural RGBA billboard assets for the land-cover pipeline.

Requirements:
    Python 3 and Pillow:
        python -m pip install Pillow

Example:
    python generate_billboard_assets.py \
        --output-dir ./billboard_assets \
        --size 256 \
        --seed 12345

Outputs:
    vegetation.json:
        Library manifest with relative paths to asset definitions.

    <asset_id>/asset.json:
        Physical card dimensions, pivot, category, and LOD texture names.

    <asset_id>/lod<level>_<view>.png:
        Four RGBA textures per LOD, with three LOD levels.

Rendering:
    Each view is a procedural placeholder, not a projection of a 3D model.
    Lower LODs are resized versions of the same source view, preserving
    silhouette and physical dimensions.

    Artwork extends from the declared ground pivot toward the card top.
    Transparent side margins mean visible width may be smaller than the
    declared card width.

Compatibility:
    JSON fields and texture filenames follow the supplied V4-style layout.
    The viewer must accept different texture resolutions across LODs.
    pivot_uv assumes V=0 is the bottom of the card.

Failure behavior:
    The output directory must be empty or nonexistent.
    Failed runs may leave partial output; inspect or remove it before retrying.

Reproducibility:
    Results are repeatable under the same implementation and environment.
    Reordering ASSETS changes the seeds assigned to individual assets.
"""

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw


# Each entry maps a placement key to:
#     (height_range_metres, width_to_height_ratio_range, base_rgb)
#
# Midpoints determine canonical physical card dimensions. Placement code
# may scale these dimensions per instance.
# Keys must match those emitted by the placement generator.
ASSETS = {
    "tree_pine": ((22, 32), (0.32, 0.42), (44, 86, 54)),
    "tree_deciduous": ((20, 30), (0.40, 0.55), (82, 132, 58)),
    "tree_mixed": ((18, 30), (0.38, 0.52), (66, 116, 56)),
    "bush_leafy": ((1.2, 2.8), (1.0, 1.5), (68, 116, 52)),
    "grass_meadow": ((0.35, 0.65), (0.8, 1.4), (118, 148, 57)),
    "grass_lawn": ((0.25, 0.45), (0.9, 1.5), (88, 142, 52)),
    "grass_exposed": ((0.2, 0.4), (0.9, 1.4), (96, 145, 56)),
    "reed_wetland": ((0.7, 1.4), (0.35, 0.7), (76, 123, 61)),
    "wheat": ((0.5, 0.9), (0.35, 0.65), (190, 166, 68)),
    "rock_moss": ((0.6, 1.8), (0.8, 1.5), (112, 115, 93)),
    # Finnish-style birch: pale bark with dark horizontal markings.
    # Physical dimensions use the same midpoint convention as other assets.
    "tree_birch": ((18, 26), (0.35, 0.50), (105, 151, 65)),

}

# These factors affect texture resolution, not physical plant dimensions.
LOD_FACTORS = (1.0, 0.55, 0.28)

# Pillow image rows increase downward; the manifest assumes UV V increases
# upward. main() derives the pivot from the actual raster ground position.
GROUND_V = 0.02


def asset_category(key: str) -> str:
    """Return the manifest category associated with an asset ID."""
    if key.startswith("rock_"):
        return "rock"
    return key.split("_", 1)[0]


def varied_colour(colour: tuple, rng: random.Random) -> tuple:
    """Return an opaque RGB variation, clamped to valid channel values."""
    offset = rng.randint(-14, 14)
    return tuple(max(0, min(255, value + offset)) for value in colour) + (255,)


def draw_birch_trunk(
    draw: ImageDraw.ImageDraw,
    center: int,
    top: int,
    base: int,
    half_width: int,
    rng: random.Random,
) -> None:
    """Draw white birch bark with irregular dark horizontal markings.

    Args:
        draw: Drawing interface for the destination RGBA image.
        center: Horizontal centre of the trunk, in pixels.
        top: Upper trunk coordinate, in pixels.
        base: Ground-level trunk coordinate, in pixels.
        half_width: Half of the trunk width, at least one pixel.
        rng: Seeded random generator for repeatable bark markings.

    Markings stay inside the trunk bounds. Some begin at either edge,
    while others form shorter interior dashes. Narrow trunks and reduced
    LODs necessarily simplify this detail.
    """
    half_width = max(1, half_width)
    left = center - half_width
    right = center + half_width
    trunk_height = base - top

    # Slightly shaded edges suggest a rounded, off-white trunk.
    draw.rectangle((left, top, right, base), fill=(232, 232, 220, 255))
    draw.line((left, top, left, base), fill=(183, 188, 180, 255))
    draw.line((right, top, right, base), fill=(204, 206, 195, 255))

    # Vary spacing, length, and thickness rather than using uniform bands.
    spacing = max(4, trunk_height // 14)
    y = top + rng.randint(1, spacing)
    while y <= base:
        length = rng.randint(1, max(1, right - left))
        style = rng.randrange(3)

        if style == 0:
            stripe_left = left
        elif style == 1:
            stripe_left = right - length
        else:
            stripe_left = rng.randint(left, right - length)

        stripe_right = stripe_left + length
        thickness = rng.randint(1, max(1, half_width // 2))
        stripe_bottom = min(base, y + thickness - 1)

        draw.rectangle(
            (stripe_left, y, stripe_right, stripe_bottom),
            fill=(43, 46, 43, 255),
        )
        y += rng.randint(max(2, spacing // 2), spacing + 2)


def draw_asset(key: str, size: int, colour: tuple, seed: int) -> Image.Image:
    """Draw one source view on a transparent square image.

    Args:
        key: Known asset ID; explicitly determines the drawing style.
        size: Image width and height, at least 64 pixels.
        colour: Three-channel base RGB colour.
        seed: Seed for this view's private random generator.

    Returns:
        An RGBA Pillow image owned by the caller.

    Raises:
        KeyError: If key is not defined in ASSETS.
        ValueError: If size is less than 64.

    Shapes stay within the image boundaries. The shared baseline matches
    the pivot recorded by main(). These are stylized placeholder assets,
    not physically accurate botanical models.
    """
    if key not in ASSETS:
        raise KeyError(f"Unknown asset ID: {key}")
    if size < 64:
        raise ValueError("Image size must be at least 64.")

    rng = random.Random(seed)
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    center = size // 2
    base = round((size - 1) * (1.0 - GROUND_V))
    top = max(1, round(size * 0.02))
    span = base - top
    category = asset_category(key)

    if category == "rock":
        left = round(size * 0.08)
        right = round(size * 0.92)
        draw.ellipse((left, top, right, base), fill=(*colour, 255))
        draw.ellipse(
            (
                round(size * 0.20),
                top + round(span * 0.10),
                round(size * 0.65),
                top + round(span * 0.55),
            ),
            fill=(145, 148, 111, 255),
        )

    elif category in ("grass", "reed", "wheat"):
        # Include one full-height blade so the visible height corresponds
        # approximately to the physical card height.
        line_width = max(1, round(size * 0.012))
        for blade in range(24):
            x = rng.randint(round(size * 0.18), round(size * 0.82))
            tip_x = max(1, min(size - 2, x + rng.randint(
                -max(1, size // 20), max(1, size // 20)
            )))
            tip_y = top if blade == 0 else rng.randint(
                top, top + round(span * 0.55)
            )
            draw.line(
                (x, base, tip_x, tip_y),
                fill=varied_colour(colour, rng),
                width=line_width,
            )

    elif key == "tree_pine":
        trunk_half = max(1, size // 70)
        draw.rectangle(
            (center - trunk_half, top + span // 3,
             center + trunk_half, base),
            fill=(112, 78, 47, 255),
        )
        # Overlapping triangular tiers produce a conifer silhouette.
        for tier in range(4):
            apex = top + round(span * tier * 0.17)
            bottom = top + round(span * (0.42 + tier * 0.16))
            half_width = round(size * (0.17 + tier * 0.075))
            draw.polygon(
                (
                    (center, apex),
                    (center - half_width, bottom),
                    (center + half_width, bottom),
                ),
                fill=varied_colour(colour, rng),
            )

    else:
        # Deciduous trees, birches, and bushes share a rounded crown.
        # Birch gets its own bark treatment; other assets keep brown trunks.
        is_tree = category == "tree"
        trunk_top = top + span // 3

        if key == "tree_birch":
            # A slightly wider placeholder trunk keeps bark markings
            # readable at the default source resolution.
            trunk_half = max(2, size // 48)
            draw_birch_trunk(
                draw, center, trunk_top, base, trunk_half, rng
            )
        else:
            trunk_half = max(1, size // 60)
            draw.rectangle(
                (center - trunk_half, trunk_top,
                 center + trunk_half, base),
                fill=(112, 78, 47, 255),
            )

        crown_bottom = top + round(span * (0.78 if is_tree else 0.97))
        draw.ellipse(
            (round(size * 0.13), top,
             round(size * 0.87), crown_bottom),
            fill=(*colour, 255),
        )

        # Keep all leaf clusters within the usable image region.
        for _ in range(14 if is_tree else 10):
            radius = rng.randint(max(2, size // 16), max(3, size // 8))
            x = rng.randint(round(size * 0.08) + radius,
                            round(size * 0.92) - radius)
            y = rng.randint(top + radius, crown_bottom - radius)
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=varied_colour(colour, rng),
            )

    return image


def write_json(path: Path, document: dict) -> None:
    """Write a UTF-8 JSON document with indentation and a trailing newline."""
    path.write_text(
        json.dumps(document, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Validate arguments and generate the complete billboard library.

    Every source view is rendered once and reused for all LODs.
    The library manifest is written only after all assets are complete.
    """
    parser = argparse.ArgumentParser(
        description="Generate land-cover billboard assets."
    )
    parser.add_argument(
        "-o", "--output-dir", required=True,
        help="Destination directory; must be empty or nonexistent.",
    )
    parser.add_argument(
        "--size", type=int, default=256,
        help="Source texture width and height, at least 64 (default: 256).",
    )
    parser.add_argument(
        "--seed", type=int, default=12345,
        help="Base random seed (default: 12345).",
    )
    args = parser.parse_args()

    if args.size < 64:
        parser.error("--size must be at least 64.")

    try:
        output = Path(args.output_dir).expanduser().resolve()
        if output.exists():
            if not output.is_dir():
                parser.error(f"Output path is not a directory: {output}")
            if any(output.iterdir()):
                parser.error(f"Output directory must be empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(f"Cannot prepare output directory: {exc}")

    # Match the manifest pivot to the rounded source-image baseline.
    baseline = round((args.size - 1) * (1.0 - GROUND_V))
    pivot_v = 1.0 - baseline / (args.size - 1)

    records = []
    print(
        f"Generating {len(ASSETS)} billboard assets in {output}...",
        flush=True,
    )

    try:
        for index, (key, configuration) in enumerate(ASSETS.items()):
            height_range, width_range, colour = configuration
            height = sum(height_range) / 2.0
            width = height * (sum(width_range) / 2.0)

            print(
                f"Asset {index + 1}/{len(ASSETS)}: {key}",
                flush=True,
            )

            asset_dir = output / key
            asset_dir.mkdir()

            # Render independent placeholder views, but preserve each view
            # across LOD transitions by resizing rather than regenerating.
            for view in range(4):
                source = draw_asset(
                    key, args.size, colour,
                    args.seed + index * 100 + view,
                )
                try:
                    for lod, factor in enumerate(LOD_FACTORS):
                        filename = asset_dir / f"lod{lod}_{view}.png"
                        resolution = max(1, round(args.size * factor))

                        if resolution == args.size:
                            source.save(filename, format="PNG")
                        else:
                            # Pillow's RGBA resizing handles alpha during
                            # filtering; thin features may still fade at
                            # lower resolutions under viewer alpha testing.
                            reduced = source.resize(
                                (resolution, resolution),
                                resample=Image.Resampling.LANCZOS,
                            )
                            try:
                                reduced.save(filename, format="PNG")
                            finally:
                                reduced.close()
                finally:
                    source.close()

            lods = [
                {
                    "front": f"lod{lod}_0.png",
                    "side": f"lod{lod}_1.png",
                    "third": f"lod{lod}_2.png",
                    "fourth": f"lod{lod}_3.png",
                }
                for lod in range(len(LOD_FACTORS))
            ]

            definition = {
                "id": key,
                "species": key,
                "category": asset_category(key),
                "plant_height_metres": height,
                "card_width_metres": width,
                "card_height_metres": height,
                "pivot_uv": [0.5, pivot_v],
                "lods": lods,
            }
            definition_path = asset_dir / "asset.json"
            write_json(definition_path, definition)

            records.append({
                "id": key,
                "definition": definition_path.relative_to(output).as_posix(),
            })

        write_json(
            output / "vegetation.json",
            {"format": "landcover_assets_v1", "assets": records},
        )
    except (OSError, MemoryError, OverflowError) as exc:
        parser.error(
            f"Generation failed: {exc}. "
            f"Partial output may remain in {output}; "
            "inspect or remove it before rerunning."
        )

    print(f"Generated {len(records)} assets in {output}", flush=True)


if __name__ == "__main__":
    main()
