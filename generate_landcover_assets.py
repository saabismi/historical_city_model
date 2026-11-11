#!/usr/bin/env python3
"""
Generate low-dependency billboard assets for the land-cover pipeline.

The generated library covers the placement keys emitted by
generate_landcover.py: trees, bushes, grass, reeds, and mossy rocks. Assets
use four upright cards per LOD and remain compatible with the legacy V4
viewer texture layout.
"""

import argparse
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw


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
}


def draw_asset(path, width, height, colour, seed):
    rng = random.Random(seed)
    size = 256
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    center = size // 2
    base = size - 12
    rx = int(size * 0.18)
    ry = int(size * 0.38)

    if "rock" in path.stem:
        draw.ellipse(
            (center - rx, base - ry, center + rx, base + 8),
            fill=(*colour, 255),
        )
        draw.ellipse(
            (center - rx // 2, base - ry, center + rx // 3, base - ry // 3),
            fill=(145, 148, 111, 190),
        )
    elif (
        "grass" in path.stem
        or "reed" in path.stem
        or "wheat" in path.stem
    ):
        for _ in range(18):
            x = center + rng.randint(-rx, rx)
            top = base - rng.randint(int(size * 0.25), int(size * 0.75))
            draw.line((x, base, x + rng.randint(-12, 12), top), fill=(*colour, 235), width=3)
    else:
        crown_width = int(size * (0.22 if width > 10 else 0.36))
        crown_height = int(size * (0.50 if height > 10 else 0.38))
        draw.rectangle(
            (center - 3, base - int(size * 0.42), center + 3, base),
            fill=(112, 78, 47, 255),
        )
        for _ in range(12 if height > 10 else 8):
            x = center + rng.randint(-crown_width, crown_width)
            y = base - rng.randint(crown_height // 3, crown_height)
            radius = rng.randint(max(8, crown_width // 5), max(12, crown_width // 2))
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*colour, 235))

    image.save(path)


def main():
    parser = argparse.ArgumentParser(description="Generate land-cover billboard assets.")
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Output folder must be empty or not yet exist.")
    output.mkdir(parents=True, exist_ok=True)

    records = []
    for index, (key, (height_range, width_range, colour)) in enumerate(ASSETS.items()):
        height = sum(height_range) / 2.0
        width = height * (sum(width_range) / 2.0)
        asset_dir = output / key
        asset_dir.mkdir()
        lods = []
        for lod, factor in enumerate((1.0, 0.55, 0.28)):
            names = []
            for view in range(4):
                filename = f"lod{lod}_{view}.png"
                draw_asset(
                    asset_dir / filename,
                    width,
                    height * factor,
                    colour,
                    args.seed + index * 100 + lod * 10 + view,
                )
                names.append(filename)
            lods.append({
                "front": f"lod{lod}_0.png",
                "side": f"lod{lod}_1.png",
                "third": f"lod{lod}_2.png",
                "fourth": f"lod{lod}_3.png",
            })

        definition = {
            "id": key,
            "species": key,
            "category": "rock" if "rock" in key else (
                "tree" if key.startswith("tree_") else key.split("_")[0]
            ),
            "plant_height_metres": height,
            "card_width_metres": width,
            "card_height_metres": height,
            "pivot_uv": [0.5, 0.02],
            "lods": lods,
        }
        definition_path = asset_dir / "asset.json"
        definition_path.write_text(json.dumps(definition, indent=2), encoding="utf-8")
        records.append({"id": key, "definition": str(definition_path.relative_to(output))})

    (output / "vegetation.json").write_text(
        json.dumps({"format": "landcover_assets_v1", "assets": records}, indent=2),
        encoding="utf-8",
    )
    print(f"Generated {len(records)} assets in {output}")


if __name__ == "__main__":
    main()
