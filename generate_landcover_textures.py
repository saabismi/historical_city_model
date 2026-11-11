#!/usr/bin/env python3
"""Generate tileable low-dependency terrain textures for land-cover classes."""

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw


PALETTES = {
    "grass": ((102, 145, 73), (137, 161, 85)),
    "meadow": ((119, 145, 67), (177, 166, 83)),
    "forest_floor": ((62, 91, 52), (106, 112, 62)),
    "wetland_dark": ((48, 77, 55), (89, 105, 69)),
    "exposed_rock": ((108, 108, 99), (156, 148, 120)),
    "gravel_yard": ((119, 113, 101), (164, 151, 126)),
    "agriculture": ((164, 142, 62), (208, 183, 80)),
}


def make_texture(name, size, seed):
    rng = random.Random(seed)
    dark, light = PALETTES[name]
    image = Image.new("RGB", (size, size), dark)
    pixels = image.load()
    for y in range(size):
        for x in range(size):
            wave = (x * 13 + y * 7 + rng.randint(-18, 18)) % 100
            fraction = wave / 100.0
            pixels[x, y] = tuple(
                int(dark[index] * (1 - fraction) + light[index] * fraction)
                for index in range(3)
            )
    draw = ImageDraw.Draw(image)
    for _ in range(size // 3):
        x = rng.randrange(size)
        y = rng.randrange(size)
        radius = rng.randint(1, max(2, size // 32))
        colour = light if rng.random() < 0.5 else dark
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=colour)
    return image


def main():
    parser = argparse.ArgumentParser(description="Generate land-cover terrain textures.")
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    if args.size < 64 or args.size & (args.size - 1):
        parser.error("--size must be a power of two of at least 64.")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Output folder must be empty or not yet exist.")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    print(
        f"Generating {len(PALETTES)} terrain materials in {output}...",
        flush=True,
    )
    for index, name in enumerate(PALETTES):
        filename = f"{name}.png"
        print(
            f"Material {index + 1}/{len(PALETTES)}: {name}",
            flush=True,
        )
        make_texture(name, args.size, args.seed + index).save(output / filename)
        records.append({"id": name, "texture": filename, "tileable": True})
    (output / "terrain_materials.json").write_text(
        json.dumps({
            "format": "terrain_materials_v1",
            "materials": records,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Generated {len(records)} terrain materials in {output}", flush=True)


if __name__ == "__main__":
    main()
