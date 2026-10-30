#!/usr/bin/env python3
"""
Package generated vegetation for the legacy C++ viewer.

Usage:
    python pack_viewer_vegetation.py \
        -i 2008_vegetation_test -o 2008_vegetation_viewer

Arguments:
    -i, --input       Folder containing vegetation_manifest.json.
    -o, --output      New or empty packaged-output folder.
    --texture-size   Maximum near texture dimension; default 256.

Requires Pillow:
    pip install pillow

Output:
    assets.txt
    vegetation_index.txt
    textures/*.tga
    vegetation/*.veg
"""

import argparse
import json
import shutil
import struct
from pathlib import Path

from PIL import Image


def write_tga(image, destination):
    """Write uncompressed 32-bit TGA with a bottom-left image origin."""
    image = image.convert("RGBA")
    width, height = image.size

    # TGA image type 2: uncompressed true-color.
    # Descriptor 8: eight alpha bits and bottom-left origin.
    header = struct.pack(
        "<BBBHHBHHHHBB",
        0, 0, 2,
        0, 0, 0,
        0, 0,
        width, height,
        32, 8,
    )

    # TGA stores BGRA pixels. Reverse rows for bottom-left origin.
    pixels = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    data = pixels.tobytes("raw", "BGRA")

    with destination.open("wb") as handle:
        handle.write(header)
        handle.write(data)


def main():
    parser = argparse.ArgumentParser(
        description="Package vegetation for the legacy viewer."
    )
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--texture-size", type=int, default=256)
    args = parser.parse_args()

    size = args.texture_size
    if size < 64 or size > 1024 or size & (size - 1):
        parser.error("--texture-size must be a power of two from 64 to 1024.")

    source = Path(args.input).resolve()
    output = Path(args.output).resolve()

    if output.exists() and (
        not output.is_dir() or any(output.iterdir())
    ):
        parser.error("Output folder must be empty or not yet exist.")

    manifest = json.loads(
        (source / "vegetation_manifest.json").read_text(encoding="utf-8")
    )

    textures = output / "textures"
    placements = output / "vegetation"
    textures.mkdir(parents=True)
    placements.mkdir()

    asset_table = sorted(
        manifest["asset_table"],
        key=lambda item: item["numeric_id"],
    )

    if [item["numeric_id"] for item in asset_table] != list(
        range(len(asset_table))
    ):
        raise ValueError("Asset IDs must be contiguous starting at zero.")

    records = []

    for item in asset_table:
        definition = Path(item["definition"])
        if not definition.is_absolute():
            definition = source / definition

        if not definition.is_file():
            raise ValueError(
                f"Asset definition not found: {definition}\n"
                "Package the vegetation before moving the original assets, "
                "or correct the manifest's asset paths."
            )

        asset = json.loads(definition.read_text(encoding="utf-8"))
        lods = asset["lods"]

        # Five textures used by the renderer:
        # near front/side, medium front/side, far front.
        entries = [
            (lods[0]["front"], size),
            (lods[0]["side"], size),
            (lods[1]["front"], size // 2),
            (lods[1]["side"], size // 2),
            (lods[2]["front"], size // 4),
        ]

        texture_paths = []

        for texture_number, (filename, limit) in enumerate(entries):
            original = definition.parent / filename

            with Image.open(original) as opened:
                image = opened.convert("RGBA")

            # Generated source images are square. Force a power-of-two
            # output compatible with old OpenGL hardware.
            image = image.resize(
                (limit, limit), Image.Resampling.LANCZOS
            )

            name = (
                f"asset_{item['numeric_id']:04d}_{texture_number}.tga"
            )
            write_tga(image, textures / name)
            texture_paths.append(f"textures/{name}")

        records.append((
            item["numeric_id"],
            asset["card_width_metres"],
            asset["card_height_metres"],
            asset["pivot_uv"][0],
            asset["pivot_uv"][1],
            texture_paths,
        ))

    # Line-based metadata avoids a C++ JSON dependency.
    with (output / "assets.txt").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write("VEGETATION_ASSETS_V1\n")
        handle.write(f"{len(records)}\n")

        for asset_id, width, height, pu, pv, paths in records:
            handle.write(
                f"{asset_id} {width:.9f} {height:.9f} "
                f"{pu:.9f} {pv:.9f}\n"
            )
            for path in paths:
                handle.write(path + "\n")

    tile_records = []

    for tile in manifest["tiles"]:
        source_file = source / tile["vegetation_filename"]
        filename = source_file.name
        shutil.copy2(source_file, placements / filename)

        # Match using basename: terrain_index can omit water-only tiles.
        terrain_name = Path(tile["terrain_filename"]).name
        tile_records.append((terrain_name, f"vegetation/{filename}"))

    with (output / "vegetation_index.txt").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write("VEGETATION_INDEX_V1\n")
        handle.write(f"{len(tile_records)}\n")
        for terrain_name, vegetation_name in tile_records:
            handle.write(terrain_name + "\n")
            handle.write(vegetation_name + "\n")

    print(
        f"Packaged {len(records)} assets and "
        f"{len(tile_records)} vegetation tiles in {output}"
    )


if __name__ == "__main__":
    main()
