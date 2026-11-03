#!/usr/bin/env python3
"""
Package generated vegetation for the legacy C++ viewer.

The package combines renderer-neutral VPL1 placement files with an appearance
library and emits VEG1 placement files plus VEGETATION_ASSETS_V4 metadata.
Each asset has twelve textures: four upright near cards, four upright medium
cards, and four reduced upright far cards. The viewer selects two, three, or
four cards according to the active preset.

Usage:
    python pack_viewer_vegetation.py -i Historyviewer/espoo2008_veg_tiles_new -o Historyviewer/espoo2008_veg_viewer_new

Arguments:
    -i, --input       Folder containing vegetation_manifest.json and
                      placements/*.vpl (or legacy vegetation/*.veg).
    -o, --output      New or empty packaged-output folder.
    --assets          Appearance library containing vegetation.json and
                      generated asset definitions; required for VPL1.
    --texture-size   Maximum near texture dimension; default 256.

Requires Pillow:
    pip install pillow

Output:
    assets.txt                 VEGETATION_ASSETS_V4 metadata.
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


def load_asset_table(source, manifest, override):
    """Load appearance definitions independently from placement tiles."""
    if override:
        library = json.loads(
            (override / "vegetation.json").read_text(encoding="utf-8")
        )
        return [
            {
                "numeric_id": numeric_id,
                "id": item["id"],
                "definition": str(
                    (override / item["definition"]).resolve()
                ),
            }
            for numeric_id, item in enumerate(library["assets"])
        ]

    return sorted(
        manifest["asset_table"],
        key=lambda item: item["numeric_id"],
    )


def read_vpl(path, asset_info):
    """Read VPL1 placements and convert physical dimensions at export time."""
    with path.open("rb") as handle:
        header = handle.read(4)
        if header != b"VPL1":
            raise ValueError(f"Unsupported placement format: {path}")

        xmin, ymax, cell_size, cell_count, plant_count = struct.unpack(
            "<2df2I", handle.read(struct.calcsize("<2df2I"))
        )
        cells = []
        total = 0

        for _ in range(cell_count):
            column, row, count = struct.unpack(
                "<3I", handle.read(struct.calcsize("<3I"))
            )
            plants = []
            for _ in range(count):
                raw_key, east, south, elevation, height, width_multiplier, yaw, tint, priority = struct.unpack(
                    "<64s8f", handle.read(struct.calcsize("<64s8f"))
                )
                key = raw_key.split(b"\0", 1)[0].decode("utf-8")
                if key not in asset_info:
                    raise ValueError(
                        f"Placement references missing asset key {key!r}: {path}"
                    )
                numeric_id, plant_height = asset_info[key]
                plants.append((
                    numeric_id,
                    east,
                    south,
                    elevation,
                    height / plant_height,
                    (height / plant_height) * width_multiplier,
                    yaw,
                    tint,
                    priority,
                ))
            cells.append((column, row, plants))
            total += count

        if total != plant_count:
            raise ValueError(f"Invalid VPL1 plant count: {path}")

    return xmin, ymax, cell_size, cells, plant_count


def write_veg(path, placement):
    """Write viewer VEG1 data from canonical VPL1 placement records."""
    xmin, ymax, cell_size, cells, plant_count = placement
    with path.open("wb") as handle:
        handle.write(b"VEG1")
        handle.write(struct.pack(
            "<2df2I",
            xmin, ymax, cell_size, len(cells), plant_count,
        ))
        for column, row, plants in cells:
            handle.write(struct.pack("<3I", column, row, len(plants)))
            for asset_id, east, south, elevation, height, width, yaw, tint, priority in plants:
                handle.write(struct.pack(
                    "<I8f",
                    asset_id,
                    east,
                    south,
                    elevation,
                    height,
                    width,
                    yaw,
                    tint,
                    priority,
                ))


def main():
    parser = argparse.ArgumentParser(
        description="Package vegetation for the legacy viewer."
    )
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument(
        "--assets",
        help="Appearance library folder; required when exporting VPL1.",
    )
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

    asset_folder = (
        Path(args.assets).resolve()
        if args.assets else None
    )
    asset_table = load_asset_table(source, manifest, asset_folder)

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

        # Twelve textures used by the renderer: four upright cards for each
        # of the near, medium, and far distance LODs.
        entries = [
            (lods[0]["front"], size),
            (lods[0]["side"], size),
            (lods[0]["third"], size),
            (lods[0]["fourth"], size),
            (lods[1]["front"], size // 2),
            (lods[1]["side"], size // 2),
            (lods[1]["third"], size // 2),
            (lods[1]["fourth"], size // 2),
            (lods[2]["front"], size // 4),
            (lods[2]["side"], size // 4),
            (lods[2]["third"], size // 4),
            (lods[2]["fourth"], size // 4),
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
        handle.write("VEGETATION_ASSETS_V4\n")
        handle.write(f"{len(records)}\n")

        for asset_id, width, height, pu, pv, paths in records:
            handle.write(
                f"{asset_id} {width:.9f} {height:.9f} "
                f"{pu:.9f} {pv:.9f}\n"
            )
            for path in paths:
                handle.write(path + "\n")

    tile_records = []

    asset_info = {}
    for item in asset_table:
        definition = Path(item["definition"])
        if not definition.is_absolute():
            definition = source / definition
        asset = json.loads(
            definition.read_text(encoding="utf-8")
        )
        asset_info[item["id"]] = (
            item["numeric_id"],
            asset["plant_height_metres"],
        )

    for tile in manifest["tiles"]:
        placement_name = tile.get(
            "placement_filename",
            tile.get("vegetation_filename"),
        )
        if not placement_name:
            raise ValueError("Manifest tile has no placement filename.")
        source_file = source / placement_name
        filename = source_file.name
        if source_file.suffix.lower() == ".vpl":
            if not asset_folder and manifest.get("format") == "vegetation_manifest_v2":
                raise ValueError(
                    "VPL1 export requires --assets pointing to the appearance library."
                )
            placement = read_vpl(source_file, asset_info)
            output_name = source_file.with_suffix(".veg").name
            write_veg(placements / output_name, placement)
            filename = output_name
        else:
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
