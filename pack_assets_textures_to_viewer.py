#!/usr/bin/env python3
"""
Pack land-cover placements and generated assets for the legacy viewer.

This is the viewer-specific final stage. It converts VPL1 placement tiles to
the existing VEG1 tile format, writes the V4 billboard texture table, and
copies the land-cover material maps so a future terrain renderer can consume
them without rerunning placement generation.

With --deform-terrain, this final stage also creates a separate deformed
terrain folder from existing TRN1 terrain and ROAD1 road tiles. The original
terrain is never modified.
"""

import argparse
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

from PIL import Image


def write_tga(image, destination):
    image = image.convert("RGBA")
    width, height = image.size
    header = struct.pack(
        "<BBBHHBHHHHBB", 0, 0, 2, 0, 0, 0, 0, 0,
        width, height, 32, 8,
    )
    pixels = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    with destination.open("wb") as handle:
        handle.write(header)
        handle.write(pixels.tobytes("raw", "BGRA"))


def read_vpl(path, asset_info):
    with path.open("rb") as handle:
        if handle.read(4) != b"VPL1":
            raise ValueError(f"Unsupported placement file: {path}")
        xmin, ymax, cell_size, cell_count, plant_count = struct.unpack(
            "<2df2I", handle.read(struct.calcsize("<2df2I"))
        )
        cells = []
        total = 0
        for _ in range(cell_count):
            column, row, count = struct.unpack("<3I", handle.read(12))
            plants = []
            for _ in range(count):
                raw_key, east, south, elevation, height, width, yaw, tint, priority = struct.unpack(
                    "<64s8f", handle.read(struct.calcsize("<64s8f"))
                )
                key = raw_key.split(b"\0", 1)[0].decode("utf-8")
                if key not in asset_info:
                    raise ValueError(f"Missing asset key {key!r}")
                asset_id, asset_height = asset_info[key]
                plants.append((
                    asset_id, east, south, elevation,
                    height / asset_height,
                    (height / asset_height) * width,
                    yaw, tint, priority,
                ))
            cells.append((column, row, plants))
            total += count
        if total != plant_count:
            raise ValueError(f"Invalid placement count in {path}")
    return xmin, ymax, cell_size, cells, plant_count


def write_veg(path, placement):
    xmin, ymax, cell_size, cells, plant_count = placement
    with path.open("wb") as handle:
        handle.write(b"VEG1")
        handle.write(struct.pack("<2df2I", xmin, ymax, cell_size, len(cells), plant_count))
        for column, row, plants in cells:
            handle.write(struct.pack("<3I", column, row, len(plants)))
            for plant in plants:
                handle.write(struct.pack("<I8f", *plant))


def main():
    parser = argparse.ArgumentParser(description="Pack land-cover assets for Historyviewer.")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--texture-size", type=int, default=256)
    parser.add_argument(
        "--deform-terrain",
        action="store_true",
        help="Create output/terrain with terrain lowered beneath ROAD1 meshes.",
    )
    parser.add_argument(
        "--terrain",
        help="Prepared viewer terrain folder used with --deform-terrain.",
    )
    parser.add_argument(
        "--roads",
        help="Generated viewer road folder used with --deform-terrain.",
    )
    args = parser.parse_args()

    if args.deform_terrain and (not args.terrain or not args.roads):
        parser.error("--deform-terrain requires --terrain and --roads.")

    source = Path(args.input).resolve()
    assets = Path(args.assets).resolve()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Output must be empty or not yet exist.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "textures").mkdir()
    (output / "vegetation").mkdir()
    (output / "landcover").mkdir()

    library = json.loads((assets / "vegetation.json").read_text(encoding="utf-8"))
    asset_info = {}
    records = []
    for asset_id, item in enumerate(library["assets"]):
        definition_path = assets / item["definition"]
        definition = json.loads(definition_path.read_text(encoding="utf-8"))
        asset_info[definition["id"]] = (
            asset_id, definition["plant_height_metres"]
        )
        paths = []
        lods = definition["lods"]
        entries = [
            (lods[0]["front"], args.texture_size),
            (lods[0]["side"], args.texture_size),
            (lods[0]["third"], args.texture_size),
            (lods[0]["fourth"], args.texture_size),
            (lods[1]["front"], args.texture_size // 2),
            (lods[1]["side"], args.texture_size // 2),
            (lods[1]["third"], args.texture_size // 2),
            (lods[1]["fourth"], args.texture_size // 2),
            (lods[2]["front"], args.texture_size // 4),
            (lods[2]["side"], args.texture_size // 4),
            (lods[2]["third"], args.texture_size // 4),
            (lods[2]["fourth"], args.texture_size // 4),
        ]
        for texture_index, (filename, size) in enumerate(entries):
            with Image.open(definition_path.parent / filename) as image:
                image = image.convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)
            name = f"asset_{asset_id:04d}_{texture_index}.tga"
            write_tga(image, output / "textures" / name)
            paths.append(f"textures/{name}")
        records.append((
            asset_id, definition["card_width_metres"],
            definition["card_height_metres"],
            definition["pivot_uv"][0], definition["pivot_uv"][1], paths,
        ))

    with (output / "assets.txt").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"VEGETATION_ASSETS_V4\n{len(records)}\n")
        for asset_id, width, height, pu, pv, paths in records:
            handle.write(f"{asset_id} {width:.9f} {height:.9f} {pu:.9f} {pv:.9f}\n")
            handle.write("\n".join(paths) + "\n")

    manifest = json.loads((source / "landcover_manifest.json").read_text(encoding="utf-8"))
    tile_records = []
    for tile in manifest["tiles"]:
        input_path = source / tile["placement_filename"]
        output_name = input_path.with_suffix(".veg").name
        write_veg(
            output / "vegetation" / output_name,
            read_vpl(input_path, asset_info),
        )
        terrain_name = Path(tile["terrain_filename"]).name
        tile_records.append((terrain_name, f"vegetation/{output_name}"))
        shutil.copy2(
            source / tile["landcover_filename"],
            output / "landcover" / Path(tile["landcover_filename"]).name,
        )

    with (output / "vegetation_index.txt").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"VEGETATION_INDEX_V1\n{len(tile_records)}\n")
        for terrain_name, vegetation_name in tile_records:
            handle.write(terrain_name + "\n" + vegetation_name + "\n")

    shutil.copy2(source / "landcover_manifest.json", output / "landcover_manifest.json")
    textures_source = source / "terrain_textures"
    if textures_source.is_dir():
        shutil.copytree(
            textures_source,
            output / "terrain_textures",
            dirs_exist_ok=True,
        )

    if args.deform_terrain:
        script = Path(__file__).with_name("deform_terrain_to_roads.py")
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--terrain",
                str(Path(args.terrain).resolve()),
                "--roads",
                str(Path(args.roads).resolve()),
                "--output",
                str(output / "terrain"),
            ],
            check=True,
        )
    print(f"Packaged {len(records)} assets and {len(tile_records)} tiles in {output}")


if __name__ == "__main__":
    main()
