#!/usr/bin/env python3
"""
Pack land-cover placements and generated assets for the legacy viewer.

This is the viewer-specific final stage of the prototype pipeline:

1. ``generate_landcover.py`` writes renderer-neutral VPL1 placements and
   per-tile land-cover material maps.
2. ``generate_landcover_assets.py`` writes ``vegetation.json`` and the
   procedural PNG billboard library.
3. ``generate_landcover_textures.py`` writes terrain-material PNGs and
   ``terrain_materials.json``.
4. This script converts the placements and PNG billboards to the legacy
   Historyviewer formats.

The output contains the files required by Historyviewer:
``assets.txt``, twelve TGA textures per asset, ``vegetation_index.txt``, and
one VEG1 file per terrain tile. Land-cover JSON maps and generated terrain
textures are copied alongside those files for inspection and for future
terrain-material rendering.

The current legacy viewer consumes the packed vegetation assets and VEG1
placements. It does not yet read ``landcover/*.json`` or
``terrain_textures/*.png``; those files are still exported so terrain
material rendering can be added without regenerating land-cover placements.

With --deform-terrain, this final stage also creates a separate deformed
terrain folder from existing TRN1 terrain and ROAD1 road tiles. The original
terrain is never modified.

Example (Windows cmd.exe):
    python generate_landcover.py ^
        --dem espoo2008_dem_tiled ^
        --mtk MTK_2009_clipped ^
        --pointcloud espoo1965_clouds ^
        --buildings MTK_2009_clipped/mtk-rakennus_2009_clipped.gpkg ^
        --roads MTK_2009_clipped/mtk-tie_2009_clipped.gpkg ^
        --workers 4 ^
        -o espoo2008_landcover_output

    python generate_landcover_assets.py ^
        --output-dir espoo2008_assets

    python generate_landcover_textures.py ^
        --output-dir espoo2008_textures

    python pack_assets_textures_to_viewer.py ^
        --input espoo2008_landcover_output ^
        --assets espoo2008_assets ^
        --textures espoo2008_textures ^
        --output Historyviewer/2008_landcover_viewer

The resulting viewer folder must be supplied through the existing
Historyviewer vegetation-assets setting. Its terrain and building/road
folders remain the normal prepared viewer data folders.
"""

import argparse
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

from PIL import Image


# ``generate_landcover.py`` writes semantic categories in each tile map,
# while ``generate_landcover_textures.py`` writes renderer material IDs.
# Keep this small export-facing mapping in sync with CONFIG["materials"].
LANDCOVER_MATERIALS = {
    "forest": "forest_floor",
    "wetland": "wetland_dark",
    "rock": "exposed_rock",
    "yard": "gravel_yard",
    "cemetery": "grass",
    "agriculture": "agriculture",
    "reed": "grass",
    "meadow": "meadow",
    "lawn": "grass",
    "exposed": "grass",
}


def load_json(path):
    """Load a UTF-8 JSON file with a path-specific error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read JSON file {path}: {error}") from error


def validate_material_textures(source, textures):
    """Check generated material IDs cover semantic land-cover map values."""
    manifest_path = textures / "terrain_materials.json"
    if not manifest_path.is_file():
        raise ValueError(
            f"Missing generated terrain material manifest: {manifest_path}"
        )

    manifest = load_json(manifest_path)
    available = {
        item["id"]
        for item in manifest.get("materials", [])
        if isinstance(item, dict) and item.get("id")
    }

    missing = set()
    required = set()
    for path in sorted((source / "landcover").glob("*.json")):
        data = load_json(path)
        for row in data.get("materials", []):
            for category in row:
                if category in available:
                    required.add(category)
                    continue
                if category not in LANDCOVER_MATERIALS:
                    raise ValueError(
                        f"Unknown land-cover category {category!r} in {path}"
                    )
                required.add(LANDCOVER_MATERIALS[category])

    missing.update(required - available)

    if missing:
        raise ValueError(
            "Generated terrain textures do not define material IDs required "
            "by land-cover maps: "
            + ", ".join(sorted(missing))
        )

    texture_files = {
        item["id"]: item.get("texture")
        for item in manifest.get("materials", [])
        if isinstance(item, dict) and item.get("id")
    }
    missing_files = sorted(
        material
        for material in required
        if not texture_files.get(material)
        or not (textures / texture_files[material]).is_file()
    )
    if missing_files:
        raise ValueError(
            "Generated terrain material files are missing: "
            + ", ".join(missing_files)
        )


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


def write_ground_map(path, data, material_ids):
    """Write compact LCM1 material cells for the legacy terrain renderer."""
    materials = data.get("materials")
    if not materials or not isinstance(materials, list):
        raise ValueError(f"Land-cover map has no materials: {path}")
    rows = len(materials)
    columns = len(materials[0])
    if columns == 0 or any(len(row) != columns for row in materials):
        raise ValueError(f"Land-cover map rows have inconsistent widths: {path}")
    resolution = float(data.get("resolution", 0))
    if resolution <= 0:
        raise ValueError(f"Invalid land-cover resolution: {path}")
    values = []
    for row in materials:
        for category in row:
            category = LANDCOVER_MATERIALS.get(category, category)
            if category not in material_ids:
                raise ValueError(
                    f"Material {category!r} is not in the texture manifest: {path}"
                )
            values.append(material_ids[category])
    with path.open("wb") as handle:
        handle.write(b"LCM1")
        handle.write(struct.pack("<IIf", columns, rows, resolution))
        handle.write(struct.pack(f"<{len(values)}H", *values))


def read_vpl(path, asset_info):
    """Convert physical VPL1 placements to viewer-relative VEG1 scales."""
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
    """Write one tile-batched VEG1 package for the legacy C++ viewer."""
    xmin, ymax, cell_size, cells, plant_count = placement
    with path.open("wb") as handle:
        handle.write(b"VEG1")
        handle.write(struct.pack("<2df2I", xmin, ymax, cell_size, len(cells), plant_count))
        for column, row, plants in cells:
            handle.write(struct.pack("<3I", column, row, len(plants)))
            for plant in plants:
                handle.write(struct.pack("<I8f", *plant))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert land-cover VPL1 placements and generated billboard "
            "assets into Historyviewer VEG1/V4 files."
        ),
        epilog=(
            "The legacy viewer renders the packed vegetation. Use --textures "
            "to copy and validate terrain materials; the current viewer does "
            "not render those PNGs yet."
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="generate_landcover.py output directory.",
    )
    parser.add_argument(
        "--assets",
        required=True,
        help="generate_landcover_assets.py output directory.",
    )
    parser.add_argument(
        "--textures",
        help=(
            "generate_landcover_textures.py output directory. If omitted, "
            "input/terrain_textures is used when present."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Empty directory to create for Historyviewer.",
    )
    parser.add_argument(
        "--texture-size",
        type=int,
        default=256,
        help="Near-LOD TGA size; medium/far use half/quarter size.",
    )
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
    textures = (
        Path(args.textures).resolve()
        if args.textures
        else source / "terrain_textures"
    )
    output = Path(args.output).resolve()
    for label, path in (
        ("land-cover input", source),
        ("asset input", assets),
    ):
        if not path.is_dir():
            parser.error(f"{label} directory does not exist: {path}")
    if args.textures and not textures.is_dir():
        parser.error(f"Texture directory does not exist: {textures}")
    if args.texture_size < 64 or args.texture_size % 4:
        parser.error("--texture-size must be at least 64 and divisible by 4.")

    if output.exists() and any(output.iterdir()):
        parser.error("Output must be empty or not yet exist.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "textures").mkdir()
    (output / "vegetation").mkdir()
    (output / "landcover").mkdir()

    # The appearance library is independent of placement generation. This
    # lets billboard art change without recomputing point-cloud placement.
    library = load_json(assets / "vegetation.json")
    if library.get("format") != "landcover_assets_v1":
        parser.error("Unsupported vegetation asset manifest format.")
    asset_info = {}
    records = []
    print(
        f"Packing {len(library['assets'])} assets for the viewer...",
        flush=True,
    )
    for asset_id, item in enumerate(library["assets"]):
        definition_path = assets / item["definition"]
        definition = load_json(definition_path)
        print(
            f"Asset {asset_id + 1}/{len(library['assets'])}: "
            f"{definition['id']} -> TGA textures",
            flush=True,
        )
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

    manifest = load_json(source / "landcover_manifest.json")
    if manifest.get("placement_format") != "VPL1":
        parser.error("Unsupported land-cover placement format.")
    if not textures.is_dir():
        print(
            "No terrain texture directory supplied; vegetation will still "
            "be packaged.",
            flush=True,
        )
    else:
        validate_material_textures(source, textures)

    material_ids = {}
    if textures.is_dir():
        texture_manifest = load_json(textures / "terrain_materials.json")
        material_ids = {
            item["id"]: index
            for index, item in enumerate(texture_manifest.get("materials", []))
            if isinstance(item, dict) and item.get("id")
        }
        texture_records = []
        for item in texture_manifest.get("materials", []):
            material_id = item["id"]
            source_texture = textures / item["texture"]
            output_texture = output / "textures" / f"ground_{material_id}.tga"
            with Image.open(source_texture) as image:
                write_tga(image, output_texture)
            texture_records.append((material_id, f"textures/{output_texture.name}"))

        with (output / "ground_materials.txt").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            handle.write(f"GROUND_MATERIALS_V1\n{len(texture_records)}\n")
            for material_id, relative in texture_records:
                handle.write(f"{material_id} {relative}\n")

    tile_records = []
    print(
        f"Packing {len(manifest['tiles'])} placement/land-cover tiles...",
        flush=True,
    )
    for tile_number, tile in enumerate(manifest["tiles"], 1):
        print(
            f"Tile {tile_number}/{len(manifest['tiles'])}: "
            f"converting placements and copying land-cover map",
            flush=True,
        )
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
        if material_ids:
            landcover_data = load_json(source / tile["landcover_filename"])
            ground_name = Path(tile["terrain_filename"]).stem + ".lcm"
            write_ground_map(
                output / "landcover" / ground_name,
                landcover_data,
                material_ids,
            )

    with (output / "vegetation_index.txt").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"VEGETATION_INDEX_V1\n{len(tile_records)}\n")
        for terrain_name, vegetation_name in tile_records:
            handle.write(terrain_name + "\n" + vegetation_name + "\n")

    if material_ids:
        with (output / "ground_index.txt").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            handle.write(f"GROUND_INDEX_V1\n{len(tile_records)}\n")
            for terrain_name, _ in tile_records:
                ground_name = Path(terrain_name).stem + ".lcm"
                handle.write(terrain_name + "\n")
                handle.write(f"landcover/{ground_name}\n")

    shutil.copy2(
        source / "landcover_manifest.json",
        output / "landcover_manifest.json",
    )
    textures_source = textures
    if textures_source.is_dir():
        shutil.copytree(
            textures_source,
            output / "terrain_textures",
            dirs_exist_ok=True,
        )

    if args.deform_terrain:
        print(
            "Starting terrain deformation beneath ROAD1 road meshes...",
            flush=True,
        )
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
    print(
        f"Packaged {len(records)} assets and {len(tile_records)} tiles in {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
