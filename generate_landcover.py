#!/usr/bin/env python3
"""
Generate renderer-neutral land-cover, vegetation, and terrain-material data.

This is the single placement-generation stage. It reads DEM/terrain tiles,
MTK vector layers, and an independent MTK LiDAR point-cloud directory. It
does not consume prepared vegetation output.

The output contains VPL1 placement tiles for trees, bushes, grass, reeds,
wheat, and rocks, plus a small per-tile land-cover material map. The existing
viewer packer can consume the placement files after assets are generated.

Example:
    python generate_landcover.py ^
        --dem espoo2008_dem_tiled ^
        --mtk MTK_2009_clipped ^
        --pointcloud espoo_1965_clouds ^
        --buildings MTK_2009_clipped/mtk-rakennus_2009_clipped.gpkg ^
        --roads MTK_2009_clipped/mtk-tie_2009_clipped.gpkg ^
        -o espoo2008_landcover_output

Requires GDAL/OGR, rasterio, numpy, shapely, and laspy+lazrs.
"""

import argparse
import hashlib
import json
import math
import struct
import sys
from collections import defaultdict
from pathlib import Path

import laspy
import numpy as np
import rasterio
from osgeo import ogr, osr
from shapely.geometry import Point, box
from shapely.strtree import STRtree


CONFIG = {
    # All distances are metres. Placement spacings are approximate lattice
    # spacings, not measured object counts. Lower values create more
    # instances and fuller ground cover at the cost of larger VPL1 files.
    "grid_resolution": 10.0,
    "placement_seed": 12345,
    "tree_classes": (3, 4, 5),
    "tree_spacing": 5.5,
    "grass_spacing": 2.5,
    "bush_spacing": 7.0,
    "rock_spacing": 8.0,
    "road_buffer": 4.0,
    "building_buffer": 3.0,
    "pointcloud_height_floor": 1.5,
    "pointcloud_height_ceiling": 45.0,
    "forest_classification_layer": "metsamaankasvillisuus",
    "forest_classification_field": "LUOKKA",
    "polygon_layers": {
        "suo": "wetland",
        "soistuma": "wetland",
        "kallioalue": "rock",
        "autoliikennealue": "yard",
        "hautausmaa": "cemetery",
        "maatalousmaa": "agriculture",
        "matalikko": "reed",
        "niitty": "meadow",
        "puisto": "lawn",
        "varastoalue": "yard",
    },
    "materials": {
        # These IDs must match the material IDs emitted by
        # generate_landcover_textures.py. Puisto/lawn intentionally maps to
        # the default grass material; forests map to forest_floor instead.
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
    },
}

EPSG = 3067


def stable_rng(*parts):
    digest = hashlib.blake2b(
        (":".join(map(str, parts))).encode("utf-8"),
        digest_size=8,
    ).digest()
    return np.random.default_rng(int.from_bytes(digest, "little"))


def normalize_crs(crs):
    result = crs.Clone()
    result.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return result


def open_layer(path, layer_name):
    dataset = ogr.Open(str(path), 0)
    if dataset is None:
        raise ValueError(f"Cannot open vector source: {path}")
    layer = dataset.GetLayerByName(layer_name)
    if layer is None:
        dataset = None
        raise ValueError(f"Layer {layer_name!r} not found in {path}")
    source_crs = layer.GetSpatialRef()
    if source_crs is None:
        dataset = None
        raise ValueError(f"Layer has no CRS: {path}:{layer_name}")
    if not normalize_crs(source_crs).IsSame(
        normalize_crs(_epsg_3067())
    ):
        dataset = None
        raise ValueError(f"Expected EPSG:3067: {path}:{layer_name}")
    return dataset, layer


def _epsg_3067():
    crs = osr.SpatialReference()
    crs.ImportFromEPSG(EPSG)
    crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return crs


def read_geometries(path, layer_name, field=None):
    dataset, layer = open_layer(path, layer_name)
    geometries = []
    values = []
    for feature in layer:
        geometry = feature.GetGeometryRef()
        if geometry is None or geometry.IsEmpty():
            continue
        shape = ogr.CreateGeometryFromWkb(geometry.ExportToWkb())
        if not shape.IsValid():
            shape = shape.MakeValid()
        if shape is None or shape.IsEmpty():
            continue
        geometries.append(shape)
        values.append(feature.GetField(field) if field else None)
    dataset = None
    return geometries, values


def read_terrain_index(folder):
    with (folder / "tile_manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    entries = []
    for tile in manifest["tiles"]:
        bounds = tile.get("core_bounds")
        if bounds and tile.get("dem"):
            entries.append({
                "name": tile["tile"],
                "bounds": tuple(bounds),
                "dem": folder / tile["dem"],
            })
    if not entries:
        raise ValueError("No DEM-backed tiles found in tile_manifest.json")
    return entries


def read_pointcloud_files(folder):
    files = sorted(
        path for path in folder.rglob("*")
        if path.suffix.lower() in {".las", ".laz"}
    )
    if not files:
        raise ValueError(f"No LAS/LAZ files found under {folder}")
    return files


def load_points(files, bounds):
    """Load candidate vegetation points intersecting one tile bounds."""
    xmin, ymin, xmax, ymax = bounds
    chunks = []
    for filename in files:
        with laspy.open(filename) as reader:
            header = reader.header
            if (
                header.maxs[0] < xmin or header.mins[0] > xmax
                or header.maxs[1] < ymin or header.mins[1] > ymax
            ):
                continue
            for block in reader.chunk_iterator(500_000):
                classification = np.asarray(block.classification)
                keep = np.isin(classification, CONFIG["tree_classes"])
                x = np.asarray(block.x)
                y = np.asarray(block.y)
                keep &= (x >= xmin) & (x <= xmax)
                keep &= (y >= ymin) & (y <= ymax)
                if np.any(keep):
                    chunks.append(np.column_stack((
                        x[keep], y[keep], np.asarray(block.z)[keep],
                    )))
    return np.concatenate(chunks) if chunks else np.empty((0, 3))


def sample_dem(dataset, x, y):
    row, column = dataset.index(x, y)
    if row < 0 or row >= dataset.height or column < 0 or column >= dataset.width:
        return None
    value = dataset.read(1, window=((row, row + 1), (column, column + 1)))[0, 0]
    if dataset.nodata is not None and value == dataset.nodata:
        return None
    return float(value) if np.isfinite(value) else None


def load_sources(mtk, buildings, roads):
    layers = defaultdict(list)
    for filename, layer_name in (
        (mtk / "mtk-muut_2009_clipped.gpkg", None),
        (mtk / "mtk-vakavesi_2009_clipped.gpkg", None),
    ):
        if not filename.is_file():
            continue
        dataset = ogr.Open(str(filename), 0)
        for index in range(dataset.GetLayerCount()):
            layer = dataset.GetLayerByIndex(index)
            name = layer.GetName()
            if name in CONFIG["polygon_layers"]:
                geometries, _ = read_geometries(filename, name)
                layers[CONFIG["polygon_layers"][name]].extend(geometries)
        dataset = None

    forest_file = mtk / "mtk-muut_2009_clipped.gpkg"
    forest_points = []
    forest_codes = []
    if forest_file.is_file():
        try:
            forest_points, forest_codes = read_geometries(
                forest_file,
                CONFIG["forest_classification_layer"],
                CONFIG["forest_classification_field"],
            )
        except ValueError:
            pass

    exclusions = []
    if buildings:
        path = Path(buildings)
        if path.is_file():
            geometries, _ = read_geometries(path, "rakennus")
            exclusions.extend(g.buffer(CONFIG["building_buffer"]) for g in geometries)
    if roads:
        path = Path(roads)
        if path.is_file():
            try:
                geometries, _ = read_geometries(path, "tieviiva")
                exclusions.extend(g.buffer(CONFIG["road_buffer"]) for g in geometries)
            except ValueError:
                pass

    return layers, exclusions, forest_points, forest_codes


def tree_species(code):
    code = int(code) if code is not None else 0
    if code in {32710, 32714}:
        return "tree_pine"
    if code == 32713:
        return "tree_deciduous"
    return "tree_mixed"


def inside_any(point, geometries):
    return any(geometry.Contains(point) for geometry in geometries)


def material_at(point, layers):
    priority = (
        ("yard", 100), ("rock", 90), ("wetland", 80),
        ("agriculture", 70), ("meadow", 60), ("lawn", 50),
        ("cemetery", 50), ("reed", 40),
    )
    for name, _ in priority:
        if inside_any(point, layers.get(name, [])):
            return name
    return "forest" if inside_any(point, layers.get("forest", [])) else "exposed"


def polygon_for_point(point, layers, name):
    return inside_any(point, layers.get(name, []))


def write_vpl(path, bounds, placements, cell_size=25.0):
    """Write VPL1 records grouped into coarse viewer streaming cells.

    Each placement is:
        asset key, east offset, south offset, ground elevation,
        physical height, width multiplier, yaw radians, tint, priority.

    The 25 m cells are an exporter/storage grouping. The viewer can load one
    terrain tile and its complete vegetation package rather than opening one
    file per plant.
    """
    xmin, ymin, xmax, ymax = bounds
    cells = defaultdict(list)
    for placement in placements:
        east, south = placement[1], placement[2]
        cells[
            int((east - xmin) // cell_size),
            int((south - ymin) // cell_size),
        ].append(placement)
    with path.open("wb") as handle:
        handle.write(b"VPL1")
        handle.write(struct.pack("<2df2I", xmin, ymax, cell_size, len(cells), len(placements)))
        for (column, row), members in sorted(cells.items()):
            handle.write(struct.pack("<3I", column, row, len(members)))
            for placement in members:
                key = placement[0].encode("utf-8")
                handle.write(struct.pack("<64s8f", key, *placement[1:]))


def generate_tile(tile, pointcloud_files, layers, exclusions, forest_points, forest_codes, output):
    bounds = tile["bounds"]
    xmin, ymin, xmax, ymax = bounds
    tile_box = box(*bounds)
    # Convert OGR geometries to Shapely once per tile for inexpensive queries.
    from shapely import from_wkt
    local_layers = {
        name: [
            from_wkt(geometry.ExportToWkt())
            for geometry in geometries
            if from_wkt(geometry.ExportToWkt()).intersects(tile_box)
        ]
        for name, geometries in layers.items()
    }
    local_exclusions = [
        from_wkt(geometry.ExportToWkt())
        for geometry in exclusions
        if from_wkt(geometry.ExportToWkt()).intersects(tile_box)
    ]
    points = load_points(pointcloud_files, bounds)
    if len(points):
        from shapely.ops import unary_union
        local_layers["forest"] = [
            unary_union([
                Point(float(x), float(y)).buffer(12.0)
                for x, y in points[::max(1, len(points) // 5000)]
            ])
        ]
    else:
        local_layers["forest"] = []
    placements = []
    with rasterio.open(tile["dem"]) as dem:
        for x, y, z in points:
            point = Point(float(x), float(y))
            if any(point.intersects(exclusion) for exclusion in local_exclusions):
                continue
            ground = sample_dem(dem, float(x), float(y))
            if ground is None:
                continue
            height = float(z - ground)
            if not CONFIG["pointcloud_height_floor"] <= height <= CONFIG["pointcloud_height_ceiling"]:
                continue
            species = "tree_mixed"
            if forest_points:
                nearest = min(
                    range(len(forest_points)),
                    key=lambda index: forest_points[index].Distance(
                        ogr.CreateGeometryFromWkt(f"POINT ({x} {y})")
                    ),
                )
                species = tree_species(forest_codes[nearest])
            placements.append((
                species, float(x - xmin), float(ymax - y), float(ground),
                height, 1.0, 0.0, 1.0, 1.0,
            ))

        rng = stable_rng(CONFIG["placement_seed"], tile["name"])
        for name, spacing, asset in (
            ("meadow", CONFIG["grass_spacing"], "grass_meadow"),
            ("lawn", CONFIG["grass_spacing"], "grass_lawn"),
            ("cemetery", CONFIG["grass_spacing"], "grass_lawn"),
            ("wetland", CONFIG["grass_spacing"], "reed_wetland"),
            ("rock", CONFIG["rock_spacing"], "rock_moss"),
            ("agriculture", 1.5, "wheat"),
        ):
            for x in np.arange(xmin, xmax, spacing):
                for y in np.arange(ymin, ymax, spacing):
                    point = Point(float(x), float(y))
                    if not polygon_for_point(point, local_layers, name):
                        continue
                    if any(point.intersects(exclusion) for exclusion in local_exclusions):
                        continue
                    ground = sample_dem(dem, float(x), float(y))
                    if ground is None or rng.random() > 0.55:
                        continue
                    height = {
                        "grass_meadow": 0.45,
                        "grass_lawn": 0.30,
                        "reed_wetland": 0.9,
                        "rock_moss": 1.2,
                        "wheat": 0.75,
                    }[asset]
                    placements.append((
                        asset, float(x - xmin), float(ymax - y),
                        float(ground), height, float(rng.uniform(0.8, 1.2)),
                        float(rng.uniform(0, math.tau)), 1.0, 0.5,
                    ))

        for x in np.arange(xmin, xmax, 4.0):
            for y in np.arange(ymin, ymax, 4.0):
                point = Point(float(x), float(y))
                if material_at(point, local_layers) != "exposed":
                    continue
                if any(point.intersects(exclusion) for exclusion in local_exclusions):
                    continue
                ground = sample_dem(dem, float(x), float(y))
                if ground is not None and rng.random() < 0.18:
                    placements.append((
                        "grass_exposed", float(x - xmin), float(ymax - y),
                        float(ground), 0.3, float(rng.uniform(0.8, 1.2)),
                        float(rng.uniform(0, math.tau)), 1.0, 0.2,
                    ))

        for x in np.arange(xmin, xmax, CONFIG["bush_spacing"]):
            for y in np.arange(ymin, ymax, CONFIG["bush_spacing"]):
                point = Point(float(x), float(y))
                if not polygon_for_point(point, local_layers, "forest"):
                    continue
                if any(point.intersects(exclusion) for exclusion in local_exclusions):
                    continue
                ground = sample_dem(dem, float(x), float(y))
                if ground is not None and rng.random() < 0.35:
                    placements.append((
                        "bush_leafy", float(x - xmin), float(ymax - y),
                        float(ground), float(rng.uniform(1.0, 2.5)),
                        float(rng.uniform(0.9, 1.2)),
                        float(rng.uniform(0, math.tau)), 1.0, 0.4,
                    ))

    placements_path = output / "placements" / f"{tile['name']}.vpl"
    write_vpl(placements_path, bounds, placements)

    resolution = CONFIG["grid_resolution"]
    columns = max(1, math.ceil((xmax - xmin) / resolution))
    rows = max(1, math.ceil((ymax - ymin) / resolution))
    material_grid = []
    for row in range(rows):
        values = []
        for column in range(columns):
            point = Point(
                xmin + (column + 0.5) * resolution,
                ymin + (row + 0.5) * resolution,
            )
            values.append(material_at(point, local_layers))
        material_grid.append(values)
    (output / "landcover" / f"{tile['name']}.json").write_text(
        json.dumps({
            "tile": tile["name"],
            "bounds": bounds,
            "resolution": resolution,
            "materials": material_grid,
            "placement_count": len(placements),
        }),
        encoding="utf-8",
    )
    return len(placements)


def main():
    parser = argparse.ArgumentParser(
        description="Generate land-cover maps and vegetation placements.",
    )
    parser.add_argument("--dem", required=True)
    parser.add_argument("--mtk", required=True)
    parser.add_argument("--pointcloud", required=True)
    parser.add_argument("--buildings")
    parser.add_argument("--roads")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    dem = Path(args.dem).resolve()
    mtk = Path(args.mtk).resolve()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Output must be empty or not yet exist.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "placements").mkdir()
    (output / "landcover").mkdir()

    tiles = read_terrain_index(dem)
    pointcloud_files = read_pointcloud_files(Path(args.pointcloud))
    layers, exclusions, forest_points, forest_codes = load_sources(
        mtk,
        args.buildings,
        args.roads,
    )
    print(f"Processing {len(tiles)} tiles and {len(pointcloud_files)} point-cloud files.", flush=True)
    total = 0
    manifest_tiles = []
    for index, tile in enumerate(tiles, 1):
        count = generate_tile(
            tile, pointcloud_files, layers, exclusions,
            forest_points, forest_codes, output,
        )
        total += count
        print(f"Tile {index}/{len(tiles)}: {count} placements.", flush=True)
        manifest_tiles.append({
            "terrain_filename": tile["name"],
            "placement_filename": f"placements/{tile['name']}.vpl",
            "landcover_filename": f"landcover/{tile['name']}.json",
        })

    (output / "landcover_manifest.json").write_text(
        json.dumps({
            "format": "landcover_manifest_v1",
            "crs": "EPSG:3067",
            "placement_format": "VPL1",
            "tiles": manifest_tiles,
            "config": CONFIG,
            "total_placements": total,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Finished: {total} placements in {output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
