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
import csv
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
    "building_buffer": 3.0,
    # Keep these road envelopes aligned with generate_road_network.py.
    # The exclusion radius is half the generated surface width plus its
    # generated shoulder width.
    "road_classes": {
        12111: {"width": 13.0, "shoulder": 2.0},
        12112: {"width": 13.0, "shoulder": 2.0},
        12121: {"width": 9.0, "shoulder": 1.2},
        12122: {"width": 8.0, "shoulder": 1.0},
        12131: {"width": 6.5, "shoulder": 0.5},
        12141: {"width": 4.0, "shoulder": 0.4},
        12313: {"width": 2.5, "shoulder": 0.25},
        12314: {"width": 2.5, "shoulder": 0.20},
        12316: {"width": 1.8, "shoulder": 0.20},
        "fallback": {"width": 4.0, "shoulder": 0.4},
    },
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
GENERATOR_VERSION = 2
MANIFEST_NAME = "landcover_manifest.json"


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


def road_exclusion_radius(luokka):
    """Return the generated road envelope for one MTK LUOKKA value."""
    try:
        code = int(luokka)
    except (TypeError, ValueError):
        code = None
    settings = CONFIG["road_classes"].get(
        code,
        CONFIG["road_classes"]["fallback"],
    )
    return settings["width"] / 2.0 + settings["shoulder"]


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
    """Read the downloader output structure and return manifest-listed files.

    ``download_oldest_pointcloud.py`` writes a manifest at the download root
    and stores files in its flat ``pointclouds/`` directory. The manifest is
    the authoritative list because it records the acquisition year and the
    exact local path. A recursive scan remains a compatibility fallback for
    older folders that do not contain the manifest.
    """
    folder = Path(folder)
    manifest = folder / "download_manifest.csv"

    if manifest.is_file():
        files = []
        years = defaultdict(int)
        with manifest.open(
            "r", newline="", encoding="utf-8-sig"
        ) as handle:
            reader = csv.DictReader(handle)
            required = {"year", "local_path"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(
                    f"Point-cloud manifest is missing {sorted(required)}: "
                    f"{manifest}"
                )

            for row in reader:
                raw_path = (row.get("local_path") or "").strip()
                if not raw_path:
                    raise ValueError(
                        f"Point-cloud manifest has an empty local_path: "
                        f"{manifest}"
                    )

                candidate = Path(raw_path)
                if not candidate.is_absolute():
                    candidate = folder / candidate
                if not candidate.is_file():
                    fallback = folder / "pointclouds" / candidate.name
                    if fallback.is_file():
                        candidate = fallback
                    else:
                        raise ValueError(
                            f"Manifest-listed point cloud does not exist: "
                            f"{candidate}"
                        )

                if candidate.suffix.lower() not in {".las", ".laz"}:
                    raise ValueError(
                        f"Manifest local_path is not LAS/LAZ: {candidate}"
                    )

                files.append(candidate.resolve())
                years[str(row["year"])] += 1

        if not files:
            raise ValueError(f"Point-cloud manifest contains no files: {manifest}")

        print(
            f"Point-cloud manifest: {len(files)} files "
            f"({', '.join(f'{year}: {count}' for year, count in sorted(years.items()))})",
            flush=True,
        )
        return sorted(set(files))

    files = sorted(
        path.resolve() for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in {".las", ".laz"}
    )
    if not files:
        raise ValueError(
            f"No download_manifest.csv or LAS/LAZ files found under {folder}. "
            "Expected the output of download_oldest_pointcloud.py."
        )
    print(
        f"Point-cloud manifest not found; compatibility scan found "
        f"{len(files)} LAS/LAZ files under {folder}.",
        flush=True,
    )
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
            print(
                f"  Tile bounds intersect point cloud: {filename.name}",
                flush=True,
            )
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
            exclusions.extend(
                geometry.Buffer(CONFIG["building_buffer"])
                for geometry in geometries
            )
    if roads:
        path = Path(roads)
        if path.is_file():
            try:
                geometries, classes = read_geometries(
                    path,
                    "tieviiva",
                    "LUOKKA",
                )
                exclusions.extend(
                    geometry.Buffer(road_exclusion_radius(luokka))
                    for geometry, luokka in zip(geometries, classes)
                )
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


def atomic_replace(path, writer):
    """Write a generated file through a temporary sibling, then replace it."""
    temporary = path.with_name(path.name + ".tmp")
    try:
        writer(temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def source_signature(path):
    """Return stable metadata for an input file or relevant source folder."""
    path = Path(path).resolve()
    if path.is_file():
        stat = path.stat()
        return {
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    entries = []
    if path.is_dir():
        for candidate in sorted(path.rglob("*")):
            if not candidate.is_file():
                continue
            if (
                candidate.name == "tile_manifest.json"
                or candidate.name == "download_manifest.csv"
                or candidate.suffix.lower() in {".gpkg", ".las", ".laz"}
            ):
                stat = candidate.stat()
                entries.append({
                    "path": str(candidate.relative_to(path)),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                })
    return {"path": str(path), "entries": entries}


def run_signature(dem, mtk, pointcloud, buildings, roads):
    payload = {
        "generator_version": GENERATOR_VERSION,
        "config": CONFIG,
        "dem": source_signature(dem),
        "mtk": source_signature(mtk),
        "pointcloud": source_signature(pointcloud),
        "buildings": source_signature(buildings) if buildings else None,
        "roads": source_signature(roads) if roads else None,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_resume_state(path, signature, force_rebuild):
    """Load completed tile records that are safe to reuse."""
    if force_rebuild or not path.is_file():
        return {}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if manifest.get("run_signature") != signature:
        return {}
    return {
        tile["terrain_filename"]: tile
        for tile in manifest.get("tiles", [])
        if tile.get("status") == "complete"
    }


def write_manifest(path, signature, tiles, total):
    payload = {
        "format": "landcover_manifest_v1",
        "generator_version": GENERATOR_VERSION,
        "run_signature": signature,
        "crs": "EPSG:3067",
        "placement_format": "VPL1",
        "tiles": tiles,
        "config": CONFIG,
        "total_placements": total,
    }
    atomic_replace(
        path,
        lambda temporary: temporary.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        ),
    )


def reusable_tile(output, tile, record):
    """Check that the recorded tile outputs still exist and are readable."""
    placement = output / record["placement_filename"]
    landcover = output / record["landcover_filename"]
    if not placement.is_file() or placement.stat().st_size < 4:
        return False
    if not landcover.is_file():
        return False
    try:
        data = json.loads(landcover.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        data.get("tile") == tile["name"]
        and data.get("placement_count") == record.get("placement_count")
    )


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
    atomic_replace(
        placements_path,
        lambda temporary: write_vpl(temporary, bounds, placements),
    )

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
    landcover_path = output / "landcover" / f"{tile['name']}.json"
    atomic_replace(
        landcover_path,
        lambda temporary: temporary.write_text(
            json.dumps({
            "tile": tile["name"],
            "bounds": bounds,
            "resolution": resolution,
            "materials": material_grid,
            "placement_count": len(placements),
            }),
            encoding="utf-8",
        ),
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
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Regenerate every tile instead of resuming completed tiles.",
    )
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    dem = Path(args.dem).resolve()
    mtk = Path(args.mtk).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "placements").mkdir(exist_ok=True)
    (output / "landcover").mkdir(exist_ok=True)

    tiles = read_terrain_index(dem)
    pointcloud = Path(args.pointcloud).resolve()
    pointcloud_files = read_pointcloud_files(pointcloud)
    layers, exclusions, forest_points, forest_codes = load_sources(
        mtk,
        args.buildings,
        args.roads,
    )
    signature = run_signature(
        dem,
        mtk,
        pointcloud,
        args.buildings,
        args.roads,
    )
    manifest_path = output / MANIFEST_NAME
    resume = read_resume_state(
        manifest_path,
        signature,
        args.force_rebuild,
    )
    if resume:
        print(
            f"Resuming: {len(resume)} completed tiles are available for reuse.",
            flush=True,
        )
    elif manifest_path.is_file() and not args.force_rebuild:
        print(
            "Existing manifest does not match the current inputs/config; "
            "regenerating tiles.",
            flush=True,
        )
    print(f"Processing {len(tiles)} tiles and {len(pointcloud_files)} point-cloud files.", flush=True)
    total = 0
    manifest_tiles = []
    for index, tile in enumerate(tiles, 1):
        record = resume.get(tile["name"])
        if record and reusable_tile(output, tile, record):
            count = record["placement_count"]
            print(
                f"Tile {index}/{len(tiles)}: reusing {count} placements.",
                flush=True,
            )
        else:
            count = generate_tile(
                tile, pointcloud_files, layers, exclusions,
                forest_points, forest_codes, output,
            )
            print(f"Tile {index}/{len(tiles)}: {count} placements.", flush=True)
        total += count
        manifest_tiles.append({
            "terrain_filename": tile["name"],
            "placement_filename": f"placements/{tile['name']}.vpl",
            "landcover_filename": f"landcover/{tile['name']}.json",
            "placement_count": count,
            "status": "complete",
        })
        write_manifest(manifest_path, signature, manifest_tiles, total)

    print(f"Finished: {total} placements in {output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
