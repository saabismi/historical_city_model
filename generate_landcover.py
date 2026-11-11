#!/usr/bin/env python3
"""
Generate renderer-neutral land-cover maps and VPL1 vegetation placements.

Inputs:
    - DEM tiles described by tile_manifest.json.
    - MTK vector layers in EPSG:3067.
    - Independent LAS/LAZ point clouds.
    - Optional building and road vector files.

Outputs:
    - placements/<tile>.vpl
    - landcover/<tile>.json
    - landcover_manifest.json

Processing:
    - Geometry operations use Shapely after vectors are read through OGR.
    - Spatial indexes accelerate polygon and nearest-feature queries.
    - Each worker loads its vector sources once.
    - Each tile's DEM band is read once.
    - Only the parent process updates the resume manifest.

Important:
    Tree placement behavior is retained from the original generator:
    one tree placement is emitted per accepted LiDAR return, not per
    individually detected tree. CONFIG["tree_spacing"] remains unused.

Requirements:
    GDAL/OGR, rasterio, numpy, Shapely >= 2.0, laspy, and lazrs.

Windows example:
    python generate_landcover.py ^
        --dem espoo2008_dem_tiled ^
        --mtk MTK_2009_clipped ^
        --pointcloud espoo1965_clouds ^
        --workers 4 ^
        -o espoo2008_landcover_output

Use --workers 1 for sequential processing.
"""

import argparse
import csv
import hashlib
import json
import math
import multiprocessing
import os
import struct
import sys
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import laspy
import numpy as np
import rasterio
import shapely
from osgeo import ogr, osr
from shapely import from_wkb, make_valid
from shapely.geometry import Point, box
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.strtree import STRtree


CONFIG = {
    # Distances are metres. Placement spacings describe approximate lattices.
    "grid_resolution": 10.0,
    "placement_seed": 12345,
    "tree_classes": (3, 4, 5),
    "tree_spacing": 5.5,
    "grass_spacing": 2.5,
    "bush_spacing": 7.0,
    "rock_spacing": 8.0,
    "building_buffer": 3.0,
    # Exclusion radius = half the road surface width plus shoulder width.
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

# Incremented because geometry handling and VPL cell indexing were corrected.
# Older manifests will therefore not silently reuse incompatible outputs.
GENERATOR_VERSION = 3
MANIFEST_NAME = "landcover_manifest.json"

VPL_HEADER_FORMAT = "<2df2I"
VPL_CELL_FORMAT = "<3I"
VPL_RECORD_FORMAT = "<64s8f"

# This state belongs to each individual process, never shared between workers.
_WORKER_STATE = None


def check_dependencies():
    """Require Shapely 2.x spatial-index semantics used by this script."""
    major = int(shapely.__version__.split(".", 1)[0])
    if major < 2:
        raise RuntimeError(
            f"Shapely >= 2.0 is required; installed version is "
            f"{shapely.__version__}."
        )


def stable_rng(*parts):
    """Create a deterministic random generator from stable input labels."""
    digest = hashlib.blake2b(
        ":".join(map(str, parts)).encode("utf-8"),
        digest_size=8,
    ).digest()
    return np.random.default_rng(int.from_bytes(digest, "little"))


def normalize_crs(crs):
    """Clone an OGR CRS and use traditional GIS axis ordering."""
    result = crs.Clone()
    result.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return result


def _epsg_3067():
    """Construct the expected projected coordinate reference system."""
    crs = osr.SpatialReference()
    crs.ImportFromEPSG(EPSG)
    crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return crs


def open_layer(path, layer_name):
    """Open an OGR vector layer and require EPSG:3067.

    Return the dataset and layer together so the dataset remains alive while
    its layer is being read.
    """
    dataset = ogr.Open(str(path), 0)
    if dataset is None:
        raise ValueError(f"Cannot open vector source: {path}")

    layer = dataset.GetLayerByName(layer_name)
    if layer is None:
        raise ValueError(f"Layer {layer_name!r} not found in {path}")

    source_crs = layer.GetSpatialRef()
    if source_crs is None:
        raise ValueError(f"Layer has no CRS: {path}:{layer_name}")

    if not normalize_crs(source_crs).IsSame(_epsg_3067()):
        raise ValueError(f"Expected EPSG:3067: {path}:{layer_name}")

    return dataset, layer


def available_layer_names(path):
    """Return the layer names present in an existing vector dataset."""
    dataset = ogr.Open(str(path), 0)
    if dataset is None:
        raise ValueError(f"Cannot open vector source: {path}")

    try:
        return {
            dataset.GetLayerByIndex(index).GetName()
            for index in range(dataset.GetLayerCount())
        }
    finally:
        dataset = None


def read_geometries(path, layer_name, field=None):
    """Read OGR features and immediately convert their geometry to Shapely.

    Return parallel lists of geometries and optional attribute values.
    Invalid geometries are repaired before they are used in spatial queries.
    No borrowed OGR geometry references escape this function.
    """
    dataset, layer = open_layer(path, layer_name)
    geometries = []
    values = []

    try:
        if field and layer.GetLayerDefn().GetFieldIndex(field) < 0:
            raise ValueError(
                f"Field {field!r} not found in {path}:{layer_name}"
            )

        for feature in layer:
            geometry = feature.GetGeometryRef()
            if geometry is None or geometry.IsEmpty():
                continue

            shape = from_wkb(bytes(geometry.ExportToWkb()))
            if not shape.is_valid:
                shape = make_valid(shape)

            if shape is None or shape.is_empty:
                continue

            geometries.append(shape)
            values.append(feature.GetField(field) if field else None)

        return geometries, values
    finally:
        dataset = None


class GeometryIndex:
    """Store Shapely geometries with an optional STRtree spatial index.

    Shapely 2.x STRtree methods return indices into the stored geometry list.
    Empty collections are handled without constructing an index.
    """

    def __init__(self, geometries):
        """Retain non-empty geometries and build their spatial index."""
        self.geometries = [
            geometry
            for geometry in geometries
            if geometry is not None and not geometry.is_empty
        ]
        self.tree = (
            STRtree(self.geometries)
            if self.geometries
            else None
        )

    def intersecting(self, geometry):
        """Return stored geometries intersecting the supplied geometry."""
        if self.tree is None:
            return []

        indices = self.tree.query(geometry, predicate="intersects")
        return [self.geometries[int(index)] for index in indices]

    def contains_point(self, point):
        """Test strict point containment, retaining original boundary rules.

        The query is point.within(stored_polygon), equivalent to the
        stored polygon's lowercase Shapely contains(point) operation.
        """
        if self.tree is None:
            return False

        return len(self.tree.query(point, predicate="within")) > 0

    def intersects_point(self, point):
        """Test whether any stored geometry touches or contains the point."""
        if self.tree is None:
            return False

        return len(self.tree.query(point, predicate="intersects")) > 0


class DemSampler:
    """Read one DEM band once and serve scalar ground-elevation samples."""

    def __init__(self, dataset):
        """Cache the band, validity mask, and inverse affine transform."""
        band = dataset.read(1, masked=True)
        self.values = np.asarray(band.data)
        self.mask = np.ma.getmaskarray(band)
        self.inverse_transform = ~dataset.transform
        self.height = dataset.height
        self.width = dataset.width

    def sample(self, x, y):
        """Return the containing pixel's elevation, or None if invalid."""
        column_float, row_float = self.inverse_transform * (x, y)
        row = math.floor(row_float)
        column = math.floor(column_float)

        if (
            row < 0
            or row >= self.height
            or column < 0
            or column >= self.width
        ):
            return None

        if self.mask[row, column]:
            return None

        value = float(self.values[row, column])
        return value if math.isfinite(value) else None


def road_exclusion_radius(luokka):
    """Return half the configured road width plus its shoulder width."""
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
    """Read DEM-backed tiles from the terrain manifest and validate bounds."""
    manifest_path = folder / "tile_manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    entries = []
    names = set()

    for tile in manifest["tiles"]:
        bounds = tile.get("core_bounds")
        if not bounds or not tile.get("dem"):
            continue

        name = tile["tile"]
        if name in names:
            raise ValueError(f"Duplicate terrain tile name: {name}")
        names.add(name)

        if len(bounds) != 4:
            raise ValueError(f"Invalid core_bounds for tile {name}")

        bounds = tuple(float(value) for value in bounds)
        xmin, ymin, xmax, ymax = bounds
        if (
            not all(math.isfinite(value) for value in bounds)
            or xmax <= xmin
            or ymax <= ymin
        ):
            raise ValueError(f"Invalid core_bounds for tile {name}: {bounds}")

        dem_path = folder / tile["dem"]
        if not dem_path.is_file():
            raise ValueError(f"DEM does not exist: {dem_path}")

        entries.append({
            "name": name,
            "bounds": bounds,
            "dem": dem_path,
        })

    if not entries:
        raise ValueError("No DEM-backed tiles found in tile_manifest.json")

    return entries


def read_pointcloud_files(folder):
    """Resolve manifest-listed LAS/LAZ files, with a legacy scan fallback."""
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
                    f"Point-cloud manifest requires columns "
                    f"{sorted(required)}: {manifest}"
                )

            for row in reader:
                raw_path = (row.get("local_path") or "").strip()
                if not raw_path:
                    raise ValueError(
                        f"Empty local_path in point-cloud manifest: {manifest}"
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
            raise ValueError(
                f"Point-cloud manifest contains no files: {manifest}"
            )

        year_summary = ", ".join(
            f"{year}: {count}"
            for year, count in sorted(years.items())
        )
        print(
            f"Point-cloud manifest: {len(files)} files ({year_summary})",
            flush=True,
        )
        return sorted(set(files))

    files = sorted(
        path.resolve()
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in {".las", ".laz"}
    )

    if not files:
        raise ValueError(
            f"No download_manifest.csv or LAS/LAZ files found under {folder}."
        )

    print(
        f"Point-cloud manifest not found; compatibility scan found "
        f"{len(files)} LAS/LAZ files.",
        flush=True,
    )
    return files


def index_pointcloud_files(files):
    """Read file bounds once per worker to avoid repeated header-only opens."""
    entries = []

    for filename in files:
        with laspy.open(filename) as reader:
            entries.append({
                "path": Path(filename),
                "bounds": (
                    float(reader.header.mins[0]),
                    float(reader.header.mins[1]),
                    float(reader.header.maxs[0]),
                    float(reader.header.maxs[1]),
                ),
            })

    return entries


def load_points(file_entries, bounds, tile_name):
    """Load classified vegetation XYZ points intersecting one terrain tile."""
    xmin, ymin, xmax, ymax = bounds
    chunks = []

    for entry in file_entries:
        cloud_xmin, cloud_ymin, cloud_xmax, cloud_ymax = entry["bounds"]

        if (
            cloud_xmax < xmin
            or cloud_xmin > xmax
            or cloud_ymax < ymin
            or cloud_ymin > ymax
        ):
            continue

        filename = entry["path"]
        print(
            f"  [{tile_name}] Point cloud: {filename.name}",
            flush=True,
        )

        with laspy.open(filename) as reader:
            for block in reader.chunk_iterator(500_000):
                classification = np.asarray(block.classification)
                keep = np.isin(classification, CONFIG["tree_classes"])

                if not np.any(keep):
                    continue

                x = np.asarray(block.x)
                y = np.asarray(block.y)
                keep &= (x >= xmin) & (x <= xmax)
                keep &= (y >= ymin) & (y <= ymax)

                if np.any(keep):
                    chunks.append(
                        np.column_stack((
                            x[keep],
                            y[keep],
                            np.asarray(block.z)[keep],
                        ))
                    )

    if not chunks:
        return np.empty((0, 3), dtype=np.float64)

    return np.concatenate(chunks)


def load_sources(mtk, buildings, roads):
    """Load polygon layers, exclusion buffers, and forest classifications.

    All returned geometries are Shapely objects. Missing optional MTK
    classification layers are allowed, but malformed present layers fail
    explicitly rather than silently bypassing CRS validation.
    """
    layers = defaultdict(list)

    for filename in (
        mtk / "mtk-muut_2009_clipped.gpkg",
        mtk / "mtk-vakavesi_2009_clipped.gpkg",
    ):
        if not filename.is_file():
            continue

        names = available_layer_names(filename)
        for name in sorted(names):
            if name in CONFIG["polygon_layers"]:
                geometries, _ = read_geometries(filename, name)
                layers[CONFIG["polygon_layers"][name]].extend(geometries)

    forest_points = []
    forest_codes = []
    forest_file = mtk / "mtk-muut_2009_clipped.gpkg"
    forest_layer = CONFIG["forest_classification_layer"]

    if (
        forest_file.is_file()
        and forest_layer in available_layer_names(forest_file)
    ):
        forest_points, forest_codes = read_geometries(
            forest_file,
            forest_layer,
            CONFIG["forest_classification_field"],
        )

    exclusions = []

    if buildings:
        path = Path(buildings)
        if not path.is_file():
            raise ValueError(f"Building source does not exist: {path}")

        geometries, _ = read_geometries(path, "rakennus")
        exclusions.extend(
            geometry.buffer(CONFIG["building_buffer"])
            for geometry in geometries
        )

    if roads:
        path = Path(roads)
        if not path.is_file():
            raise ValueError(f"Road source does not exist: {path}")

        geometries, classes = read_geometries(path, "tieviiva", "LUOKKA")
        exclusions.extend(
            geometry.buffer(road_exclusion_radius(luokka))
            for geometry, luokka in zip(geometries, classes)
        )

    return layers, exclusions, forest_points, forest_codes


def tree_species(code):
    """Map an MTK forest classification to the original tree asset key."""
    try:
        code = int(code) if code is not None else 0
    except (TypeError, ValueError):
        code = 0

    if code in {32710, 32714}:
        return "tree_pine"
    if code == 32713:
        return "tree_deciduous"
    return "tree_mixed"


def inside_any(point, geometry_index):
    """Test polygon containment using Shapely, never OGR Contains()."""
    return (
        geometry_index is not None
        and geometry_index.contains_point(point)
    )


def material_at(point, layers):
    """Return the highest-priority land-cover category at one point."""
    for name in (
        "yard",
        "rock",
        "wetland",
        "agriculture",
        "meadow",
        "lawn",
        "cemetery",
        "reed",
    ):
        if inside_any(point, layers.get(name)):
            return name

    if inside_any(point, layers.get("forest")):
        return "forest"

    return "exposed"


def polygon_for_point(point, layers, name):
    """Test whether a point is strictly inside a named land-cover category."""
    return inside_any(point, layers.get(name))


def write_vpl(path, bounds, placements, cell_size=25.0):
    """Write the original VPL1 binary layout using correct local cells.

    Each placement contains:
        asset key, east offset, south offset, ground elevation,
        physical height, width multiplier, yaw, tint, and priority.

    Offsets are already relative to the tile's northwest corner:
        east = x - xmin
        south = ymax - y

    The header stores the projected northwest origin. Cell indices are
    calculated from local offsets without subtracting that origin again.
    """
    xmin, ymin, xmax, ymax = bounds

    if not math.isfinite(cell_size) or cell_size <= 0:
        raise ValueError("VPL1 cell_size must be finite and positive")

    cells = defaultdict(list)

    for placement in placements:
        if len(placement) != 9:
            raise ValueError(f"Invalid VPL1 placement: {placement!r}")

        values = tuple(float(value) for value in placement[1:])
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite VPL1 placement: {placement!r}")

        east, south = values[:2]
        if east < 0 or south < 0:
            raise ValueError(
                f"Negative VPL1 offsets: east={east}, south={south}"
            )

        column = math.floor(east / cell_size)
        row = math.floor(south / cell_size)
        cells[column, row].append(placement)

    with path.open("wb") as handle:
        handle.write(b"VPL1")
        handle.write(
            struct.pack(
                VPL_HEADER_FORMAT,
                xmin,
                ymax,
                cell_size,
                len(cells),
                len(placements),
            )
        )

        for (column, row), members in sorted(cells.items()):
            handle.write(
                struct.pack(VPL_CELL_FORMAT, column, row, len(members))
            )

            for placement in members:
                key = placement[0].encode("utf-8")
                if len(key) > 64:
                    raise ValueError(
                        f"VPL1 asset key exceeds 64 bytes: {placement[0]!r}"
                    )

                handle.write(
                    struct.pack(
                        VPL_RECORD_FORMAT,
                        key,
                        *placement[1:],
                    )
                )


def atomic_replace(path, writer):
    """Write through a process-specific temporary sibling, then replace."""
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")

    try:
        writer(temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def source_signature(path):
    """Return input path, size, and modification-time metadata.

    Directory signatures include vector, point-cloud, and common DEM files,
    along with the two input manifests.
    """
    path = Path(path).resolve()

    if path.is_file():
        stat = path.stat()
        return {
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    if not path.is_dir():
        raise ValueError(f"Input path does not exist: {path}")

    relevant_suffixes = {
        ".gpkg", ".las", ".laz",
        ".tif", ".tiff", ".vrt", ".img", ".bil", ".asc",
    }

    entries = []
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue

        if (
            candidate.name in {
                "tile_manifest.json",
                "download_manifest.csv",
            }
            or candidate.suffix.lower() in relevant_suffixes
        ):
            stat = candidate.stat()
            entries.append({
                "path": str(candidate.relative_to(path)),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            })

    return {"path": str(path), "entries": entries}


def run_signature(dem, mtk, pointcloud, buildings, roads):
    """Hash generation settings and inputs; worker count is intentionally absent."""
    payload = {
        "generator_version": GENERATOR_VERSION,
        # Normalize mixed integer/string road keys before sorting JSON keys.
        "config": json.loads(json.dumps(CONFIG)),
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
    """Return completed tile records only when the run signature matches."""
    if force_rebuild or not path.is_file():
        return {}

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(manifest, dict):
        return {}

    if manifest.get("run_signature") != signature:
        return {}

    records = {}
    for tile in manifest.get("tiles", []):
        if (
            isinstance(tile, dict)
            and tile.get("status") == "complete"
            and tile.get("terrain_filename")
        ):
            records[tile["terrain_filename"]] = tile

    return records


def write_manifest(path, signature, tiles, total):
    """Atomically checkpoint completed tiles and their placement total."""
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
    """Validate a saved material map and the VPL1 header/file length."""
    try:
        placement = output / record["placement_filename"]
        landcover = output / record["landcover_filename"]
        count = record["placement_count"]

        if not isinstance(count, int) or count < 0:
            return False

        if not placement.is_file() or not landcover.is_file():
            return False

        header_size = struct.calcsize(VPL_HEADER_FORMAT)

        with placement.open("rb") as handle:
            if handle.read(4) != b"VPL1":
                return False

            header = handle.read(header_size)
            if len(header) != header_size:
                return False

            origin_x, origin_y, cell_size, cell_count, total = struct.unpack(
                VPL_HEADER_FORMAT,
                header,
            )

        xmin, ymin, xmax, ymax = tile["bounds"]
        if (
            origin_x != xmin
            or origin_y != ymax
            or not math.isfinite(cell_size)
            or cell_size <= 0
            or total != count
            or cell_count > total
        ):
            return False

        expected_size = (
            4
            + header_size
            + cell_count * struct.calcsize(VPL_CELL_FORMAT)
            + total * struct.calcsize(VPL_RECORD_FORMAT)
        )
        if placement.stat().st_size != expected_size:
            return False

        data = json.loads(landcover.read_text(encoding="utf-8"))

        return (
            isinstance(data, dict)
            and data.get("tile") == tile["name"]
            and data.get("placement_count") == count
            and data.get("bounds") == list(tile["bounds"])
        )
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        struct.error,
    ):
        return False


def generate_tile(tile, state):
    """Generate one tile using vector and cloud indexes owned by its worker."""
    bounds = tile["bounds"]
    xmin, ymin, xmax, ymax = bounds
    tile_box = box(*bounds)

    # Sources have already been converted to Shapely once per worker.
    local_layers = {
        name: GeometryIndex(index.intersecting(tile_box))
        for name, index in state["layers"].items()
    }
    local_exclusions = GeometryIndex(
        state["exclusions"].intersecting(tile_box)
    )

    points = load_points(
        state["pointcloud_entries"],
        bounds,
        tile["name"],
    )

    # Preserve the original forest-footprint approximation. Point rows
    # contain XYZ, so explicitly select XY before unpacking.
    if len(points):
        stride = max(1, len(points) // 5000)
        sampled_xy = points[::stride, :2]
        forest_shape = unary_union([
            Point(float(x), float(y)).buffer(12.0)
            for x, y in sampled_xy
        ])
        forest_shapes = [forest_shape]
        prepared_forest = prep(forest_shape)
    else:
        forest_shapes = []
        prepared_forest = None

    local_layers["forest"] = GeometryIndex(forest_shapes)
    placements = []

    with rasterio.open(tile["dem"]) as dataset:
        if dataset.crs is None or dataset.crs != rasterio.crs.CRS.from_epsg(EPSG):
            raise ValueError(
                f"Expected EPSG:3067 DEM for {tile['name']}: {tile['dem']}"
            )
        dem = DemSampler(dataset)

    forest_tree = state["forest_tree"]
    forest_codes = state["forest_codes"]

    # Retain one tree placement per accepted vegetation-class LiDAR return.
    for x, y, z in points:
        x = float(x)
        y = float(y)
        point = Point(x, y)

        if local_exclusions.intersects_point(point):
            continue

        ground = dem.sample(x, y)
        if ground is None:
            continue

        height = float(z - ground)
        if not (
            CONFIG["pointcloud_height_floor"]
            <= height
            <= CONFIG["pointcloud_height_ceiling"]
        ):
            continue

        species = "tree_mixed"
        if forest_tree is not None:
            nearest_index = forest_tree.nearest(point)
            if nearest_index is not None:
                species = tree_species(forest_codes[int(nearest_index)])

        placements.append((
            species,
            x - xmin,
            ymax - y,
            ground,
            height,
            1.0,
            0.0,
            1.0,
            1.0,
        ))

    # Per-tile seed makes random placements independent of worker scheduling.
    rng = stable_rng(CONFIG["placement_seed"], tile["name"])

    asset_heights = {
        "grass_meadow": 0.45,
        "grass_lawn": 0.30,
        "reed_wetland": 0.9,
        "rock_moss": 1.2,
        "wheat": 0.75,
    }

    for name, spacing, asset in (
        ("meadow", CONFIG["grass_spacing"], "grass_meadow"),
        ("lawn", CONFIG["grass_spacing"], "grass_lawn"),
        ("cemetery", CONFIG["grass_spacing"], "grass_lawn"),
        ("wetland", CONFIG["grass_spacing"], "reed_wetland"),
        ("rock", CONFIG["rock_spacing"], "rock_moss"),
        ("agriculture", 1.5, "wheat"),
    ):
        category = local_layers.get(name)

        # Empty categories never consumed RNG values in the original loop.
        if category is None or category.tree is None:
            continue

        for x in np.arange(xmin, xmax, spacing):
            for y in np.arange(ymin, ymax, spacing):
                x = float(x)
                y = float(y)
                point = Point(x, y)

                if not polygon_for_point(point, local_layers, name):
                    continue
                if local_exclusions.intersects_point(point):
                    continue

                ground = dem.sample(x, y)
                if ground is None or rng.random() > 0.55:
                    continue

                placements.append((
                    asset,
                    x - xmin,
                    ymax - y,
                    ground,
                    asset_heights[asset],
                    float(rng.uniform(0.8, 1.2)),
                    float(rng.uniform(0, math.tau)),
                    1.0,
                    0.5,
                ))

    for x in np.arange(xmin, xmax, 4.0):
        for y in np.arange(ymin, ymax, 4.0):
            x = float(x)
            y = float(y)
            point = Point(x, y)

            if material_at(point, local_layers) != "exposed":
                continue
            if local_exclusions.intersects_point(point):
                continue

            ground = dem.sample(x, y)
            if ground is not None and rng.random() < 0.18:
                placements.append((
                    "grass_exposed",
                    x - xmin,
                    ymax - y,
                    ground,
                    0.3,
                    float(rng.uniform(0.8, 1.2)),
                    float(rng.uniform(0, math.tau)),
                    1.0,
                    0.2,
                ))

    if prepared_forest is not None:
        for x in np.arange(xmin, xmax, CONFIG["bush_spacing"]):
            for y in np.arange(ymin, ymax, CONFIG["bush_spacing"]):
                x = float(x)
                y = float(y)
                point = Point(x, y)

                if not prepared_forest.contains(point):
                    continue
                if local_exclusions.intersects_point(point):
                    continue

                ground = dem.sample(x, y)
                if ground is not None and rng.random() < 0.35:
                    placements.append((
                        "bush_leafy",
                        x - xmin,
                        ymax - y,
                        ground,
                        float(rng.uniform(1.0, 2.5)),
                        float(rng.uniform(0.9, 1.2)),
                        float(rng.uniform(0, math.tau)),
                        1.0,
                        0.4,
                    ))

    placements_path = (
        state["output"] / "placements" / f"{tile['name']}.vpl"
    )
    atomic_replace(
        placements_path,
        lambda temporary: write_vpl(temporary, bounds, placements),
    )

    resolution = CONFIG["grid_resolution"]
    columns = max(1, math.ceil((xmax - xmin) / resolution))
    rows = max(1, math.ceil((ymax - ymin) / resolution))

    # Preserve the original material-grid orientation: row zero is south.
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

    landcover_path = (
        state["output"] / "landcover" / f"{tile['name']}.json"
    )
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


def initialize_landcover_worker(
    mtk_folder,
    buildings_path,
    roads_path,
    pointcloud_files,
    output_folder,
):
    """Initialize process-local vector indexes and point-cloud metadata.

    Arguments are serializable paths, not open datasets or OGR geometries.
    The process pool invokes this function once per worker.
    """
    global _WORKER_STATE

    check_dependencies()
    ogr.UseExceptions()

    layers, exclusions, forest_points, forest_codes = load_sources(
        Path(mtk_folder),
        buildings_path,
        roads_path,
    )

    _WORKER_STATE = {
        "layers": {
            name: GeometryIndex(geometries)
            for name, geometries in layers.items()
        },
        "exclusions": GeometryIndex(exclusions),
        "forest_tree": STRtree(forest_points) if forest_points else None,
        "forest_codes": forest_codes,
        "pointcloud_entries": index_pointcloud_files(
            [Path(path) for path in pointcloud_files]
        ),
        "output": Path(output_folder),
    }

    print(
        f"Worker {os.getpid()}: source data and indexes loaded.",
        flush=True,
    )


def generate_tile_in_worker(tile):
    """Generate one tile and return only its placement count to the parent."""
    if _WORKER_STATE is None:
        raise RuntimeError("Land-cover worker has not been initialized")

    return generate_tile(tile, _WORKER_STATE)


def main():
    """Parse arguments, reuse valid outputs, and schedule unfinished tiles."""
    check_dependencies()
    default_workers = min(4, os.cpu_count() or 1)

    parser = argparse.ArgumentParser(
        description="Generate land-cover maps and vegetation placements.",
    )
    parser.add_argument("--dem", required=True)
    parser.add_argument("--mtk", required=True)
    parser.add_argument("--pointcloud", required=True)
    parser.add_argument("--buildings")
    parser.add_argument("--roads")
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=(
            "Tile-generation processes. "
            f"Default: {default_workers}. Use 1 for sequential execution."
        ),
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Regenerate every tile instead of reusing completed tiles.",
    )
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if sys.platform == "win32" and args.workers > 61:
        parser.error("--workers cannot exceed 61 on Windows")

    dem = Path(args.dem).resolve()
    mtk = Path(args.mtk).resolve()
    pointcloud = Path(args.pointcloud).resolve()
    output = Path(args.output).resolve()
    buildings = (
        str(Path(args.buildings).resolve())
        if args.buildings
        else None
    )
    roads = (
        str(Path(args.roads).resolve())
        if args.roads
        else None
    )

    for label, folder in (
        ("DEM", dem),
        ("MTK", mtk),
        ("Point-cloud", pointcloud),
    ):
        if not folder.is_dir():
            raise ValueError(f"{label} directory does not exist: {folder}")

    output.mkdir(parents=True, exist_ok=True)
    (output / "placements").mkdir(exist_ok=True)
    (output / "landcover").mkdir(exist_ok=True)

    tiles = read_terrain_index(dem)
    pointcloud_files = read_pointcloud_files(pointcloud)

    signature = run_signature(
        dem,
        mtk,
        pointcloud,
        buildings,
        roads,
    )
    manifest_path = output / MANIFEST_NAME
    resume = read_resume_state(
        manifest_path,
        signature,
        args.force_rebuild,
    )

    if resume:
        print(
            f"Resuming: {len(resume)} completed tile records found.",
            flush=True,
        )
    elif manifest_path.is_file() and not args.force_rebuild:
        print(
            "Existing manifest does not match the current generator/"
            "inputs/config; regenerating tiles.",
            flush=True,
        )

    records = {}
    pending_tiles = []
    total = 0

    # Preserve every valid reusable tile before writing any checkpoint.
    for tile in tiles:
        record = resume.get(tile["name"])

        if record and reusable_tile(output, tile, record):
            records[tile["name"]] = record
            total += record["placement_count"]
        else:
            pending_tiles.append(tile)

    def checkpoint():
        """Save all completed records in original terrain-manifest order."""
        ordered_records = [
            records[tile["name"]]
            for tile in tiles
            if tile["name"] in records
        ]
        write_manifest(
            manifest_path,
            signature,
            ordered_records,
            total,
        )

    def record_completed_tile(tile, count):
        """Record and checkpoint one successful tile in the parent process."""
        nonlocal total

        records[tile["name"]] = {
            "terrain_filename": tile["name"],
            "placement_filename": f"placements/{tile['name']}.vpl",
            "landcover_filename": f"landcover/{tile['name']}.json",
            "placement_count": count,
            "status": "complete",
        }
        total += count
        checkpoint()

        print(
            f"Completed {len(records)}/{len(tiles)} tiles: "
            f"{tile['name']} — {count} placements.",
            flush=True,
        )

    checkpoint()

    print(
        f"Terrain tiles: {len(tiles)}; "
        f"reusing: {len(records)}; "
        f"to generate: {len(pending_tiles)}; "
        f"point-cloud files: {len(pointcloud_files)}.",
        flush=True,
    )

    if not buildings:
        print(
            "Building exclusions disabled: --buildings was not supplied.",
            flush=True,
        )
    if not roads:
        print(
            "Road exclusions disabled: --roads was not supplied.",
            flush=True,
        )

    if not pending_tiles:
        print(
            f"Finished: {total} placements in {output}. All tiles reused.",
            flush=True,
        )
        return

    workers = min(args.workers, len(pending_tiles))
    print(f"Generating with {workers} worker(s).", flush=True)

    initializer_args = (
        str(mtk),
        buildings,
        roads,
        [str(path) for path in pointcloud_files],
        str(output),
    )

    if workers == 1:
        # Sequential mode retains the same worker initialization and logic.
        initialize_landcover_worker(*initializer_args)

        for tile in pending_tiles:
            try:
                count = generate_tile_in_worker(tile)
            except Exception:
                print(
                    f"\nFAILED tile {tile['name']}",
                    file=sys.stderr,
                    flush=True,
                )
                raise

            record_completed_tile(tile, count)

    else:
        failed_tiles = []
        context = multiprocessing.get_context("spawn")

        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=initialize_landcover_worker,
            initargs=initializer_args,
        ) as executor:
            futures = {
                executor.submit(generate_tile_in_worker, tile): tile
                for tile in pending_tiles
            }

            for future in as_completed(futures):
                tile = futures[future]

                try:
                    count = future.result()
                except Exception as error:
                    failed_tiles.append(tile["name"])
                    print(
                        f"\nFAILED tile {tile['name']}: {error}",
                        file=sys.stderr,
                        flush=True,
                    )
                    traceback.print_exception(
                        type(error),
                        error,
                        error.__traceback__,
                        file=sys.stderr,
                    )
                    # Collect remaining successes so they can be resumed.
                    continue

                record_completed_tile(tile, count)

        if failed_tiles:
            preview = ", ".join(failed_tiles[:10])
            if len(failed_tiles) > 10:
                preview += ", ..."

            raise RuntimeError(
                f"{len(failed_tiles)} tile(s) failed: {preview}. "
                f"{len(records)}/{len(tiles)} completed tiles were saved. "
                "Rerun the same command to retry unfinished tiles."
            )

    print(f"Finished: {total} placements in {output}", flush=True)


if __name__ == "__main__":
    # Required guard for safe Windows process spawning.
    multiprocessing.freeze_support()

    try:
        ogr.UseExceptions()
        main()
    except KeyboardInterrupt:
        print(
            "\nInterrupted. Checkpointed tiles can be reused on the next run.",
            file=sys.stderr,
        )
        sys.exit(130)
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
