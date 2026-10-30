#!/usr/bin/env python3
"""
Prepare LiDAR vegetation areas and tiled viewer placements in parallel.

Usage:
    python prepare_vegetation.py \
        --terrain espoo_terrain \
        --downloads espoo_pointcloud \
        --mtk MTK_2009_clipped \
        --assets assets/veg \
        -o vegetation_output --workers 2 --test-tiles 10

Arguments:
    --terrain      Folder containing tile_manifest.json and DEM files.
    --downloads    Folder containing download_manifest.csv, pointclouds/,
                   and _index/2008_latest.shp.
    --mtk          Folder containing clipped MTK GeoPackages.
    --assets       Folder containing vegetation.json and species assets.
    -o, --output   New or empty output folder.
    --workers      Parallel worker count; default 2.
    --test-tiles   First N terrain-manifest tiles; 0 means all.
    --index        Optional override for the coverage shapefile.

Outputs:
    vegetation_areas.gpkg
    vegetation_manifest.json
    vegetation/*.veg
    VEG_FORMAT.txt

Configuration is below. Point clouds, DEMs, and MTK layers must use
EPSG:3067. The coverage index is reprojected when needed.

Important:
    - Original DEM acquisition consistency is not verified.
    - Area partitions use nearest symbols, not equal-area partitioning.
    - Inspection areas are split at terrain tile boundaries.
    - LAS files are reread for intersecting processing tiles.
    - Output is not resumable; use a new folder after a failed run.
    - The existing C++ viewer needs a vegetation loader/rendering update.

Requires GDAL, numpy, scipy, rasterio, laspy+lazrs, shapely >= 2.
"""

import argparse
import csv
import hashlib
import json
import math
import multiprocessing
import struct
import sys
import tempfile

from collections import defaultdict
from concurrent.futures import (
    ProcessPoolExecutor,
    wait,
    FIRST_COMPLETED,
)
from pathlib import Path
from urllib.parse import unquote

import laspy
import numpy as np
import rasterio
import shapely

from osgeo import gdal, ogr, osr
from rasterio.features import rasterize, shapes
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely import (
    contains_xy,
    from_wkb,
    intersects_xy,
    make_valid,
    to_wkb,
    union_all,
)
from shapely.geometry import Point, box, shape
from shapely.strtree import STRtree


# ============================================================================
# EDITABLE CONFIGURATION
# ============================================================================

CONFIG = {
    "seed": 12345,

    # Internal raster resolution and neighboring processing context.
    "resolution": 2.0,
    "processing_buffer": 20.0,

    # Accept all ordinary vegetation classes, including your class-3 data.
    "vegetation_classes": [3, 4, 5],

    # A vegetation point needs another observation within this XY radius.
    "isolation_radius": 5.0,

    # Heights above the ground DEM.
    "minimum_height": 0.5,
    "tree_height_threshold": 3.0,
    "maximum_height": 45.0,
    "height_percentile": 85.0,

    # Coverage cleanup and approximate boundary smoothing.
    "closing_iterations": 1,
    "minimum_area_m2": 12.0,
    "polygon_smoothing_metres": 1.0,
    "polygon_simplification_metres": 0.5,

    # Only local DEM gaps within this distance are interpolated.
    "ground_fill_distance_metres": 150.0,

    # Placement exclusion radii.
    "building_buffer_metres": 3.0,
    "road_default_radius_metres": 3.0,

    # Edit with actual tieviiva LUOKKA codes, for example:
    # 12345: 5.0
    "road_radius_by_luokka": {},

    "classification_max_distance": 500.0,
    "viewer_cell_size": 25.0,

    # Jittered placement lattice. This is approximate spacing, not an
    # inventory-derived tree count or exact Poisson-distance sampling.
    "tree_spacing_metres": 6.0,
    "shrub_spacing_metres": 2.5,
    "placement_probability": 0.85,
    "height_variation": [0.85, 1.10],
    "width_variation": [0.85, 1.15],

    # True means detected vegetation outside parks/cemeteries is allowed.
    # False restricts vegetation to the configured allow polygons.
    "allow_detected_vegetation_elsewhere": True,

    "allow_layers": [
        ("mtk-muut_2009_clipped.gpkg", "puisto"),
        ("mtk-muut_2009_clipped.gpkg", "hautausmaa"),
    ],

    "exclude_layers": [
        (
            "mtk-rakennus_2009_clipped.gpkg",
            "rakennus",
            "buildings",
        ),
        ("mtk-vakavesi_2009_clipped.gpkg", "meri", "water"),
        ("mtk-vakavesi_2009_clipped.gpkg", "jarvi", "water"),
    ],

    "road_file": "mtk-tie_2009_clipped.gpkg",
    "road_layer": "tieviiva",

    "classification_file": "mtk-muut_2009_clipped.gpkg",
    "classification_layer": "metsamaankasvillisuus",
    "classification_field": "LUOKKA",

    "class_generators": {
        32710: "conifer_mix",
        32713: "deciduous",
        32714: "deciduous_32714",
    },

    # Separate 32714 configuration allows custom behavior later.
    "generators": {
        "conifer_mix": {
            "pine": 0.48,
            "spruce": 0.42,
            "birch": 0.08,
            "aspen": 0.02,
        },
        "deciduous": {
            "birch": 0.65,
            "aspen": 0.20,
            "alder": 0.15,
        },
        "deciduous_32714": {
            "birch": 0.65,
            "aspen": 0.20,
            "alder": 0.15,
        },
        "unknown": {
            "birch": 0.40,
            "pine": 0.25,
            "spruce": 0.25,
            "alder": 0.10,
        },
        "shrubs": {
            "willow_shrub": 0.65,
            "leafy_shrub": 0.35,
        },
    },
}

EPSG = 3067

# Each spawned process owns its own worker state and GIS handles.
WORKER = {}


# ============================================================================
# CRS AND VECTOR HELPERS
# ============================================================================

def crs_3067():
    """Create EPSG:3067 with conventional easting/northing axis order."""
    crs = osr.SpatialReference()
    crs.ImportFromEPSG(EPSG)
    crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return crs


def normalize_crs(crs):
    """Clone a spatial reference and normalize its axis order."""
    result = crs.Clone()
    result.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return result


def validate_crs(crs, description):
    """Reject absent or non-3067 CRS metadata for MTK layers."""
    if crs is None:
        raise ValueError(f"No CRS: {description}")
    if not normalize_crs(crs).IsSame(crs_3067()):
        raise ValueError(f"Expected EPSG:3067: {description}")


def polygon_parts(geometry):
    """Yield polygon components from multipart or collection geometries."""
    if geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
    elif hasattr(geometry, "geoms"):
        for child in geometry.geoms:
            yield from polygon_parts(child)


class VectorLayer:
    """An independently opened OGR layer for worker-local spatial queries."""

    def __init__(self, filename, layer_name):
        self.dataset = ogr.Open(str(filename), 0)
        if self.dataset is None:
            raise ValueError(f"Cannot open {filename}")

        self.layer = self.dataset.GetLayerByName(layer_name)
        if self.layer is None:
            raise ValueError(
                f"Layer {layer_name!r} missing in {filename}"
            )

        validate_crs(
            self.layer.GetSpatialRef(),
            f"{filename}:{layer_name}",
        )

    def query(self, bounds, field=None):
        """Yield repaired geometries and an optional attribute."""
        if field:
            if self.layer.GetLayerDefn().GetFieldIndex(field) < 0:
                raise ValueError(
                    f"Field {field!r} missing in {self.layer.GetName()}"
                )

        self.layer.SetSpatialFilterRect(*bounds)
        self.layer.ResetReading()

        for feature in self.layer:
            geometry = feature.GetGeometryRef()
            if geometry is None:
                continue

            result = make_valid(
                from_wkb(bytes(geometry.ExportToWkb()))
            )
            if result.is_empty:
                continue

            value = feature.GetField(field) if field else None
            yield result, value


def open_mtk(folder):
    """Open exclusions, optional allow polygons, roads, and class symbols."""
    exclusions = [
        (VectorLayer(folder / filename, layer), treatment)
        for filename, layer, treatment in CONFIG["exclude_layers"]
    ]

    # Allow layers are unnecessary when unrestricted detected vegetation
    # is enabled, so do not require those layers in that case.
    allowed = []
    if not CONFIG["allow_detected_vegetation_elsewhere"]:
        allowed = [
            VectorLayer(folder / filename, layer)
            for filename, layer in CONFIG["allow_layers"]
        ]

    roads = VectorLayer(
        folder / CONFIG["road_file"],
        CONFIG["road_layer"],
    )
    classes = VectorLayer(
        folder / CONFIG["classification_file"],
        CONFIG["classification_layer"],
    )

    field = CONFIG["classification_field"]
    if classes.layer.GetLayerDefn().GetFieldIndex(field) < 0:
        raise ValueError(
            f"Classification field {field!r} is missing."
        )

    return exclusions, allowed, roads, classes


def tile_exclusions(bounds, mtk):
    """Combine buffered buildings/roads, water, and optional allowed areas."""
    exclusions, allowed, roads, _ = mtk
    pieces = []

    radii = [
        CONFIG["building_buffer_metres"],
        CONFIG["road_default_radius_metres"],
        *CONFIG["road_radius_by_luokka"].values(),
    ]
    extra = max(radii)
    query_bounds = box(*bounds).buffer(extra).bounds

    for layer, treatment in exclusions:
        for geometry, _ in layer.query(query_bounds):
            if treatment == "buildings":
                geometry = geometry.buffer(
                    CONFIG["building_buffer_metres"]
                )
            pieces.append(geometry)

    for geometry, value in roads.query(query_bounds, "LUOKKA"):
        try:
            code = int(value)
        except (TypeError, ValueError):
            code = -1

        radius = CONFIG["road_radius_by_luokka"].get(
            code, CONFIG["road_default_radius_metres"]
        )
        pieces.append(geometry.buffer(radius))

    exclusion = (
        union_all(pieces)
        if pieces else shapely.GeometryCollection()
    )

    if not CONFIG["allow_detected_vegetation_elsewhere"]:
        allowed_shapes = [
            geometry
            for layer in allowed
            for geometry, _ in layer.query(bounds)
        ]
        allow = (
            union_all(allowed_shapes)
            if allowed_shapes else shapely.GeometryCollection()
        )
        exclusion = union_all([
            exclusion,
            box(*bounds).difference(allow),
        ])

    return make_valid(exclusion)


# ============================================================================
# ACQUISITION FOOTPRINTS AND OLDEST-DATA OWNERSHIP
# ============================================================================

def archive_relative(value):
    """Normalize an archive path to YEAR/subdirectories/filename."""
    text = unquote(str(value).strip()).replace("\\", "/")
    text = text.lstrip("/")
    prefix = "mml/laserkeilaus/2008_latest/"

    if text.startswith(prefix):
        text = text[len(prefix):]

    return text


def load_oldest_sources(downloads, index_path):
    """
    Locate downloads and construct oldest-year spatial ownership.

    Index polygons are reprojected to EPSG:3067 if necessary.
    Same-year files can overlap. Later years contribute only outside
    earlier-year downloaded coverage.
    """
    manifest_path = downloads / "download_manifest.csv"
    if not manifest_path.is_file():
        raise ValueError(
            f"Download manifest does not exist: {manifest_path}"
        )

    with manifest_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError("The download manifest is empty.")

    requested = {}

    for row in rows:
        relative = archive_relative(row.get("path") or "")

        if not relative:
            url = row.get("url", "")
            marker = "/2008_latest/"
            if marker not in url:
                raise ValueError(
                    f"Cannot parse manifest URL: {url}"
                )
            relative = archive_relative(url.split(marker, 1)[1])

        filename = Path(relative).name
        local = downloads / "pointclouds" / filename

        if not local.is_file():
            recorded = row.get("local_path", "")
            if recorded:
                local = Path(recorded)

        if not local.is_file():
            raise ValueError(
                f"Missing downloaded point cloud: {filename}"
            )

        requested[relative] = {
            "file": local,
            "year": int(row["year"]),
            "footprints": [],
        }

    dataset = ogr.Open(str(index_path), 0)
    if dataset is None:
        raise ValueError(
            f"Cannot open coverage index: {index_path}"
        )

    layer = dataset.GetLayer(0)
    if layer is None:
        raise ValueError("Coverage index has no readable layer.")

    index_crs = layer.GetSpatialRef()
    if index_crs is None:
        raise ValueError(
            f"Coverage index has no CRS: {index_path}\n"
            "Keep its matching .prj file beside the shapefile."
        )

    index_crs = normalize_crs(index_crs)
    destination_crs = crs_3067()
    transformation = None

    if not index_crs.IsSame(destination_crs):
        print(
            f"Worker {multiprocessing.current_process().name}: "
            "reprojecting index polygons to EPSG:3067.",
            flush=True,
        )
        transformation = osr.CoordinateTransformation(
            index_crs, destination_crs
        )

    definition = layer.GetLayerDefn()
    fields = {
        definition.GetFieldDefn(i).GetName().lower():
        definition.GetFieldDefn(i).GetName()
        for i in range(definition.GetFieldCount())
    }
    if "path" not in fields:
        raise ValueError("Coverage index has no path attribute.")

    layer.ResetReading()
    for feature in layer:
        value = feature.GetField(fields["path"])
        if value is None:
            continue

        relative = archive_relative(value)
        if relative not in requested:
            continue

        geometry = feature.GetGeometryRef()
        if geometry is None or geometry.IsEmpty():
            continue

        geometry = geometry.Clone()
        if transformation is not None:
            if geometry.Transform(transformation) != 0:
                raise RuntimeError(
                    f"Could not transform index footprint: {relative}"
                )

        footprint = make_valid(
            from_wkb(bytes(geometry.ExportToWkb()))
        )
        if not footprint.is_empty:
            requested[relative]["footprints"].append(footprint)

    layer = None
    dataset = None

    by_year = defaultdict(list)
    for relative, entry in requested.items():
        if not entry["footprints"]:
            raise ValueError(
                f"No index footprint found for {relative}"
            )
        entry["footprint"] = make_valid(
            union_all(entry.pop("footprints"))
        )
        by_year[entry["year"]].append(entry)

    older_coverage = shapely.GeometryCollection()
    sources = []

    for year in sorted(by_year):
        entries = by_year[year]

        for entry in entries:
            owner = make_valid(
                entry["footprint"].difference(older_coverage)
            )
            if not owner.is_empty:
                entry["owner"] = owner
                sources.append(entry)

        # Extend coverage only after processing all files of this year.
        older_coverage = make_valid(union_all([
            older_coverage,
            *[entry["footprint"] for entry in entries],
        ]))

    if not sources:
        raise ValueError("No usable oldest-acquisition coverage.")

    source_tree = STRtree([
        entry["owner"] for entry in sources
    ])

    print(
        f"Worker {multiprocessing.current_process().name}: "
        f"prepared ownership for {len(sources)} source files.",
        flush=True,
    )

    return sources, source_tree


# ============================================================================
# GROUND DEM AND LIDAR PROCESSING
# ============================================================================

def build_ground_vrt(terrain, manifest, directory):
    """Build a read-only VRT referencing original DEM tiles."""
    files = [
        str(terrain / tile["dem"])
        for tile in manifest["tiles"]
        if tile.get("dem")
    ]
    if not files:
        raise ValueError("Terrain manifest has no DEM files.")

    expected = rasterio.crs.CRS.from_epsg(EPSG)
    for filename in files:
        with rasterio.open(filename) as dataset:
            if dataset.crs != expected:
                raise ValueError(
                    f"DEM must use EPSG:3067: {filename}"
                )

    path = directory / "ground.vrt"
    result = gdal.BuildVRT(
        str(path),
        files,
        options=gdal.BuildVRTOptions(VRTNodata=-9999),
    )
    if result is None:
        raise RuntimeError("Could not build ground DEM VRT.")

    result = None
    return path


def ground_grid(vrt, bounds, resolution, directory):
    """Resample ground to a tile grid and interpolate small local gaps."""
    path = directory / "tile_ground.tif"

    result = gdal.Warp(
        str(path),
        str(vrt),
        format="GTiff",
        outputBounds=bounds,
        xRes=resolution,
        yRes=resolution,
        dstSRS=f"EPSG:{EPSG}",
        resampleAlg="bilinear",
        dstNodata=-9999,
    )
    if result is None:
        raise RuntimeError("Ground resampling failed.")

    band = result.GetRasterBand(1)
    status = gdal.FillNodata(
        targetBand=band,
        maskBand=band.GetMaskBand(),
        maxSearchDist=(
            CONFIG["ground_fill_distance_metres"] / resolution
        ),
        smoothingIterations=0,
    )
    if status != 0:
        raise RuntimeError("Local ground interpolation failed.")

    band = None
    result.FlushCache()
    result = None

    with rasterio.open(path) as dataset:
        data = (
            dataset.read(1, masked=True)
            .astype(float)
            .filled(np.nan)
        )
        transform = dataset.transform

    return data, transform


def read_vegetation(bounds, sources, source_tree):
    """Read oldest-owned vegetation points and reject isolated observations."""
    query = box(*bounds)
    arrays = []

    for source_index in source_tree.query(
        query, predicate="intersects"
    ):
        entry = sources[int(source_index)]
        owner = entry["owner"].intersection(query)
        if owner.is_empty:
            continue

        with laspy.open(entry["file"]) as reader:
            source_crs = reader.header.parse_crs()
            if source_crs is not None:
                if rasterio.crs.CRS.from_user_input(
                    source_crs
                ) != rasterio.crs.CRS.from_epsg(EPSG):
                    raise ValueError(
                        f"LAS CRS mismatch: {entry['file']}"
                    )

            for points in reader.chunk_iterator(1_000_000):
                keep = (
                    np.isin(
                        np.asarray(points.classification),
                        CONFIG["vegetation_classes"],
                    )
                    & ~np.asarray(points.withheld, dtype=bool)
                )
                if not np.any(keep):
                    continue

                x = np.asarray(points.x)[keep]
                y = np.asarray(points.y)[keep]
                z = np.asarray(points.z)[keep]

                inside = (
                    np.isfinite(x)
                    & np.isfinite(y)
                    & np.isfinite(z)
                    & (x >= bounds[0])
                    & (x < bounds[2])
                    & (y >= bounds[1])
                    & (y < bounds[3])
                )
                x, y, z = x[inside], y[inside], z[inside]
                if not len(x):
                    continue

                selected = intersects_xy(owner, x, y)
                if np.any(selected):
                    arrays.append(np.column_stack((
                        x[selected], y[selected], z[selected]
                    )))

    if not arrays:
        return np.empty((0, 3))

    # Exact duplicate points should not support themselves.
    points = np.unique(np.concatenate(arrays), axis=0)
    if len(points) < 2:
        return np.empty((0, 3))

    distances, _ = cKDTree(points[:, :2]).query(
        points[:, :2],
        k=2,
        distance_upper_bound=CONFIG["isolation_radius"],
    )

    return points[np.isfinite(distances[:, 1])]


def vegetation_grid(points, ground, transform, exclusion):
    """Calculate representative heights and classify shrub/tree coverage."""
    height, width = ground.shape
    representative = np.full(ground.shape, np.nan)

    if len(points):
        columns = np.floor(
            (points[:, 0] - transform.c) / transform.a
        ).astype(int)
        rows = np.floor(
            (points[:, 1] - transform.f) / transform.e
        ).astype(int)

        inside = (
            (columns >= 0) & (columns < width)
            & (rows >= 0) & (rows < height)
        )
        points = points[inside]
        columns, rows = columns[inside], rows[inside]

        above_ground = points[:, 2] - ground[rows, columns]
        valid = (
            np.isfinite(above_ground)
            & (above_ground >= CONFIG["minimum_height"])
            & (above_ground <= CONFIG["maximum_height"])
        )

        ids = rows[valid] * width + columns[valid]
        values = above_ground[valid]

        order = np.argsort(ids)
        ids, values = ids[order], values[order]

        if len(ids):
            starts = np.r_[0, np.flatnonzero(np.diff(ids)) + 1]
            ends = np.r_[starts[1:], len(ids)]

            for start, end in zip(starts, ends):
                row, column = divmod(int(ids[start]), width)
                representative[row, column] = np.percentile(
                    values[start:end],
                    CONFIG["height_percentile"],
                )

    occupied = np.isfinite(representative)
    if CONFIG["closing_iterations"]:
        occupied = ndimage.binary_closing(
            occupied,
            iterations=CONFIG["closing_iterations"],
        )

    measured = np.isfinite(representative)
    if np.any(measured):
        _, nearest = ndimage.distance_transform_edt(
            ~measured, return_indices=True
        )
        representative[occupied] = representative[
            nearest[0][occupied], nearest[1][occupied]
        ]

    blocked = np.zeros(ground.shape, dtype=bool)
    if not exclusion.is_empty:
        blocked = rasterize(
            [(exclusion, 1)],
            out_shape=ground.shape,
            transform=transform,
            all_touched=True,
            dtype="uint8",
        ).astype(bool)

    occupied &= ~blocked & np.isfinite(ground)

    kinds = np.zeros(ground.shape, dtype=np.uint8)
    kinds[occupied] = 1
    kinds[
        occupied
        & (
            representative >= CONFIG["tree_height_threshold"]
        )
    ] = 2

    minimum_cells = math.ceil(
        CONFIG["minimum_area_m2"] / transform.a**2
    )

    for kind in (1, 2):
        labels, _ = ndimage.label(kinds == kind)
        sizes = np.bincount(labels.ravel())
        keep = sizes >= minimum_cells
        keep[0] = False
        kinds[(kinds == kind) & ~keep[labels]] = 0

    return representative, kinds


# ============================================================================
# MTK CLASSIFICATION AND INSPECTION POLYGONS
# ============================================================================

def load_class_points(layer):
    """Read MTK vegetation symbols and build a nearest-neighbor index."""
    coordinates, codes = [], []

    layer.layer.SetSpatialFilter(None)
    layer.layer.ResetReading()

    for feature in layer.layer:
        geometry = feature.GetGeometryRef()
        if geometry is None:
            continue
        if ogr.GT_Flatten(
            geometry.GetGeometryType()
        ) != ogr.wkbPoint:
            continue

        try:
            code = int(feature.GetField(
                CONFIG["classification_field"]
            ))
        except (TypeError, ValueError):
            continue

        coordinates.append((geometry.GetX(), geometry.GetY()))
        codes.append(code)

    coordinates = np.asarray(
        coordinates, dtype=float
    ).reshape(-1, 2)
    codes = np.asarray(codes, dtype=np.int32)
    tree = cKDTree(coordinates) if len(coordinates) else None

    return coordinates, codes, tree


def classify_grid(kinds, transform, class_points):
    """
    Assign LUOKKA to connected coverage regions.

    Different contained classes use nearest-contained-symbol partitioning.
    With no contained symbols, use the nearest symbol to the region within
    the configured maximum distance.
    """
    coordinates, codes, tree = class_points
    assigned = np.zeros(kinds.shape, dtype=np.int32)

    if tree is None:
        return assigned

    for kind in (1, 2):
        labels, number = ndimage.label(kinds == kind)

        for label in range(1, number + 1):
            mask = labels == label
            rows, columns = np.nonzero(mask)

            polygons = [
                shape(geometry)
                for geometry, value in shapes(
                    mask.astype(np.uint8),
                    mask=mask,
                    transform=transform,
                )
                if value == 1
            ]
            area = union_all(polygons)
            xmin, ymin, xmax, ymax = area.bounds

            center = ((xmin + xmax) / 2, (ymin + ymax) / 2)
            radius = math.hypot(xmax - xmin, ymax - ymin) / 2

            candidates = tree.query_ball_point(
                center,
                radius + CONFIG["classification_max_distance"],
            )
            if not candidates:
                continue

            indexes = np.asarray(candidates, dtype=int)
            contained = indexes[contains_xy(
                area,
                coordinates[indexes, 0],
                coordinates[indexes, 1],
            )]

            if len(contained):
                distinct = np.unique(codes[contained])

                if len(distinct) == 1:
                    assigned[rows, columns] = distinct[0]
                else:
                    positions = np.column_stack((
                        transform.c
                        + (columns + 0.5) * transform.a,
                        transform.f
                        + (rows + 0.5) * transform.e,
                    ))
                    _, nearest = cKDTree(
                        coordinates[contained]
                    ).query(positions)
                    assigned[rows, columns] = codes[
                        contained[nearest]
                    ]
            else:
                nearest_index = min(
                    candidates,
                    key=lambda i: area.distance(
                        Point(coordinates[i])
                    ),
                )
                distance = area.distance(
                    Point(coordinates[nearest_index])
                )
                if distance <= CONFIG["classification_max_distance"]:
                    assigned[rows, columns] = codes[nearest_index]

    return assigned


def create_inspection(path):
    """Create the shared GeoPackage; only the main process writes it."""
    dataset = ogr.GetDriverByName("GPKG").CreateDataSource(
        str(path)
    )
    layer = dataset.CreateLayer(
        "vegetation_areas",
        srs=crs_3067(),
        geom_type=ogr.wkbMultiPolygon,
    )

    for name, field_type in (
        ("tile", ogr.OFTString),
        ("LUOKKA", ogr.OFTInteger),
        ("plant_type", ogr.OFTString),
        ("generator", ogr.OFTString),
        ("height_m", ogr.OFTReal),
        ("area_m2", ogr.OFTReal),
    ):
        layer.CreateField(ogr.FieldDefn(name, field_type))

    return dataset, layer


def make_areas(kinds, classes, heights, transform, core, exclusion):
    """Polygonize coverage, smooth boundaries, and reapply exclusions."""
    areas = []
    core_polygon = box(*core)

    # Prevent smoothing from creating overlaps between semantic classes.
    claimed = shapely.GeometryCollection()

    for kind in (1, 2):
        for code in np.unique(classes[kinds == kind]):
            mask = (kinds == kind) & (classes == code)
            if not np.any(mask):
                continue

            geometries = [
                shape(geometry)
                for geometry, _ in shapes(
                    mask.astype(np.uint8),
                    mask=mask,
                    transform=transform,
                )
            ]

            geometry = union_all(geometries)
            smoothing = CONFIG["polygon_smoothing_metres"]

            if smoothing:
                geometry = geometry.buffer(
                    smoothing
                ).buffer(-smoothing)

            geometry = geometry.simplify(
                CONFIG["polygon_simplification_metres"],
                preserve_topology=True,
            )
            geometry = make_valid(
                geometry.intersection(core_polygon)
                .difference(exclusion)
                .difference(claimed)
            )

            generator = (
                "shrubs" if kind == 1
                else CONFIG["class_generators"].get(
                    int(code), "unknown"
                )
            )

            accepted_parts = []

            for polygon in polygon_parts(geometry):
                if polygon.area < CONFIG["minimum_area_m2"]:
                    continue

                part_mask = rasterize(
                    [(polygon, 1)],
                    out_shape=kinds.shape,
                    transform=transform,
                    dtype="uint8",
                ).astype(bool)

                selected = heights[part_mask & mask]
                selected = selected[np.isfinite(selected)]
                representative_height = (
                    float(np.median(selected))
                    if len(selected)
                    else (1.5 if kind == 1 else 12.0)
                )

                areas.append({
                    "geometry": polygon,
                    "kind": kind,
                    "code": int(code),
                    "generator": generator,
                    "height": representative_height,
                })
                accepted_parts.append(polygon)

            if accepted_parts:
                claimed = union_all([claimed, *accepted_parts])

    return areas


def write_area(layer, tile_name, area):
    """Write one polygon and its classification/height attributes."""
    feature = ogr.Feature(layer.GetLayerDefn())
    geometry = ogr.CreateGeometryFromWkb(
        to_wkb(area["geometry"])
    )
    feature.SetGeometry(ogr.ForceToMultiPolygon(geometry))

    feature.SetField("tile", tile_name)
    feature.SetField("LUOKKA", area["code"])
    feature.SetField(
        "plant_type", "shrub" if area["kind"] == 1 else "tree"
    )
    feature.SetField("generator", area["generator"])
    feature.SetField("height_m", area["height"])
    feature.SetField("area_m2", area["geometry"].area)

    layer.CreateFeature(feature)


# ============================================================================
# ASSET LIBRARY AND PLANT GENERATION
# ============================================================================

def load_assets(folder):
    """Assign stable numeric IDs to the generated vegetation asset library."""
    library = json.loads(
        (folder / "vegetation.json").read_text(encoding="utf-8")
    )
    assets = []
    by_species = defaultdict(list)

    for item in library["assets"]:
        definition = folder / item["definition"]
        asset = json.loads(
            definition.read_text(encoding="utf-8")
        )
        asset["numeric_id"] = len(assets)
        asset["definition_path"] = str(definition.resolve())

        if asset["plant_height_metres"] <= 0:
            raise ValueError(
                f"Invalid plant height in {definition}"
            )

        assets.append(asset)
        by_species[asset["species"]].append(
            asset["numeric_id"]
        )

    for mixture in CONFIG["generators"].values():
        if sum(mixture.values()) <= 0:
            raise ValueError("A species mixture has no positive weights.")
        for species, weight in mixture.items():
            if weight < 0:
                raise ValueError("Species weights cannot be negative.")
            if species not in by_species:
                raise ValueError(
                    f"Missing asset species: {species}"
                )

    return assets, by_species


def stable_rng(kind, ix, iy):
    """Anchor random placement to a stable global lattice."""
    text = (
        f"{CONFIG['seed']}:{kind}:{ix}:{iy}"
    ).encode("ascii")

    seed = int.from_bytes(
        hashlib.blake2b(text, digest_size=8).digest(),
        "little",
    )
    return np.random.default_rng(seed)


def sample_cell(array, transform, x, y):
    """Read the nearest working-grid cell at a map coordinate."""
    column = int(math.floor(
        (x - transform.c) / transform.a
    ))
    row = int(math.floor(
        (y - transform.f) / transform.e
    ))

    if not (
        0 <= row < array.shape[0]
        and 0 <= column < array.shape[1]
    ):
        return np.nan

    return array[row, column]


def generate_plants(
    core, areas, ground, heights, transform, assets, by_species
):
    """Generate repeatable plants with local heights and species variation."""
    plants = []
    xmin, ymin, xmax, ymax = core

    for kind in (1, 2):
        eligible = [
            area for area in areas if area["kind"] == kind
        ]
        if not eligible:
            continue

        spatial = STRtree([
            area["geometry"] for area in eligible
        ])
        spacing = CONFIG[
            "shrub_spacing_metres"
            if kind == 1 else "tree_spacing_metres"
        ]

        for ix in range(
            math.floor(xmin / spacing),
            math.ceil(xmax / spacing),
        ):
            for iy in range(
                math.floor(ymin / spacing),
                math.ceil(ymax / spacing),
            ):
                rng = stable_rng(kind, ix, iy)
                x = (
                    ix + 0.5 + rng.uniform(-0.20, 0.20)
                ) * spacing
                y = (
                    iy + 0.5 + rng.uniform(-0.20, 0.20)
                ) * spacing

                if not (
                    xmin <= x < xmax and ymin <= y < ymax
                ):
                    continue
                if rng.random() > CONFIG["placement_probability"]:
                    continue

                matches = spatial.query(
                    Point(x, y), predicate="intersects"
                )
                if not len(matches):
                    continue

                area = eligible[int(matches[0])]
                elevation = sample_cell(
                    ground, transform, x, y
                )
                if not np.isfinite(elevation):
                    continue

                measured = sample_cell(
                    heights, transform, x, y
                )
                plant_height = (
                    measured
                    if np.isfinite(measured)
                    else area["height"]
                )
                plant_height *= rng.uniform(
                    *CONFIG["height_variation"]
                )

                if kind == 1:
                    plant_height = np.clip(
                        plant_height, 0.5, 2.95
                    )
                else:
                    plant_height = np.clip(
                        plant_height,
                        3.05,
                        CONFIG["maximum_height"],
                    )

                mixture = CONFIG["generators"][
                    area["generator"]
                ]
                species = list(mixture)
                weights = np.asarray(
                    list(mixture.values()), dtype=float
                )
                weights /= weights.sum()

                selected_species = rng.choice(
                    species, p=weights
                )
                asset_id = int(rng.choice(
                    by_species[selected_species]
                ))
                asset = assets[asset_id]

                height_scale = (
                    plant_height
                    / asset["plant_height_metres"]
                )

                plants.append((
                    asset_id,
                    float(x - xmin),
                    float(ymax - y),
                    float(elevation),
                    float(height_scale),
                    float(
                        height_scale
                        * rng.uniform(*CONFIG["width_variation"])
                    ),
                    float(rng.uniform(0, math.tau)),
                    float(rng.uniform(0.90, 1.08)),
                    float(rng.random()),
                ))

    return plants


def write_veg(path, core, plants):
    """Write little-endian VEG1 data grouped into occupied viewer cells."""
    cell_size = CONFIG["viewer_cell_size"]
    cells = defaultdict(list)

    for plant in plants:
        cell = (
            int(plant[1] // cell_size),
            int(plant[2] // cell_size),
        )
        cells[cell].append(plant)

    with path.open("wb") as output:
        output.write(b"VEG1")
        output.write(struct.pack(
            "<2df2I",
            core[0],
            core[3],
            cell_size,
            len(cells),
            len(plants),
        ))

        for (column, row), members in sorted(cells.items()):
            output.write(struct.pack(
                "<3I", column, row, len(members)
            ))
            for plant in members:
                output.write(struct.pack("<I8f", *plant))


# ============================================================================
# PARALLEL WORKERS
# ============================================================================

def initialize_worker(
    downloads_path,
    index_path,
    mtk_path,
    assets_path,
    output_path,
    ground_vrt_path,
    config,
):
    """Initialize worker-local GIS sources and spatial indexes once."""
    global CONFIG, WORKER
    CONFIG = config

    gdal.UseExceptions()
    ogr.UseExceptions()
    osr.UseExceptions()

    # Per-worker GDAL cache; point arrays and geometries use extra RAM.
    gdal.SetCacheMax(64 * 1024 * 1024)

    sources, source_tree = load_oldest_sources(
        Path(downloads_path), Path(index_path)
    )
    mtk = open_mtk(Path(mtk_path))
    class_points = load_class_points(mtk[3])
    assets, by_species = load_assets(Path(assets_path))

    WORKER = {
        "sources": sources,
        "source_tree": source_tree,
        "mtk": mtk,
        "class_points": class_points,
        "assets": assets,
        "by_species": by_species,
        "output": Path(output_path),
        "ground_vrt": Path(ground_vrt_path),
    }


def process_tile_task(task):
    """
    Process one tile independently.

    Writes only its own VEG file. Returns polygon WKB for the main process
    to append to the shared inspection GeoPackage.
    """
    number, tile = task

    if not tile.get("core_bounds"):
        raise ValueError(
            f"Tile {tile['tile']} has no core_bounds."
        )

    core = tuple(tile["core_bounds"])
    buffer_metres = CONFIG["processing_buffer"]
    resolution = CONFIG["resolution"]

    bounds = (
        math.floor(
            (core[0] - buffer_metres) / resolution
        ) * resolution,
        math.floor(
            (core[1] - buffer_metres) / resolution
        ) * resolution,
        math.ceil(
            (core[2] + buffer_metres) / resolution
        ) * resolution,
        math.ceil(
            (core[3] + buffer_metres) / resolution
        ) * resolution,
    )

    # Each task has unique working files, preventing parallel collisions.
    with tempfile.TemporaryDirectory(
        prefix=f"_veg_tile_{number:06d}_",
        dir=str(WORKER["output"]),
    ) as temporary:
        directory = Path(temporary)

        ground, transform = ground_grid(
            WORKER["ground_vrt"],
            bounds,
            resolution,
            directory,
        )
        exclusion = tile_exclusions(
            bounds, WORKER["mtk"]
        )
        points = read_vegetation(
            bounds,
            WORKER["sources"],
            WORKER["source_tree"],
        )
        heights, kinds = vegetation_grid(
            points, ground, transform, exclusion
        )
        classes = classify_grid(
            kinds, transform, WORKER["class_points"]
        )
        areas = make_areas(
            kinds, classes, heights, transform,
            core, exclusion,
        )
        plants = generate_plants(
            core, areas, ground, heights, transform,
            WORKER["assets"], WORKER["by_species"],
        )

        filename = f"tile_{number:06d}.veg"
        destination = (
            WORKER["output"] / "vegetation" / filename
        )
        partial = destination.with_suffix(".veg.part")

        write_veg(partial, core, plants)
        partial.replace(destination)

        serialized_areas = []
        for area in areas:
            serialized = {
                key: value
                for key, value in area.items()
                if key != "geometry"
            }
            serialized["geometry_wkb"] = to_wkb(
                area["geometry"]
            )
            serialized_areas.append(serialized)

        return {
            "number": number,
            "tile_name": tile["tile"],
            "areas": serialized_areas,
            "manifest_entry": {
                "tile": tile["tile"],
                "terrain_filename": (
                    f"tiles/tile_{number:06d}.trn"
                ),
                "vegetation_filename": (
                    f"vegetation/{filename}"
                ),
                "core_bounds": core,
                "area_count": len(areas),
                "plant_count": len(plants),
                "retained_vegetation_points": len(points),
            },
        }


def parallel_tile_results(tasks, workers, initializer_args):
    """Yield completed tasks using a bounded Windows-safe process pool."""
    task_iterator = iter(tasks)
    context = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=initialize_worker,
        initargs=initializer_args,
    ) as executor:
        pending = {}

        def submit_next():
            try:
                task = next(task_iterator)
            except StopIteration:
                return False

            future = executor.submit(
                process_tile_task, task
            )
            pending[future] = task
            return True

        # Limit queued work and completed-result memory.
        for _ in range(workers * 2):
            if not submit_next():
                break

        try:
            while pending:
                finished, _ = wait(
                    pending,
                    return_when=FIRST_COMPLETED,
                )

                for future in finished:
                    task = pending.pop(future)
                    try:
                        result = future.result()
                    except Exception as error:
                        raise RuntimeError(
                            f"Tile {task[1]['tile']} failed: {error}"
                        ) from error

                    yield result
                    submit_next()

        finally:
            # Running tasks finish safely; queued tasks can be cancelled.
            for future in pending:
                future.cancel()


# ============================================================================
# VALIDATION AND MAIN PROCESS
# ============================================================================

def validate_config():
    """Catch invalid editable settings before launching worker processes."""
    for name in (
        "resolution",
        "isolation_radius",
        "tree_spacing_metres",
        "shrub_spacing_metres",
        "viewer_cell_size",
    ):
        if CONFIG[name] <= 0:
            raise ValueError(f"CONFIG[{name!r}] must be positive.")

    if CONFIG["processing_buffer"] < CONFIG["isolation_radius"]:
        raise ValueError(
            "processing_buffer must be at least isolation_radius."
        )

    if not 0 <= CONFIG["placement_probability"] <= 1:
        raise ValueError("placement_probability must be between 0 and 1.")

    if CONFIG["building_buffer_metres"] < 0:
        raise ValueError("Building buffer cannot be negative.")

    if CONFIG["road_default_radius_metres"] <= 0:
        raise ValueError("Road buffer radius must be positive.")

    if any(
        value <= 0
        for value in CONFIG["road_radius_by_luokka"].values()
    ):
        raise ValueError("Road-class buffer radii must be positive.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare oldest-LiDAR vegetation areas "
            "and viewer files in parallel."
        )
    )
    parser.add_argument("--terrain", required=True)
    parser.add_argument("--downloads", required=True)
    parser.add_argument("--mtk", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--test-tiles", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--index")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1.")
    if args.test_tiles < 0:
        parser.error("--test-tiles cannot be negative.")

    validate_config()
    gdal.UseExceptions()
    ogr.UseExceptions()
    osr.UseExceptions()
    gdal.SetCacheMax(128 * 1024 * 1024)

    terrain = Path(args.terrain).resolve()
    downloads = Path(args.downloads).resolve()
    mtk_folder = Path(args.mtk).resolve()
    asset_folder = Path(args.assets).resolve()
    output = Path(args.output).resolve()
    index_path = (
        Path(args.index).resolve()
        if args.index
        else downloads / "_index" / "2008_latest.shp"
    )

    if output.exists() and (
        not output.is_dir() or any(output.iterdir())
    ):
        parser.error(
            "Output folder must be empty or not yet exist."
        )

    if not index_path.is_file():
        raise ValueError(
            f"Coverage index does not exist: {index_path}"
        )
    if not (downloads / "download_manifest.csv").is_file():
        raise ValueError(
            f"Download manifest missing in {downloads}"
        )

    terrain_manifest = json.loads(
        (terrain / "tile_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    # Keep original manifest indexes even if task selection changes later.
    tasks = list(enumerate(terrain_manifest["tiles"]))
    if args.test_tiles:
        tasks = tasks[:args.test_tiles]
    if not tasks:
        raise ValueError("No terrain tiles selected.")

    worker_count = min(args.workers, len(tasks))

    print("Validating MTK layers and vegetation assets...")
    assets, _ = load_assets(asset_folder)

    # Open/validate these once in the parent to report missing layers clearly.
    # Workers reopen their own handles; parent handles are not passed to them.
    validation_mtk = open_mtk(mtk_folder)
    del validation_mtk

    output.mkdir(parents=True, exist_ok=True)
    (output / "vegetation").mkdir()

    inspection, inspection_layer = create_inspection(
        output / "vegetation_areas.gpkg"
    )

    result_manifest = {
        "format": "vegetation_manifest_v1",
        "crs": "EPSG:3067",
        "source_terrain_manifest": str(
            terrain / "tile_manifest.json"
        ),
        "asset_library": str(
            asset_folder / "vegetation.json"
        ),
        "asset_table": [
            {
                "numeric_id": asset["numeric_id"],
                "id": asset["id"],
                "definition": asset["definition_path"],
            }
            for asset in assets
        ],
        "config": CONFIG,
        "workers": worker_count,
        "test_tiles": args.test_tiles,
        "completed_tile_count": 0,
        "tiles": [],
        "warnings": [
            "Ground DEM acquisition consistency is not verified.",
            "Class partitions use nearest symbols, not equal-area.",
            "Inspection areas are split at terrain tile boundaries.",
            "Smoothed raster boundaries are approximations.",
            "LAS files are reread for intersecting processing tiles.",
            "Buffered processing can give different classification "
            "near task boundaries than global connected-area processing.",
            "Plant base positions respect exclusions; crowns can extend "
            "beyond those positions.",
        ],
    }

    print(
        "WARNING: Existing DEMs may contain mixed-year ground data. "
        "Vegetation points are oldest-filtered, but DEMs are not rebuilt."
    )
    print(
        f"Starting {worker_count} worker(s). "
        "Each worker initializes its own coverage and GIS indexes."
    )

    total_plants = 0
    tile_results = {}

    try:
        with tempfile.TemporaryDirectory(
            prefix="_vegetation_work_",
            dir=str(output),
        ) as temporary:
            directory = Path(temporary)
            vrt = build_ground_vrt(
                terrain, terrain_manifest, directory
            )

            initializer_args = (
                str(downloads),
                str(index_path),
                str(mtk_folder),
                str(asset_folder),
                str(output),
                str(vrt),
                CONFIG,
            )

            completed_count = 0

            for result in parallel_tile_results(
                tasks,
                worker_count,
                initializer_args,
            ):
                completed_count += 1
                entry = result["manifest_entry"]

                # Only the main process writes the shared GeoPackage.
                inspection.StartTransaction()
                try:
                    for serialized in result["areas"]:
                        area = {
                            key: value
                            for key, value in serialized.items()
                            if key != "geometry_wkb"
                        }
                        area["geometry"] = from_wkb(
                            serialized["geometry_wkb"]
                        )
                        write_area(
                            inspection_layer,
                            result["tile_name"],
                            area,
                        )

                    inspection.CommitTransaction()
                except Exception:
                    inspection.RollbackTransaction()
                    raise

                total_plants += entry["plant_count"]
                tile_results[result["number"]] = entry

                # Completion order differs, but manifest ordering is stable.
                result_manifest["tiles"] = [
                    tile_results[number]
                    for number in sorted(tile_results)
                ]
                result_manifest["completed_tile_count"] = (
                    completed_count
                )
                result_manifest["total_plants"] = total_plants

                print(
                    f"[{completed_count}/{len(tasks)}] "
                    f"Finished {result['tile_name']}: "
                    f"{entry['retained_vegetation_points']} points, "
                    f"{entry['area_count']} areas, "
                    f"{entry['plant_count']} plants",
                    flush=True,
                )

                inspection.FlushCache()
                (output / "vegetation_manifest.json").write_text(
                    json.dumps(result_manifest, indent=2),
                    encoding="utf-8",
                )

    finally:
        inspection_layer = None
        inspection = None

    (output / "vegetation_manifest.json").write_text(
        json.dumps(result_manifest, indent=2),
        encoding="utf-8",
    )

    (output / "VEG_FORMAT.txt").write_text(
        """VEG1 binary format: little endian

Header:
  char[4] signature: VEG1
  float64 tile xmin (absolute easting)
  float64 tile ymax (absolute northing)
  float32 cell size metres
  uint32 occupied cell count
  uint32 total plant count

For each occupied cell:
  uint32 column
  uint32 row
  uint32 plant count

For each plant:
  uint32 asset ID (vegetation_manifest.json asset_table)
  float32 local east metres
  float32 local south metres
  float32 absolute elevation metres
  float32 height scale
  float32 width scale
  float32 yaw radians
  float32 RGB tint multiplier
  float32 stable density priority [0,1]

Plant scale multiplies the corresponding asset card dimensions.
Asset pivot_uv defines the trunk-base pivot.

For viewer placement:
  easting = tile xmin + local east
  northing = tile ymax - local south
  elevation = stored absolute elevation

Convert these to the viewer's shared terrain origin.
Cell culling bounds must include tree height and crown extent.
Species textures and billboard geometry are loaded separately.

This file format does not include terrain or vegetation textures.
""",
        encoding="utf-8",
    )

    print(
        f"\nFinished: {total_plants} plants across "
        f"{len(tile_results)} tiles in {output}"
    )


if __name__ == "__main__":
    # Required for Windows multiprocessing and packaged executables.
    multiprocessing.freeze_support()

    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nInterrupted. Running worker tasks may take a little "
            "time to finish safely.",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
