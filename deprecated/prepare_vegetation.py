#!/usr/bin/env python3
"""
Prepare LiDAR-derived vegetation areas and tiled plant placements.

Example:
    python prepare_vegetation.py \
        --terrain espoo_terrain \
        --downloads espoo1965_merged_pointcloud \
        --mtk MTK_2009_clipped \
        --assets vegetation_assets \
        -o vegetation_output --test-tiles 10

Arguments:
    --terrain      Folder containing tile_manifest.json and DEM tiles.
    --downloads    Folder containing download_manifest.csv, pointclouds/ (previously downloaded pointcloud map sheets, not merged point cloud),
                   and _index/2008_latest.shp.
    --mtk          Folder containing the clipped MTK GeoPackages.
    --assets       Folder containing vegetation.json and generated assets.
    -o, --output   New or empty output folder.
    --test-tiles   Process the first N manifest tiles; 0 means all.
    --index        Override the coverage shapefile location.

All GIS and LiDAR inputs must use EPSG:3067.
Plant/area generation settings are in CONFIG below.

Requires:
    GDAL Python bindings, numpy, scipy, rasterio, laspy+lazrs, shapely >= 2.
"""

import argparse
import csv
import hashlib
import json
import math
import struct
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import laspy
import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage
from scipy.spatial import cKDTree
import shapely
from shapely import (
    from_wkb, to_wkb, intersects_xy, contains_xy,
    union_all, make_valid
)
from shapely.geometry import box, Point, shape
from shapely.strtree import STRtree
from rasterio.features import rasterize, shapes
from osgeo import gdal, ogr, osr


# ============================================================================
# EDITABLE CONFIGURATION
# ============================================================================

CONFIG = {
    "seed": 12345,

    # Raster used internally to derive coverage and height.
    "resolution": 2.0,
    "processing_buffer": 20.0,

    # LAS classification codes accepted as vegetation.
    "vegetation_classes": [3, 4, 5],
    "isolation_radius": 5.0,

    # Reject implausible heights rather than treating them as giant trees.
    "minimum_height": 0.5,
    "tree_height_threshold": 3.0,
    "maximum_height": 45.0,
    "height_percentile": 85.0,

    # Coverage cleanup.
    "closing_iterations": 1,
    "minimum_area_m2": 12.0,
    "polygon_smoothing_metres": 1.0,
    "polygon_simplification_metres": 0.5,

    # Ground interpolation for small DEM holes. Very large gaps are not
    # guessed; vegetation without a usable ground elevation is skipped.
    "ground_fill_distance_metres": 150.0,

    "building_buffer_metres": 3.0,
    "road_default_radius_metres": 3.0,

    # Add overrides later, using actual tieviiva LUOKKA values:
    # 12345: 5.0 means a five-metre buffer on each side.
    "road_radius_by_luokka": {},

    "classification_max_distance": 500.0,

    # Viewer subdivision within each terrain tile.
    "viewer_cell_size": 25.0,

    # Placement spacing and deterministic size variation.
    "tree_spacing_metres": 6.0,
    "shrub_spacing_metres": 2.5,
    "placement_probability": 0.85,
    "height_variation": [0.85, 1.10],
    "width_variation": [0.85, 1.15],

    # Areas outside these optional allow layers remain eligible when True.
    # Set False to restrict vegetation to parks/cemeteries, for example.
    "allow_detected_vegetation_elsewhere": True,

    "allow_layers": [
        ("mtk-muut_2009_clipped.gpkg", "puisto"),
        ("mtk-muut_2009_clipped.gpkg", "hautausmaa"),
    ],

    # Additional polygon exclusions can be added here.
    "exclude_layers": [
        ("mtk-rakennus_2009_clipped.gpkg", "rakennus", "buildings"),
        ("mtk-vakavesi_2009_clipped.gpkg", "meri", "water"),
        ("mtk-vakavesi_2009_clipped.gpkg", "jarvi", "water"),
    ],

    "road_file": "mtk-tie_2009_clipped.gpkg",
    "road_layer": "tieviiva",

    "classification_file": "mtk-muut_2009_clipped.gpkg",
    "classification_layer": "metsamaankasvillisuus",
    "classification_field": "LUOKKA",

    # LUOKKA -> generator key.
    "class_generators": {
        32710: "conifer_mix",
        32713: "deciduous",
        32714: "deciduous_32714",
    },

    # Separate 32714 entry allows later custom behavior.
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


# ============================================================================
# GIS HELPERS
# ============================================================================

def crs_3067():
    """Return EPSG:3067 with traditional easting/northing axis order."""
    crs = osr.SpatialReference()
    crs.ImportFromEPSG(EPSG)
    crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return crs


def validate_crs(crs, description):
    """Reject unknown or mismatching CRS metadata."""
    if crs is None:
        raise ValueError(f"No CRS: {description}")
    candidate = crs.Clone()
    candidate.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    if not candidate.IsSame(crs_3067()):
        raise ValueError(f"Expected EPSG:3067: {description}")


def polygon_parts(geometry):
    """Yield polygon pieces from polygons, multipolygons, or collections."""
    if geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
    elif hasattr(geometry, "geoms"):
        for child in geometry.geoms:
            yield from polygon_parts(child)


class VectorLayer:
    """Keep an OGR layer open for repeated spatially filtered tile queries."""

    def __init__(self, filename, layer_name):
        self.dataset = ogr.Open(str(filename), 0)
        if self.dataset is None:
            raise ValueError(f"Cannot open {filename}")
        self.layer = self.dataset.GetLayerByName(layer_name)
        if self.layer is None:
            raise ValueError(f"Layer {layer_name!r} missing in {filename}")
        validate_crs(self.layer.GetSpatialRef(), f"{filename}:{layer_name}")

    def query(self, bounds, field=None):
        """Yield repaired geometries and an optional attribute value."""
        self.layer.SetSpatialFilterRect(*bounds)
        self.layer.ResetReading()
        for feature in self.layer:
            geometry = feature.GetGeometryRef()
            if geometry is None:
                continue
            result = make_valid(from_wkb(bytes(geometry.ExportToWkb())))
            if result.is_empty:
                continue
            value = feature.GetField(field) if field else None
            yield result, value


def open_mtk(folder):
    """Open configured exclusions, roads, allowed areas, and class points."""
    exclusions = [
        (VectorLayer(folder / filename, layer), treatment)
        for filename, layer, treatment in CONFIG["exclude_layers"]
    ]
    allowed = [
        VectorLayer(folder / filename, layer)
        for filename, layer in CONFIG["allow_layers"]
    ]
    roads = VectorLayer(
        folder / CONFIG["road_file"], CONFIG["road_layer"]
    )
    classes = VectorLayer(
        folder / CONFIG["classification_file"],
        CONFIG["classification_layer"],
    )

    field = CONFIG["classification_field"]
    if classes.layer.GetLayerDefn().GetFieldIndex(field) < 0:
        raise ValueError(f"Classification field {field!r} is missing.")

    return exclusions, allowed, roads, classes


def tile_exclusions(bounds, mtk):
    """Construct road/building/water exclusions for a buffered tile."""
    exclusions, allowed, roads, _ = mtk
    pieces = []

    # Query farther out so a buffer from a feature outside the tile is found.
    extra = max(
        CONFIG["building_buffer_metres"],
        CONFIG["road_default_radius_metres"],
        *CONFIG["road_radius_by_luokka"].values(),
    )
    query_bounds = box(*bounds).buffer(extra).bounds

    for layer, treatment in exclusions:
        for geometry, _ in layer.query(query_bounds):
            if treatment == "buildings":
                geometry = geometry.buffer(CONFIG["building_buffer_metres"])
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

    exclusion = union_all(pieces) if pieces else shapely.GeometryCollection()

    if not CONFIG["allow_detected_vegetation_elsewhere"]:
        allow_shapes = [
            geometry
            for layer in allowed
            for geometry, _ in layer.query(bounds)
        ]
        allow = union_all(allow_shapes) if allow_shapes else shapely.GeometryCollection()
        exclusion = union_all([
            exclusion, box(*bounds).difference(allow)
        ])

    return make_valid(exclusion)


# ============================================================================
# OLDEST-ACQUISITION OWNERSHIP
# ============================================================================

def archive_relative(value):
    """Normalize an archive path to YEAR/subdirectories/filename."""
    text = str(value).strip().replace("\\", "/").lstrip("/")
    prefix = "mml/laserkeilaus/2008_latest/"
    if text.startswith(prefix):
        text = text[len(prefix):]
    return text


def load_oldest_sources(downloads, index_path):
    """
    Locate downloaded LAS/LAZ files, read coverage polygons, and assign
    spatial ownership to the oldest acquisition year.

    The coverage index may use another CRS; its polygons are transformed
    to EPSG:3067 before spatial comparisons.

    Same-year files may overlap. Later-year coverage is retained only
    outside all earlier-year downloaded coverage.
    """
    downloads = Path(downloads)
    index_path = Path(index_path)
    manifest_path = downloads / "download_manifest.csv"

    if not manifest_path.is_file():
        raise ValueError(
            f"Download manifest does not exist: {manifest_path}"
        )

    # Read the manifest using a context manager so the file is closed.
    with manifest_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError("The download manifest is empty.")

    # Match archive-relative paths to local point-cloud files.
    requested = {}

    for row in rows:
        relative = archive_relative(row.get("path") or "")

        # Older manifest versions may contain URLs but no path column.
        if not relative:
            url = row.get("url", "")
            marker = "/2008_latest/"

            if marker not in url:
                raise ValueError(
                    f"Cannot parse archive path from manifest URL: {url}"
                )

            relative = archive_relative(url.split(marker, 1)[1])

        filename = Path(relative).name

        # Prefer the current flat pointclouds folder.
        local = downloads / "pointclouds" / filename

        # Fall back to the manifest's recorded location.
        if not local.is_file():
            recorded_path = row.get("local_path", "")
            if recorded_path:
                local = Path(recorded_path)

        if not local.is_file():
            raise ValueError(
                f"Missing downloaded point cloud: {filename}\n"
                f"Expected it in: {downloads / 'pointclouds'}"
            )

        year = int(row["year"])

        requested[relative] = {
            "file": local,
            "year": year,
            "footprints": [],
        }

    # Open the coverage shapefile BEFORE accessing its layer or CRS.
    dataset = ogr.Open(str(index_path), 0)

    if dataset is None:
        raise ValueError(
            f"Cannot open coverage index: {index_path}"
        )

    layer = dataset.GetLayer(0)

    if layer is None:
        raise ValueError(
            f"Coverage index has no readable layer: {index_path}"
        )

    # Read and normalize the source coordinate reference system.
    index_crs = layer.GetSpatialRef()

    if index_crs is None:
        raise ValueError(
            f"Coverage index has no CRS: {index_path}\n"
            "Ensure its matching .prj file is beside the .shp file."
        )

    index_crs = index_crs.Clone()
    index_crs.SetAxisMappingStrategy(
        osr.OAMS_TRADITIONAL_GIS_ORDER
    )

    destination_crs = crs_3067()
    index_transform = None

    if not index_crs.IsSame(destination_crs):
        print(
            "Coverage index uses another CRS; "
            "transforming its polygons to EPSG:3067."
        )

        index_transform = osr.CoordinateTransformation(
            index_crs, destination_crs
        )

    # Locate the path attribute without assuming capitalization.
    definition = layer.GetLayerDefn()

    fields = {
        definition.GetFieldDefn(i).GetName().lower():
        definition.GetFieldDefn(i).GetName()
        for i in range(definition.GetFieldCount())
    }

    if "path" not in fields:
        raise ValueError(
            "Coverage index has no 'path' attribute."
        )

    # Collect only polygons corresponding to downloaded files.
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

        # Modify a clone, not the geometry owned by the source feature.
        geometry = geometry.Clone()

        if index_transform is not None:
            status = geometry.Transform(index_transform)

            if status != 0:
                raise RuntimeError(
                    f"Could not transform coverage polygon: {relative}"
                )

        footprint = make_valid(
            from_wkb(bytes(geometry.ExportToWkb()))
        )

        if not footprint.is_empty:
            requested[relative]["footprints"].append(footprint)

    # Close the source shapefile before constructing ownership geometries.
    layer = None
    dataset = None

    # Combine multiple index polygons belonging to the same source file.
    by_year = defaultdict(list)

    for relative, entry in requested.items():
        if not entry["footprints"]:
            raise ValueError(
                f"No index footprint found for downloaded file: {relative}"
            )

        entry["footprint"] = make_valid(
            union_all(entry.pop("footprints"))
        )

        by_year[entry["year"]].append(entry)

    older_coverage = shapely.GeometryCollection()
    sources = []

    # All files within a year are tested against the SAME earlier coverage.
    # Consequently, overlapping files from the same year remain eligible.
    for year in sorted(by_year):
        year_entries = by_year[year]

        for entry in year_entries:
            owner = make_valid(
                entry["footprint"].difference(older_coverage)
            )

            if not owner.is_empty:
                entry["owner"] = owner
                sources.append(entry)

        # Only after processing the entire year do we extend old coverage.
        older_coverage = make_valid(
            union_all([
                older_coverage,
                *[
                    entry["footprint"]
                    for entry in year_entries
                ],
            ])
        )

    if not sources:
        raise ValueError(
            "No usable oldest-year point-cloud coverage found."
        )

    # Spatial index used to find source files for each processing tile.
    source_tree = STRtree([
        entry["owner"] for entry in sources
    ])

    print(
        f"Prepared oldest-year ownership for "
        f"{len(sources)} point-cloud files."
    )

    return sources, source_tree



# ============================================================================
# GROUND AND LIDAR PROCESSING
# ============================================================================

def build_ground_vrt(terrain, manifest, directory):
    """Reference DEM tiles without loading the complete terrain into memory."""
    files = [
        str(terrain / tile["dem"])
        for tile in manifest["tiles"] if tile.get("dem")
    ]
    if not files:
        raise ValueError("Terrain manifest has no DEM files.")

    for filename in files:
        with rasterio.open(filename) as dataset:
            if dataset.crs != rasterio.crs.CRS.from_epsg(EPSG):
                raise ValueError(f"DEM must use EPSG:3067: {filename}")

    path = directory / "ground.vrt"
    result = gdal.BuildVRT(
        str(path), files,
        options=gdal.BuildVRTOptions(VRTNodata=-9999),
    )
    if result is None:
        raise RuntimeError("Could not build ground DEM VRT.")
    result = None
    return path


def ground_grid(vrt, bounds, resolution, directory):
    """
    Resample the ground to the vegetation grid and fill local ground gaps.

    Returns NaN where interpolation still has no ground support.
    """
    path = directory / "tile_ground.tif"
    result = gdal.Warp(
        str(path), str(vrt),
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
    gdal.FillNodata(
        targetBand=band,
        maskBand=band.GetMaskBand(),
        maxSearchDist=CONFIG["ground_fill_distance_metres"] / resolution,
        smoothingIterations=0,
    )
    result.FlushCache()
    band = None
    result = None

    with rasterio.open(path) as dataset:
        data = dataset.read(1, masked=True).astype(float).filled(np.nan)
        transform = dataset.transform
    return data, transform


def read_vegetation(bounds, sources, source_tree):
    """Read eligible oldest-owned vegetation points using chunked LAS reads."""
    query = box(*bounds)
    arrays = []

    for source_index in source_tree.query(query, predicate="intersects"):
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
                    raise ValueError(f"LAS CRS mismatch: {entry['file']}")

            for points in reader.chunk_iterator(1_000_000):
                classes = np.asarray(points.classification)
                keep = (
                    np.isin(classes, CONFIG["vegetation_classes"])
                    & ~np.asarray(points.withheld, dtype=bool)
                )
                if not np.any(keep):
                    continue

                x = np.asarray(points.x)[keep]
                y = np.asarray(points.y)[keep]
                z = np.asarray(points.z)[keep]

                inside = (
                    np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
                    & (x >= bounds[0]) & (x < bounds[2])
                    & (y >= bounds[1]) & (y < bounds[3])
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

    points = np.concatenate(arrays)

    # Remove exact duplicates before the isolation test. A duplicated
    # observation should not count as its own supporting neighbor.
    points = np.unique(points, axis=0)
    if len(points) < 2:
        return np.empty((0, 3))

    distances, _ = cKDTree(points[:, :2]).query(
        points[:, :2], k=2,
        distance_upper_bound=CONFIG["isolation_radius"],
    )
    return points[np.isfinite(distances[:, 1])]


def vegetation_grid(points, ground, transform, exclusion):
    """Estimate vegetation height per cell and separate shrub/tree coverage."""
    height, width = ground.shape
    representative = np.full((height, width), np.nan)
    count = np.zeros((height, width), dtype=np.int32)

    if len(points):
        cols = np.floor(
            (points[:, 0] - transform.c) / transform.a
        ).astype(int)
        rows = np.floor(
            (points[:, 1] - transform.f) / transform.e
        ).astype(int)

        inside = (
            (cols >= 0) & (cols < width)
            & (rows >= 0) & (rows < height)
        )
        points, cols, rows = points[inside], cols[inside], rows[inside]

        above_ground = points[:, 2] - ground[rows, cols]
        valid = (
            np.isfinite(above_ground)
            & (above_ground >= CONFIG["minimum_height"])
            & (above_ground <= CONFIG["maximum_height"])
        )
        ids = rows[valid] * width + cols[valid]
        heights = above_ground[valid]

        order = np.argsort(ids)
        ids, heights = ids[order], heights[order]
        if len(ids):
            starts = np.r_[0, np.flatnonzero(np.diff(ids)) + 1]
            ends = np.r_[starts[1:], len(ids)]
            for start, end in zip(starts, ends):
                index = int(ids[start])
                r, c = divmod(index, width)
                representative[r, c] = np.percentile(
                    heights[start:end], CONFIG["height_percentile"]
                )
                count[r, c] = end - start

    occupied = np.isfinite(representative)

    if CONFIG["closing_iterations"]:
        occupied = ndimage.binary_closing(
            occupied, iterations=CONFIG["closing_iterations"]
        )

    # Populate small coverage gaps with the nearest measured cell height.
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
        occupied & (
            representative >= CONFIG["tree_height_threshold"]
        )
    ] = 2

    # Remove small connected regions separately for shrubs and trees.
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
# MTK CLASSIFICATION AND AREA OUTPUT
# ============================================================================

def load_class_points(layer):
    """Load the relatively small MTK vegetation-symbol point dataset."""
    coords, codes = [], []
    layer.layer.SetSpatialFilter(None)
    layer.layer.ResetReading()

    for feature in layer.layer:
        geometry = feature.GetGeometryRef()
        if geometry is None:
            continue
        if ogr.GT_Flatten(geometry.GetGeometryType()) != ogr.wkbPoint:
            continue
        try:
            code = int(feature.GetField(CONFIG["classification_field"]))
        except (TypeError, ValueError):
            continue
        coords.append((geometry.GetX(), geometry.GetY()))
        codes.append(code)

    coords = np.asarray(coords, dtype=float).reshape(-1, 2)
    codes = np.asarray(codes, dtype=np.int32)
    tree = cKDTree(coords) if len(coords) else None
    return coords, codes, tree


def classify_grid(kinds, transform, class_points):
    """
    Assign LUOKKA by connected vegetation area.

    With multiple distinct classes inside an area, use nearest contained
    symbols. Without contained symbols, find the nearest symbol to the area.
    """
    coords, codes, tree = class_points
    assigned = np.zeros(kinds.shape, dtype=np.int32)

    for kind in (1, 2):
        labels, number = ndimage.label(kinds == kind)

        for label in range(1, number + 1):
            rows, cols = np.nonzero(labels == label)
            mask = labels == label

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

            if tree is None:
                continue

            xmin, ymin, xmax, ymax = area.bounds
            center = ((xmin + xmax) / 2, (ymin + ymax) / 2)
            radius = math.hypot(xmax - xmin, ymax - ymin) / 2

            candidates = tree.query_ball_point(
                center, radius + CONFIG["classification_max_distance"]
            )
            if not candidates:
                continue

            indexes = np.asarray(candidates, dtype=int)
            inside = contains_xy(
                area, coords[indexes, 0], coords[indexes, 1]
            )
            contained = indexes[inside]

            if len(contained):
                distinct = np.unique(codes[contained])
                if len(distinct) == 1:
                    assigned[rows, cols] = distinct[0]
                else:
                    xy = np.column_stack((
                        transform.c + (cols + 0.5) * transform.a,
                        transform.f + (rows + 0.5) * transform.e,
                    ))
                    _, nearest = cKDTree(coords[contained]).query(xy)
                    assigned[rows, cols] = codes[contained[nearest]]
            else:
                nearest_index = min(
                    candidates,
                    key=lambda i: area.distance(Point(coords[i])),
                )
                distance = area.distance(Point(coords[nearest_index]))
                if distance <= CONFIG["classification_max_distance"]:
                    assigned[rows, cols] = codes[nearest_index]

    return assigned


def create_inspection(path):
    """Create the vegetation-area GeoPackage and its attribute fields."""
    dataset = ogr.GetDriverByName("GPKG").CreateDataSource(str(path))
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
    """Polygonize and smooth each class/type while reapplying exclusions."""
    areas = []
    core_polygon = box(*core)

    for kind in (1, 2):
        for code in np.unique(classes[kinds == kind]):
            mask = (kinds == kind) & (classes == code)
            if not np.any(mask):
                continue

            geometries = [
                shape(geometry)
                for geometry, _ in shapes(
                    mask.astype(np.uint8), mask=mask, transform=transform
                )
            ]
            geometry = union_all(geometries)
            smooth = CONFIG["polygon_smoothing_metres"]
            if smooth:
                geometry = geometry.buffer(smooth).buffer(-smooth)

            geometry = geometry.simplify(
                CONFIG["polygon_simplification_metres"],
                preserve_topology=True,
            )
            geometry = make_valid(
                geometry.intersection(core_polygon).difference(exclusion)
            )

            generator = (
                "shrubs" if kind == 1 else
                CONFIG["class_generators"].get(int(code), "unknown")
            )

            for polygon in polygon_parts(geometry):
                if polygon.area < CONFIG["minimum_area_m2"]:
                    continue

                # Compute representative height within this particular part.
                part_mask = rasterize(
                    [(polygon, 1)],
                    out_shape=kinds.shape,
                    transform=transform,
                    dtype="uint8",
                ).astype(bool)
                selected = heights[part_mask & mask]
                selected = selected[np.isfinite(selected)]
                height = float(np.median(selected)) if len(selected) else (
                    1.5 if kind == 1 else 12.0
                )
                areas.append({
                    "geometry": polygon,
                    "kind": kind,
                    "code": int(code),
                    "generator": generator,
                    "height": height,
                })

    return areas


def write_area(layer, tile_name, area):
    """Append one inspection polygon and its semantic attributes."""
    feature = ogr.Feature(layer.GetLayerDefn())
    geometry = ogr.CreateGeometryFromWkb(to_wkb(area["geometry"]))
    geometry = ogr.ForceToMultiPolygon(geometry)
    feature.SetGeometry(geometry)
    feature.SetField("tile", tile_name)
    feature.SetField("LUOKKA", area["code"])
    feature.SetField("plant_type", "shrub" if area["kind"] == 1 else "tree")
    feature.SetField("generator", area["generator"])
    feature.SetField("height_m", area["height"])
    feature.SetField("area_m2", area["geometry"].area)
    layer.CreateFeature(feature)


# ============================================================================
# ASSETS, PLACEMENTS, AND BINARY OUTPUT
# ============================================================================

def load_assets(folder):
    """Load generated vegetation assets and assign stable numeric IDs."""
    library = json.loads(
        (folder / "vegetation.json").read_text(encoding="utf-8")
    )
    assets = []
    by_species = defaultdict(list)

    for item in library["assets"]:
        definition = folder / item["definition"]
        asset = json.loads(definition.read_text(encoding="utf-8"))
        asset["numeric_id"] = len(assets)
        asset["definition_path"] = str(definition.resolve())
        assets.append(asset)
        by_species[asset["species"]].append(asset["numeric_id"])

    for mixture in CONFIG["generators"].values():
        for species in mixture:
            if species not in by_species:
                raise ValueError(f"Missing asset species: {species}")

    return assets, by_species


def stable_rng(kind, ix, iy):
    """Return a repeatable generator anchored to a global placement lattice."""
    text = f"{CONFIG['seed']}:{kind}:{ix}:{iy}".encode("ascii")
    seed = int.from_bytes(hashlib.blake2b(text, digest_size=8).digest(), "little")
    return np.random.default_rng(seed)


def sample_cell(array, transform, x, y):
    """Read the nearest working-raster cell at a map coordinate."""
    column = int(math.floor((x - transform.c) / transform.a))
    row = int(math.floor((y - transform.f) / transform.e))
    if not (0 <= row < array.shape[0] and 0 <= column < array.shape[1]):
        return np.nan
    return array[row, column]


def generate_plants(core, areas, ground, heights, transform, assets, by_species):
    """
    Place plants on deterministic jittered lattices.

    The lattice is global, not restarted independently at each tile.
    Small jitter provides approximate spacing, not exact Poisson sampling.
    """
    plants = []
    xmin, ymin, xmax, ymax = core

    for kind in (1, 2):
        eligible = [area for area in areas if area["kind"] == kind]
        if not eligible:
            continue

        spatial = STRtree([area["geometry"] for area in eligible])
        spacing = CONFIG[
            "shrub_spacing_metres" if kind == 1 else "tree_spacing_metres"
        ]

        for ix in range(math.floor(xmin / spacing), math.ceil(xmax / spacing)):
            for iy in range(math.floor(ymin / spacing), math.ceil(ymax / spacing)):
                rng = stable_rng(kind, ix, iy)
                x = (ix + 0.5 + rng.uniform(-0.20, 0.20)) * spacing
                y = (iy + 0.5 + rng.uniform(-0.20, 0.20)) * spacing

                if not (xmin <= x < xmax and ymin <= y < ymax):
                    continue
                if rng.random() > CONFIG["placement_probability"]:
                    continue

                matches = spatial.query(Point(x, y), predicate="intersects")
                if not len(matches):
                    continue
                area = eligible[int(matches[0])]

                elevation = sample_cell(ground, transform, x, y)
                if not np.isfinite(elevation):
                    continue

                measured = sample_cell(heights, transform, x, y)
                height = measured if np.isfinite(measured) else area["height"]
                height *= rng.uniform(*CONFIG["height_variation"])

                # Maintain shrub/tree separation even with random variation.
                if kind == 1:
                    height = np.clip(height, 0.5, 2.95)
                else:
                    height = np.clip(height, 3.05, CONFIG["maximum_height"])

                mixture = CONFIG["generators"][area["generator"]]
                species = list(mixture)
                weights = np.asarray(list(mixture.values()), dtype=float)
                weights /= weights.sum()
                selected_species = rng.choice(species, p=weights)

                asset_id = int(rng.choice(by_species[selected_species]))
                asset = assets[asset_id]
                height_scale = height / asset["plant_height_metres"]

                plants.append((
                    asset_id,
                    float(x - xmin),
                    float(ymax - y),
                    float(elevation),
                    float(height_scale),
                    float(height_scale * rng.uniform(*CONFIG["width_variation"])),
                    float(rng.uniform(0, math.tau)),
                    float(rng.uniform(0.90, 1.08)),
                    float(rng.random()),
                ))

    return plants


def write_veg(path, core, plants):
    """
    Write VEG1 placement data grouped into viewer cells.

    Header:
        char[4]  "VEG1"
        float64  xmin, ymax
        float32  cell size
        uint32   occupied-cell count
        uint32   total plant count

    Each cell:
        uint32 cell-column, cell-row, plant-count

    Each plant:
        uint32 asset ID
        float32 local-east, local-south, ABSOLUTE elevation,
                height-scale, width-scale, yaw-radians, tint, priority

    The viewer converts absolute elevation/map origin to its own terrain origin.
    """
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
            core[0], core[3], cell_size, len(cells), len(plants),
        ))

        for (column, row), members in sorted(cells.items()):
            output.write(struct.pack("<3I", column, row, len(members)))
            for plant in members:
                output.write(struct.pack("<I8f", *plant))


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare oldest-LiDAR vegetation areas and viewer tiles."
    )
    parser.add_argument("--terrain", required=True)
    parser.add_argument("--downloads", required=True)
    parser.add_argument("--mtk", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--test-tiles", type=int, default=0)
    parser.add_argument("--index")
    args = parser.parse_args()

    if args.test_tiles < 0:
        parser.error("--test-tiles cannot be negative.")

    gdal.UseExceptions()
    ogr.UseExceptions()
    gdal.SetCacheMax(256 * 1024 * 1024)

    terrain = Path(args.terrain).resolve()
    downloads = Path(args.downloads).resolve()
    mtk_folder = Path(args.mtk).resolve()
    asset_folder = Path(args.assets).resolve()
    output = Path(args.output).resolve()
    index_path = (
        Path(args.index).resolve() if args.index
        else downloads / "_index" / "2008_latest.shp"
    )

    if output.exists() and (
        not output.is_dir() or any(output.iterdir())
    ):
        parser.error("Output folder must be empty or not yet exist.")

    terrain_manifest = json.loads(
        (terrain / "tile_manifest.json").read_text(encoding="utf-8")
    )
    selected_tiles = terrain_manifest["tiles"]
    if args.test_tiles:
        selected_tiles = selected_tiles[:args.test_tiles]

    print("Loading acquisition footprints and oldest-year ownership...")
    sources, source_tree = load_oldest_sources(downloads, index_path)

    print("Opening MTK layers and asset library...")
    mtk = open_mtk(mtk_folder)
    class_points = load_class_points(mtk[3])
    assets, by_species = load_assets(asset_folder)

    output.mkdir(parents=True, exist_ok=True)
    vegetation_dir = output / "vegetation"
    vegetation_dir.mkdir()

    inspection, inspection_layer = create_inspection(
        output / "vegetation_areas.gpkg"
    )

    result_manifest = {
        "format": "vegetation_manifest_v1",
        "crs": "EPSG:3067",
        "source_terrain_manifest": str(terrain / "tile_manifest.json"),
        "asset_library": str(asset_folder / "vegetation.json"),
        "asset_table": [
            {
                "numeric_id": asset["numeric_id"],
                "id": asset["id"],
                "definition": asset["definition_path"],
            }
            for asset in assets
        ],
        "config": CONFIG,
        "test_tiles": args.test_tiles,
        "tiles": [],
        "warnings": [
            "Ground DEM acquisition consistency is not independently verified.",
            "Area partitions are nearest-symbol partitions, not equal-area.",
            "Inspection areas are split at terrain tile boundaries.",
            "Raster-derived smoothed boundaries are approximations.",
            "LAS files are reread per intersecting processing tile.",
        ],
    }

    print(
        "WARNING: Existing DEMs may contain mixed-year ground observations. "
        "This script filters vegetation acquisitions, not the source DEM."
    )

    total_plants = 0

    try:
        with tempfile.TemporaryDirectory(
            prefix="_vegetation_work_", dir=str(output)
        ) as temporary:
            directory = Path(temporary)
            vrt = build_ground_vrt(terrain, terrain_manifest, directory)

            for number, tile in enumerate(selected_tiles):
                if not tile.get("core_bounds"):
                    raise ValueError(
                        f"Tile {tile['tile']} has no core_bounds."
                    )

                core = tuple(tile["core_bounds"])
                buffer = CONFIG["processing_buffer"]
                resolution = CONFIG["resolution"]

                # Anchor processing grids globally so cell boundaries are
                # consistent between tiles, even if terrain cores are fractional.
                bounds = (
                    math.floor((core[0] - buffer) / resolution) * resolution,
                    math.floor((core[1] - buffer) / resolution) * resolution,
                    math.ceil((core[2] + buffer) / resolution) * resolution,
                    math.ceil((core[3] + buffer) / resolution) * resolution,
                )

                print(
                    f"[{number + 1}/{len(selected_tiles)}] "
                    f"{tile['tile']}"
                )

                ground, transform = ground_grid(
                    vrt, bounds, resolution, directory
                )
                exclusion = tile_exclusions(bounds, mtk)
                points = read_vegetation(bounds, sources, source_tree)

                heights, kinds = vegetation_grid(
                    points, ground, transform, exclusion
                )
                classes = classify_grid(kinds, transform, class_points)
                areas = make_areas(
                    kinds, classes, heights, transform, core, exclusion
                )

                for area in areas:
                    write_area(inspection_layer, tile["tile"], area)

                plants = generate_plants(
                    core, areas, ground, heights, transform,
                    assets, by_species,
                )

                # Use original manifest numbering, matching prepare_terrain.py.
                filename = f"tile_{number:06d}.veg"
                write_veg(vegetation_dir / filename, core, plants)
                total_plants += len(plants)

                result_manifest["tiles"].append({
                    "tile": tile["tile"],
                    "terrain_filename": f"tiles/tile_{number:06d}.trn",
                    "vegetation_filename": f"vegetation/{filename}",
                    "core_bounds": core,
                    "area_count": len(areas),
                    "plant_count": len(plants),
                    "retained_vegetation_points": len(points),
                })

                print(
                    f"  {len(points)} points, {len(areas)} areas, "
                    f"{len(plants)} plants"
                )

                inspection.FlushCache()
                (output / "vegetation_manifest.json").write_text(
                    json.dumps(result_manifest, indent=2),
                    encoding="utf-8",
                )

    finally:
        inspection_layer = None
        inspection = None

    result_manifest["total_plants"] = total_plants
    (output / "vegetation_manifest.json").write_text(
        json.dumps(result_manifest, indent=2), encoding="utf-8"
    )

    (output / "VEG_FORMAT.txt").write_text(
        """VEG1 binary format; little endian

Header:
  4 bytes: VEG1
  float64 xmin
  float64 ymax
  float32 cell size metres
  uint32 occupied cell count
  uint32 total plant count

For each occupied cell:
  uint32 column
  uint32 row
  uint32 plant count

For each plant:
  uint32 asset ID (see vegetation_manifest.json asset_table)
  float32 local east metres
  float32 local south metres
  float32 absolute elevation metres
  float32 height scale
  float32 width scale
  float32 yaw radians
  float32 RGB tint multiplier
  float32 stable density priority [0,1]

Cell bounds must include plant crowns and height when used for culling.
The viewer must load the referenced species textures/models separately.
""",
        encoding="utf-8",
    )

    print(f"\nFinished: {total_plants} plants in {output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
