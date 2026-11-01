#!/usr/bin/env python3
"""
Prepare compact terrain tiles, filling land gaps and excluding MTK water.

Usage:
    python prepare_terrain_parallel.py -i espoo2008_dem_tiled -o Historyviewer/espoo2008_terrain_quick --water MTK_2009_clipped/mtk-vakavesi_2009_clipped.gpkg --workers 4

    python prepare_terrain_parallel.py -i espoo_terrain -o viewer_terrain \
        --water MTK_2009_clipped/mtk-vakavesi_2009_clipped.gpkg \
        --workers 4 \
        --fill-distance 200

Arguments:
    -i, --input-dir       Folder containing tile_manifest.json and dem/.
    -o, --output-dir      New or empty output folder.
    --water              GeoPackage containing water polygons.
    --levels             Number of terrain LODs; default 5.
    --workers            Parallel tile-conversion workers; default 2.
    --fill-distance      Maximum land-gap interpolation distance in metres;
                         default 200. (for example where there are large buildings and big holes in the ground surface data)
     --allow-missing-land-cells 1000 (if there are less than specified amount of land cover cells without height data, ignore it. default is 1000


Outputs:
    terrain_index.txt
    metadata.json
    tiles/*.trn

Processing:
    1. Build a combined disk-backed DEM.
    2. Rasterize MTK water polygons into a separate mask.
    3. Fill missing LAND cells only, using a bounded search distance.
    4. Keep water cells as NoData.
    5. Convert terrain tiles in parallel.

Original DEMs are not modified. Temporary rasters are deleted afterward.

Requires:
    GDAL Python bindings, numpy, rasterio.
"""

import argparse
import atexit
import json
import math
import multiprocessing
import struct
import sys
import tempfile
import time
from concurrent.futures import (
    ProcessPoolExecutor,
    FIRST_COMPLETED,
    wait,
)
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from osgeo import gdal, ogr, osr


MAGIC = b"TRN1"
NODATA = -9999.0

WORKER_DATASET = None
WORKER_ENVIRONMENT = None


# ============================================================================
# INPUT VALIDATION
# ============================================================================

def traditional_crs(crs):
    """Clone a CRS with traditional GIS x/y axis order."""
    result = crs.Clone()
    result.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return result


def inspect_dems(source, manifest):
    """
    Validate source DEMs and calculate their combined bounds.

    Tiles without DEM files are still included through their core_bounds,
    allowing global interpolation to fill their area if neighboring DEMs
    provide enough support.
    """
    filenames = []
    reference_crs = None
    reference_transform = None
    resolution = None
    all_bounds = []

    for tile in manifest["tiles"]:
        if tile.get("core_bounds"):
            all_bounds.append(tuple(tile["core_bounds"]))

        if not tile.get("dem"):
            continue

        filename = source / tile["dem"]
        if not filename.is_file():
            raise ValueError(f"Missing DEM: {filename}")

        with rasterio.open(filename) as dataset:
            transform = dataset.transform

            if dataset.count != 1:
                raise ValueError(f"Expected a single-band DEM: {filename}")
            if dataset.crs is None:
                raise ValueError(f"DEM has no CRS: {filename}")
            if (
                transform.b != 0
                or transform.d != 0
                or transform.a <= 0
                or transform.e >= 0
            ):
                raise ValueError(f"DEM must be north-up: {filename}")
            if not math.isclose(
                transform.a, -transform.e, rel_tol=1e-8
            ):
                raise ValueError(f"DEM cells must be square: {filename}")

            if reference_crs is None:
                reference_crs = dataset.crs
                reference_transform = transform
                resolution = transform.a
            else:
                if dataset.crs != reference_crs:
                    raise ValueError(f"DEM CRS mismatch: {filename}")
                if not math.isclose(
                    transform.a, resolution, rel_tol=1e-8
                ):
                    raise ValueError(f"DEM resolution mismatch: {filename}")

                for offset in (
                    (transform.c - reference_transform.c) / resolution,
                    (transform.f - reference_transform.f) / resolution,
                ):
                    if not math.isclose(offset, round(offset), abs_tol=1e-5):
                        raise ValueError(
                            f"DEM grid misalignment: {filename}"
                        )

            bounds = dataset.bounds
            all_bounds.append((
                bounds.left,
                bounds.bottom,
                bounds.right,
                bounds.top,
            ))

        filenames.append(str(filename))

    if not filenames:
        raise ValueError("The manifest contains no available DEMs.")

    for bounds in all_bounds:
        if (
            len(bounds) != 4
            or not all(math.isfinite(value) for value in bounds)
            or bounds[0] >= bounds[2]
            or bounds[1] >= bounds[3]
        ):
            raise ValueError(f"Invalid DEM/core bounds: {bounds}")

        for coordinate, origin in (
            (bounds[0], reference_transform.c),
            (bounds[2], reference_transform.c),
            (bounds[1], reference_transform.f),
            (bounds[3], reference_transform.f),
        ):
            offset = (coordinate - origin) / resolution
            if not math.isclose(offset, round(offset), abs_tol=1e-5):
                raise ValueError(
                    f"Manifest core bounds are not DEM-grid aligned: {bounds}"
                )

    combined = (
        min(bounds[0] for bounds in all_bounds),
        min(bounds[1] for bounds in all_bounds),
        max(bounds[2] for bounds in all_bounds),
        max(bounds[3] for bounds in all_bounds),
    )

    return filenames, resolution, combined


def tile_bounds(source, tile):
    """Return a tile's core bounds."""
    if tile.get("core_bounds"):
        return tuple(tile["core_bounds"])

    if tile.get("dem"):
        with rasterio.open(source / tile["dem"]) as dataset:
            bounds = dataset.bounds
            return bounds.left, bounds.bottom, bounds.right, bounds.top

    return None


# ============================================================================
# COMBINED DEM
# ============================================================================

def build_mosaic(filenames, resolution, bounds, temporary_dir):
    """
    Create a writable combined DEM mosaic.

    Missing cells are not filled here. Filling is handled later with an
    explicit land/water mask.
    """
    vrt_path = temporary_dir / "combined.vrt"
    mosaic_path = temporary_dir / "working_dem.tif"

    vrt = gdal.BuildVRT(
        str(vrt_path),
        filenames,
        options=gdal.BuildVRTOptions(
            resolution="user",
            xRes=resolution,
            yRes=resolution,
            outputBounds=bounds,
            VRTNodata=NODATA,
        ),
    )

    if vrt is None:
        raise RuntimeError("Could not build the combined DEM VRT.")

    mosaic = gdal.Translate(
        str(mosaic_path),
        vrt,
        options=gdal.TranslateOptions(
            format="GTiff",
            outputType=gdal.GDT_Float32,
            noData=NODATA,
            creationOptions=[
                "TILED=YES",
                "BIGTIFF=YES",
                "COMPRESS=DEFLATE",
            ],
        ),
    )
    vrt = None

    if mosaic is None:
        raise RuntimeError("Could not create the working DEM.")

    band = mosaic.GetRasterBand(1)

    try:
        valid_count = 0

        for y in range(0, mosaic.RasterYSize, 512):
            height = min(512, mosaic.RasterYSize - y)

            for x in range(0, mosaic.RasterXSize, 512):
                width = min(512, mosaic.RasterXSize - x)
                data = band.ReadAsArray(x, y, width, height)

                nonfinite = ~np.isfinite(data)
                if np.any(nonfinite):
                    data[nonfinite] = NODATA
                    band.WriteArray(data, x, y)

                valid = (data != NODATA) & np.isfinite(data)
                valid_count += int(valid.sum())

        band.FlushCache()

        if valid_count == 0:
            raise ValueError("Combined DEM has no valid ground elevations.")

    finally:
        band = None
        mosaic.FlushCache()
        mosaic = None

    return mosaic_path


# ============================================================================
# WATER MASKING
# ============================================================================

def create_byte_raster_like(reference_path, output_path, fill_value=0):
    """Create a tiled Byte GeoTIFF with the same grid as reference_path."""
    reference = gdal.Open(str(reference_path), gdal.GA_ReadOnly)
    if reference is None:
        raise RuntimeError(f"Cannot open reference raster: {reference_path}")

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(
        str(output_path),
        reference.RasterXSize,
        reference.RasterYSize,
        1,
        gdal.GDT_Byte,
        options=[
            "TILED=YES",
            "BIGTIFF=YES",
            "COMPRESS=DEFLATE",
        ],
    )

    if dataset is None:
        raise RuntimeError(f"Cannot create raster: {output_path}")

    dataset.SetGeoTransform(reference.GetGeoTransform())
    dataset.SetProjection(reference.GetProjection())

    band = dataset.GetRasterBand(1)
    band.Fill(fill_value)
    band.FlushCache()

    band = None
    dataset = None
    reference = None


def burn_geometry_batch(raster_path, crs, geometries, value):
    """Burn a list of OGR geometries into one raster band."""
    raster = gdal.Open(str(raster_path), gdal.GA_Update)
    if raster is None:
        raise RuntimeError(f"Cannot open raster for update: {raster_path}")

    memory = ogr.GetDriverByName("Memory").CreateDataSource("")
    layer = memory.CreateLayer("mask", srs=crs, geom_type=ogr.wkbUnknown)

    for geometry in geometries:
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(geometry)
        layer.CreateFeature(feature)
        feature = None

    status = gdal.RasterizeLayer(
        raster,
        [1],
        layer,
        burn_values=[value],
        options=["ALL_TOUCHED=TRUE"],
    )

    layer = None
    memory = None
    raster.FlushCache()
    raster = None

    if status != 0:
        raise RuntimeError("Rasterization failed.")


def create_water_mask(mosaic_path, water_path, output_path):
    """
    Rasterize MTK water polygons to a Byte mask:
        1 = water
        0 = non-water
    """
    create_byte_raster_like(mosaic_path, output_path, fill_value=0)

    mosaic = gdal.Open(str(mosaic_path), gdal.GA_ReadOnly)
    if mosaic is None:
        raise RuntimeError("Cannot open mosaic for CRS lookup.")

    target = osr.SpatialReference()
    target.ImportFromWkt(mosaic.GetProjection())
    target = traditional_crs(target)
    mosaic = None

    source = ogr.Open(str(water_path), 0)
    if source is None:
        raise ValueError(f"Cannot open water GeoPackage: {water_path}")

    polygon_count = 0

    for layer_index in range(source.GetLayerCount()):
        layer = source.GetLayerByIndex(layer_index)

        if layer.GetGeomType() == ogr.wkbNone:
            continue

        source_crs = layer.GetSpatialRef()
        if source_crs is None:
            raise ValueError(f"Water layer {layer.GetName()!r} has no CRS.")

        source_crs = traditional_crs(source_crs)
        transformation = None

        if not source_crs.IsSame(target):
            print(f"Reprojecting water layer: {layer.GetName()}")
            transformation = osr.CoordinateTransformation(
                source_crs, target
            )

        batch = []

        for feature in layer:
            geometry = feature.GetGeometryRef()
            if geometry is None or geometry.IsEmpty():
                continue

            geometry = geometry.Clone()

            if not geometry.IsValid():
                geometry = geometry.MakeValid()

            if geometry is None or geometry.IsEmpty():
                continue

            if ogr.GT_Flatten(geometry.GetGeometryType()) not in (
                ogr.wkbPolygon,
                ogr.wkbMultiPolygon,
            ):
                continue

            if transformation is not None:
                if geometry.Transform(transformation) != 0:
                    raise RuntimeError("Water geometry transformation failed.")

            batch.append(geometry)
            polygon_count += 1

            if len(batch) >= 500:
                burn_geometry_batch(output_path, target, batch, value=1)
                batch = []

        if batch:
            burn_geometry_batch(output_path, target, batch, value=1)

    source = None

    if polygon_count == 0:
        raise ValueError("The water GeoPackage contains no usable polygons.")

    print(f"Rasterized {polygon_count} water polygons.")


def create_fill_mask(mosaic_path, water_mask_path, output_path):
    """
    Create GDAL FillNodata validity mask:
        1 = valid source OR water cell that must not be filled
        0 = land NoData cell that should be interpolated

    The water cells are set to 1 in this mask to prevent FillNodata from
    trying to interpolate them. The NODATA option prevents their -9999 target
    values from being used as interpolation sources.
    """
    create_byte_raster_like(mosaic_path, output_path, fill_value=0)

    mosaic = gdal.Open(str(mosaic_path), gdal.GA_ReadOnly)
    water = gdal.Open(str(water_mask_path), gdal.GA_ReadOnly)
    mask = gdal.Open(str(output_path), gdal.GA_Update)

    if mosaic is None or water is None or mask is None:
        raise RuntimeError("Could not open rasters for fill-mask creation.")

    dem_band = mosaic.GetRasterBand(1)
    water_band = water.GetRasterBand(1)
    mask_band = mask.GetRasterBand(1)

    for y in range(0, mosaic.RasterYSize, 512):
        height = min(512, mosaic.RasterYSize - y)

        for x in range(0, mosaic.RasterXSize, 512):
            width = min(512, mosaic.RasterXSize - x)

            data = dem_band.ReadAsArray(x, y, width, height)
            water_data = water_band.ReadAsArray(x, y, width, height)

            valid_source = np.isfinite(data) & (data != NODATA)
            is_water = water_data != 0

            # Fill only cells where this mask is zero.
            validity = (valid_source | is_water).astype(np.uint8)

            mask_band.WriteArray(validity, x, y)

    mask_band.FlushCache()

    mask_band = None
    water_band = None
    dem_band = None
    mask = None
    water = None
    mosaic = None


def fill_land_gaps(mosaic_path, fill_mask_path, fill_distance):
    """
    Interpolate land gaps only, leaving water excluded.

    maxSearchDist is bounded instead of using the diagonal of the mosaic.
    """
    mosaic = gdal.Open(str(mosaic_path), gdal.GA_Update)
    mask = gdal.Open(str(fill_mask_path), gdal.GA_ReadOnly)

    if mosaic is None or mask is None:
        raise RuntimeError("Could not open DEM/mask for land-gap filling.")

    band = mosaic.GetRasterBand(1)
    mask_band = mask.GetRasterBand(1)

    print("Interpolating missing LAND elevations only...")
    print(f"Maximum fill search distance: {fill_distance:g} m")

    status = gdal.FillNodata(
        targetBand=band,
        maskBand=mask_band,
        maxSearchDist=fill_distance,
        smoothingIterations=0,
        options=[f"NODATA={NODATA}"],
    )

    if status != 0:
        raise RuntimeError("Land-only terrain interpolation failed.")

    band.FlushCache()
    mosaic.FlushCache()

    mask_band = None
    band = None
    mask = None
    mosaic = None


def apply_water_to_dem(mosaic_path, water_mask_path):
    """Set water cells back to NODATA after interpolation."""
    mosaic = gdal.Open(str(mosaic_path), gdal.GA_Update)
    water = gdal.Open(str(water_mask_path), gdal.GA_ReadOnly)

    if mosaic is None or water is None:
        raise RuntimeError("Could not open rasters for final water masking.")

    dem_band = mosaic.GetRasterBand(1)
    water_band = water.GetRasterBand(1)

    for y in range(0, mosaic.RasterYSize, 512):
        height = min(512, mosaic.RasterYSize - y)

        for x in range(0, mosaic.RasterXSize, 512):
            width = min(512, mosaic.RasterXSize - x)

            data = dem_band.ReadAsArray(x, y, width, height)
            water_data = water_band.ReadAsArray(x, y, width, height)

            water_cells = water_data != 0

            if np.any(water_cells):
                data[water_cells] = NODATA
                dem_band.WriteArray(data, x, y)

    dem_band.FlushCache()
    mosaic.FlushCache()

    water_band = None
    dem_band = None
    water = None
    mosaic = None


def verify_land_complete(mosaic_path, water_mask_path, allowed_missing_cells):
    """
    Count missing non-water cells after interpolation.

    A small number of remaining cells can occur along shorelines, narrow
    unmapped water features, or isolated DEM defects. For the viewer, these
    become tiny terrain holes rather than a reason to stop the whole process.
    """
    missing = 0

    with rasterio.open(mosaic_path) as dem, rasterio.open(water_mask_path) as water:
        for _, window in dem.block_windows(1):
            data = dem.read(1, window=window, masked=True)
            water_data = water.read(1, window=window)

            invalid = np.ma.getmaskarray(data) | ~np.isfinite(data.data)
            land_invalid = invalid & (water_data == 0)

            missing += int(land_invalid.sum())

    if missing:
        print(
            f"WARNING: {missing} non-water DEM cell(s) remain missing "
            "after interpolation."
        )

    if missing > allowed_missing_cells:
        raise RuntimeError(
            f"{missing} non-water cells remain missing after land-only "
            f"interpolation. This exceeds the allowed limit of "
            f"{allowed_missing_cells}. Increase --fill-distance, increase "
            "--allow-missing-land-cells, or inspect DEM gaps."
        )

    return missing



# ============================================================================
# SAMPLING AND TRN WRITING
# ============================================================================

def sample_vertices(dataset, xs, ys):
    """
    Bilinearly sample the common mosaic at terrain vertex positions.

    Samples may cross individual tile boundaries. The outer project boundary
    is clamped.
    """
    transform = dataset.transform

    columns = np.clip(
        (xs - transform.c) / transform.a - 0.5,
        0,
        dataset.width - 1,
    )
    rows = np.clip(
        (ys - transform.f) / transform.e - 0.5,
        0,
        dataset.height - 1,
    )

    c0 = np.floor(columns).astype(int)
    r0 = np.floor(rows).astype(int)
    c1 = np.minimum(c0 + 1, dataset.width - 1)
    r1 = np.minimum(r0 + 1, dataset.height - 1)

    left = int(c0.min())
    top = int(r0.min())
    right = int(c1.max()) + 1
    bottom = int(r1.max()) + 1

    data = dataset.read(
        1,
        window=Window(left, top, right - left, bottom - top),
        masked=True,
    ).astype(np.float64).filled(np.nan)

    fx = columns - np.floor(columns)
    fy = rows - np.floor(rows)

    c0 -= left
    c1 -= left
    r0 -= top
    r1 -= top

    neighbors = [
        data[np.ix_(r0, c0)],
        data[np.ix_(r0, c1)],
        data[np.ix_(r1, c0)],
        data[np.ix_(r1, c1)],
    ]

    weights = [
        (1 - fy[:, None]) * (1 - fx[None, :]),
        (1 - fy[:, None]) * fx[None, :],
        fy[:, None] * (1 - fx[None, :]),
        fy[:, None] * fx[None, :],
    ]

    result = np.zeros(neighbors[0].shape, dtype=np.float64)
    valid = np.ones(result.shape, dtype=bool)

    for values, weight in zip(neighbors, weights):
        needed = weight > 1e-12
        valid &= ~needed | np.isfinite(values)
        result += np.where(np.isfinite(values), values, 0) * weight

    result[~valid] = np.nan
    return result.astype(np.float32)


def axis_vertices(cell_count, stride, resolution):
    """Include both tile boundaries, even when stride does not divide evenly."""
    indexes = list(range(0, cell_count, stride)) + [cell_count]
    return np.asarray(indexes, dtype=np.float64) * resolution


def aligned_index(value, description):
    """Validate grid alignment and convert to an integer index."""
    rounded = round(value)

    if not math.isclose(value, rounded, abs_tol=1e-5):
        raise ValueError(f"Grid misalignment: {description}")

    return rounded


def prepare_tile(dataset, bounds, destination, levels):
    """
    Write one TRN1 file.

    Fully water-covered tiles are omitted.
    """
    xmin, ymin, xmax, ymax = bounds
    resolution = dataset.transform.a

    width = aligned_index((xmax - xmin) / resolution, "tile width")
    height = aligned_index((ymax - ymin) / resolution, "tile height")

    if width < 1 or height < 1:
        return None

    col = aligned_index(
        (xmin - dataset.transform.c) / resolution,
        "tile column",
    )
    row = aligned_index(
        (dataset.transform.f - ymax) / resolution,
        "tile row",
    )

    if (
        col < 0 or row < 0
        or col + width > dataset.width
        or row + height > dataset.height
    ):
        raise ValueError("Tile core lies outside the combined DEM.")

    data = dataset.read(
        1,
        window=Window(col, row, width, height),
        masked=True,
    ).astype(np.float64).filled(np.nan)

    values = data[np.isfinite(data)]

    if not len(values):
        return None

    zmin = float(values.min())
    zmax = float(values.max())

    partial = destination.with_name(destination.name + ".part")

    try:
        with partial.open("wb") as output:
            output.write(MAGIC)
            output.write(struct.pack("<I", levels))
            output.write(struct.pack(
                "<4d2f", xmin, ymin, xmax, ymax, zmin, zmax
            ))

            for level in range(levels):
                stride = 2 ** level
                east = axis_vertices(width, stride, resolution)
                south = axis_vertices(height, stride, resolution)
                halo_distance = stride * resolution

                east_halo = np.concatenate((
                    [east[0] - halo_distance],
                    east,
                    [east[-1] + halo_distance],
                ))
                south_halo = np.concatenate((
                    [south[0] - halo_distance],
                    south,
                    [south[-1] + halo_distance],
                ))

                elevations = sample_vertices(
                    dataset,
                    xmin + east_halo,
                    ymax - south_halo,
                )

                output.write(struct.pack("<2I", len(east), len(south)))
                output.write(east.astype("<f4").tobytes())
                output.write(south.astype("<f4").tobytes())
                output.write(elevations.astype("<f4").tobytes())

        partial.replace(destination)

    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    return xmin, ymin, xmax, ymax, zmin, zmax


# ============================================================================
# PARALLEL WORKERS
# ============================================================================

def close_worker_resources():
    """Close worker-local raster handles."""
    global WORKER_DATASET, WORKER_ENVIRONMENT

    if WORKER_DATASET is not None:
        WORKER_DATASET.close()
        WORKER_DATASET = None

    if WORKER_ENVIRONMENT is not None:
        WORKER_ENVIRONMENT.__exit__(None, None, None)
        WORKER_ENVIRONMENT = None


def initialize_worker(mosaic_path):
    """Open the finished mosaic once per worker process."""
    global WORKER_DATASET, WORKER_ENVIRONMENT

    WORKER_ENVIRONMENT = rasterio.Env(
        GDAL_CACHEMAX=64 * 1024 * 1024,
        GDAL_NUM_THREADS="1",
    )
    WORKER_ENVIRONMENT.__enter__()

    WORKER_DATASET = rasterio.open(mosaic_path, "r")
    atexit.register(close_worker_resources)


def process_tile(task):
    """Convert one tile in a worker."""
    number, tile_name, bounds, destination, levels = task

    result = prepare_tile(
        WORKER_DATASET,
        bounds,
        Path(destination),
        levels,
    )

    return number, tile_name, result


def parallel_results(tasks, workers, mosaic_path):
    """Yield finished tile-conversion tasks."""
    iterator = iter(tasks)
    context = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=initialize_worker,
        initargs=(str(mosaic_path),),
    ) as executor:
        pending = {}

        def submit_next():
            try:
                task = next(iterator)
            except StopIteration:
                return False

            future = executor.submit(process_tile, task)
            pending[future] = task
            return True

        for _ in range(workers * 2):
            if not submit_next():
                break

        try:
            while pending:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)

                for future in finished:
                    task = pending.pop(future)

                    try:
                        result = future.result()
                    except Exception as error:
                        raise RuntimeError(
                            f"Failed to prepare {task[1]}: {error}"
                        ) from error

                    yield result
                    submit_next()

        finally:
            for future in pending:
                future.cancel()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare water-aware compact terrain tiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input-dir", required=True)
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--levels", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--fill-distance",
        type=float,
        default=200.0,
        help="Maximum land-gap interpolation distance in metres.",
    )
    parser.add_argument(
    "--allow-missing-land-cells",
    type=int,
    default=1000,
    help=(
        "Allow up to this many non-water DEM cells to remain NoData "
        "after interpolation. Default 1000."
    ),
    )

    parser.add_argument(
        "--water",
        default="MTK_2009_clipped/mtk-vakavesi_2009_clipped.gpkg",
        help="GeoPackage containing water polygons.",
    )
    args = parser.parse_args()

    if not 1 <= args.levels <= 10:
        parser.error("--levels must be between 1 and 10.")
    if args.workers < 1:
        parser.error("--workers must be at least 1.")
    if args.fill_distance <= 0 or not math.isfinite(args.fill_distance):
        parser.error("--fill-distance must be a positive finite number.")
    if args.allow_missing_land_cells < 0:
        parser.error("--allow-missing-land-cells cannot be negative.")


    gdal.UseExceptions()
    ogr.UseExceptions()
    osr.UseExceptions()

    # Main-process cache for mosaic preparation.
    gdal.SetCacheMax(256 * 1024 * 1024)

    source = Path(args.input_dir).resolve()
    water_path = Path(args.water).resolve()
    output = Path(args.output_dir).resolve()
    manifest_path = source / "tile_manifest.json"

    if not manifest_path.is_file():
        parser.error(f"Manifest does not exist: {manifest_path}")
    if not water_path.is_file():
        parser.error(f"Water GeoPackage does not exist: {water_path}")
    if output.exists() and (
        not output.is_dir() or any(output.iterdir())
    ):
        parser.error("Output folder must be empty or not yet exist.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    filenames, resolution, bounds = inspect_dems(source, manifest)

    tiles_dir = output / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    tasks = []

    for number, tile in enumerate(manifest["tiles"]):
        core = tile_bounds(source, tile)

        if core is None:
            print(f"Skipping tile without core bounds: {tile['tile']}")
            continue

        filename = f"tile_{number:06d}.trn"
        tasks.append((
            number,
            tile["tile"],
            core,
            str(tiles_dir / filename),
            args.levels,
        ))

    if not tasks:
        raise ValueError("No terrain tiles have usable core bounds.")

    workers = min(args.workers, len(tasks))
    entries_by_number = {}
    omitted_water_tiles = 0
    timings = {}

    with tempfile.TemporaryDirectory(
        prefix="_terrain_work_", dir=str(output)
    ) as temporary:
        temporary_dir = Path(temporary)

        start = time.perf_counter()
        mosaic_path = build_mosaic(
            filenames,
            resolution,
            bounds,
            temporary_dir,
        )
        timings["mosaic_seconds"] = time.perf_counter() - start

        water_mask_path = temporary_dir / "water_mask.tif"
        fill_mask_path = temporary_dir / "fill_mask.tif"

        start = time.perf_counter()
        print("Rasterizing water polygons...")
        create_water_mask(mosaic_path, water_path, water_mask_path)
        timings["water_mask_seconds"] = time.perf_counter() - start

        start = time.perf_counter()
        print("Creating land-only interpolation mask...")
        create_fill_mask(mosaic_path, water_mask_path, fill_mask_path)
        fill_land_gaps(mosaic_path, fill_mask_path, args.fill_distance)
        apply_water_to_dem(mosaic_path, water_mask_path)
        remaining_missing_land_cells = verify_land_complete(
            mosaic_path,
            water_mask_path,
            args.allow_missing_land_cells,
        )

        timings["land_gap_fill_seconds"] = time.perf_counter() - start

        start = time.perf_counter()
        print(
            f"Converting {len(tasks)} tiles using {workers} worker(s)...",
            flush=True,
        )

        completed = 0

        for number, tile_name, result in parallel_results(
            tasks,
            workers,
            mosaic_path,
        ):
            completed += 1

            if result is None:
                omitted_water_tiles += 1
                status = "omitted: no land cells"
            else:
                relative = f"tiles/tile_{number:06d}.trn"
                entries_by_number[number] = (relative, result)
                status = "written"

            print(
                f"[{completed}/{len(tasks)}] {tile_name}: {status}",
                flush=True,
            )

        timings["tile_conversion_seconds"] = time.perf_counter() - start

    entries = [
        entries_by_number[number]
        for number in sorted(entries_by_number)
    ]

    if not entries:
        raise ValueError("No land terrain tiles were generated.")

    origin_x = min(extent[0] for _, extent in entries)
    origin_y = max(extent[3] for _, extent in entries)
    origin_z = math.floor(min(extent[4] for _, extent in entries))

    with (output / "terrain_index.txt").open(
        "w", encoding="utf-8", newline="\n"
    ) as index:
        index.write("TERRAIN_INDEX_V1\n")
        index.write(f"{origin_x:.9f} {origin_y:.9f} {origin_z:.9f}\n")
        index.write(f"{len(entries)}\n")

        for filename, tile_extent in entries:
            index.write(
                filename + " "
                + " ".join(f"{value:.9f}" for value in tile_extent)
                + "\n"
            )

    metadata = {
        "crs": manifest.get("crs"),
        "vertical_reference": manifest.get("vertical_reference"),
        "model_origin_easting_northing_elevation": [
            origin_x, origin_y, origin_z
        ],
        "levels": args.levels,
        "workers": workers,
        "fill_distance_metres": args.fill_distance,
        "remaining_missing_land_cells": remaining_missing_land_cells,
        "allowed_missing_land_cells": args.allow_missing_land_cells,
        "tile_count": len(entries),
        "omitted_water_tiles": omitted_water_tiles,
        "source_manifest": str(manifest_path),
        "water_polygons": str(water_path),
        "gap_filling": "GDAL FillNodata on land cells only",
        "water_mask": "Polygon rasterization with ALL_TOUCHED",
        "timings_seconds": timings,
    }

    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"\nPrepared {len(entries)} land tiles in {output}")

    for stage, seconds in timings.items():
        print(f"  {stage}: {seconds:.1f} seconds")


if __name__ == "__main__":
    multiprocessing.freeze_support()

    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nInterrupted. Use a fresh output folder when rerunning.",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
