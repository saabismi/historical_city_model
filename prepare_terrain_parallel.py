#!/usr/bin/env python3
"""
Prepare compact terrain tiles, filling land gaps and excluding MTK water.

Usage:
    python prepare_terrain.py -i espoo_terrain -o viewer_terrain
    python prepare_terrain.py -i espoo_terrain -o viewer_terrain \
        --water MTK_2009_clipped/mtk-vakavesi_2009_clipped.gpkg

Arguments:
    -i, --input-dir   Folder containing tile_manifest.json and dem/.
    -o, --output-dir  Output folder; must be empty or not yet exist.
    --water           Water-polygon GeoPackage; default shown above.
    --levels          Number of terrain LODs; default 5.

Outputs:
    terrain_index.txt
    metadata.json
    tiles/*.trn

Missing elevations are interpolated across a combined disk-backed DEM.
Water polygons are then imposed as NoData. Original DEMs are not modified.
Temporary rasters require additional disk space and are deleted afterward.

Requires: GDAL Python bindings, numpy, rasterio.
"""

import argparse
import json
import math
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from osgeo import gdal, ogr, osr


# Binary signature expected by the existing C++ viewer.
MAGIC = b"TRN1"

# Working rasters use this value for missing elevations.
NODATA = -9999.0


def traditional_crs(crs):
    """Clone a CRS with conventional GIS x/y axis order."""
    result = crs.Clone()
    result.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return result


def inspect_dems(source, manifest):
    """
    Validate source DEMs and determine the common grid and project bounds.

    Tiles with no DEM can still be generated if the manifest provides their
    core_bounds and the surrounding mosaic contains interpolation sources.
    """
    filenames = []
    reference_crs = None
    reference_transform = None
    resolution = None
    dem_bounds = []

    for tile in manifest["tiles"]:
        if not tile.get("dem"):
            continue

        filename = source / tile["dem"]
        if not filename.is_file():
            raise ValueError(f"Missing DEM: {filename}")

        with rasterio.open(filename) as dataset:
            t = dataset.transform

            if dataset.count != 1:
                raise ValueError(f"Expected a single-band DEM: {filename}")
            if dataset.crs is None:
                raise ValueError(f"DEM has no CRS: {filename}")
            if t.b != 0 or t.d != 0 or t.a <= 0 or t.e >= 0:
                raise ValueError(f"DEM must be north-up: {filename}")
            if not math.isclose(t.a, -t.e, rel_tol=1e-8):
                raise ValueError(f"DEM cells must be square: {filename}")

            if reference_crs is None:
                reference_crs = dataset.crs
                reference_transform = t
                resolution = t.a
            else:
                if dataset.crs != reference_crs:
                    raise ValueError(f"DEM CRS mismatch: {filename}")
                if not math.isclose(t.a, resolution, rel_tol=1e-8):
                    raise ValueError(f"DEM resolution mismatch: {filename}")

                # All tiles must sit on the same pixel lattice.
                for offset in (
                    (t.c - reference_transform.c) / resolution,
                    (t.f - reference_transform.f) / resolution,
                ):
                    if not math.isclose(offset, round(offset), abs_tol=1e-5):
                        raise ValueError(f"DEM grid misalignment: {filename}")

            b = dataset.bounds
            dem_bounds.append([b.left, b.bottom, b.right, b.top])

        filenames.append(str(filename))

    if not filenames:
        raise ValueError("The manifest contains no available DEMs.")

    # Include cores of tiles skipped during original DEM generation.
    all_bounds = dem_bounds + [
        tile["core_bounds"]
        for tile in manifest["tiles"]
        if tile.get("core_bounds")
    ]

    xmin = min(b[0] for b in all_bounds)
    ymin = min(b[1] for b in all_bounds)
    xmax = max(b[2] for b in all_bounds)
    ymax = max(b[3] for b in all_bounds)

    # Validate that the requested outer extent lies on the same grid.
    for coordinate, origin in (
        (xmin, reference_transform.c),
        (xmax, reference_transform.c),
        (ymin, reference_transform.f),
        (ymax, reference_transform.f),
    ):
        offset = (coordinate - origin) / resolution
        if not math.isclose(offset, round(offset), abs_tol=1e-5):
            raise ValueError("Manifest core bounds are not DEM-grid aligned.")

    return filenames, resolution, (xmin, ymin, xmax, ymax)


def build_filled_mosaic(filenames, resolution, bounds, temporary_dir):
    """
    Create a writable combined DEM and fill missing elevations globally.

    This removes the old hole-size and tile-boundary restrictions. Large
    unobserved regions can take a long time to interpolate.
    """
    vrt_path = temporary_dir / "combined.vrt"
    mosaic_path = temporary_dir / "filled_watermasked.tif"

    # BuildVRT respects each source's declared NoData.
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

    # Normalize nonfinite elevations to the working NoData value.
    valid_count = 0
    for y in range(0, mosaic.RasterYSize, 256):
        height = min(256, mosaic.RasterYSize - y)
        for x in range(0, mosaic.RasterXSize, 256):
            width = min(256, mosaic.RasterXSize - x)
            data = band.ReadAsArray(x, y, width, height)
            valid = np.isfinite(data) & (data != NODATA)
            valid_count += int(valid.sum())

            if np.any(~np.isfinite(data)):
                data[~np.isfinite(data)] = NODATA
                band.WriteArray(data, x, y)

    band.FlushCache()

    if not valid_count:
        raise ValueError("Combined DEM has no valid ground elevations.")

    # A search radius spanning the mosaic allows large gaps to be filled.
    search_distance = math.hypot(
        mosaic.RasterXSize, mosaic.RasterYSize
    ) + 1

    print("Interpolating missing elevations across the combined DEM...")
    print("This can be slow for large gaps, including sea before masking.")

    status = gdal.FillNodata(
        targetBand=band,
        maskBand=band.GetMaskBand(),
        maxSearchDist=search_distance,
        smoothingIterations=0,
    )

    if status != 0:
        raise RuntimeError("Terrain interpolation failed.")

    band.FlushCache()
    band = None
    mosaic.FlushCache()
    mosaic = None

    return mosaic_path


def apply_water_polygons(mosaic_path, water_path):
    """
    Burn all polygon layers from the water GeoPackage into the working DEM.

    Geometries are transformed to the DEM CRS. Nonpolygon layers are ignored.
    Rasterization uses ALL_TOUCHED, conservatively excluding shoreline cells.
    """
    mosaic = gdal.Open(str(mosaic_path), gdal.GA_Update)
    if mosaic is None:
        raise RuntimeError("Cannot reopen the working DEM.")

    target = osr.SpatialReference()
    target.ImportFromWkt(mosaic.GetProjection())
    target = traditional_crs(target)

    source = ogr.Open(str(water_path), 0)
    if source is None:
        raise ValueError(f"Cannot open water GeoPackage: {water_path}")

    polygon_count = 0

    for layer_index in range(source.GetLayerCount()):
        layer = source.GetLayerByIndex(layer_index)
        source_crs = layer.GetSpatialRef()

        # Ignore nonspatial tables.
        if layer.GetGeomType() == ogr.wkbNone:
            continue
        if source_crs is None:
            raise ValueError(f"Water layer {layer.GetName()!r} has no CRS.")

        source_crs = traditional_crs(source_crs)
        transformation = None

        if not source_crs.IsSame(target):
            print(f"Reprojecting water layer: {layer.GetName()}")
            transformation = osr.CoordinateTransformation(
                source_crs, target
            )

        # Use bounded-size batches rather than storing every polygon in RAM.
        batch = []
        for feature in layer:
            geometry = feature.GetGeometryRef()
            if geometry is None or geometry.IsEmpty():
                continue

            geometry = geometry.Clone()
            if not geometry.IsValid():
                geometry = geometry.MakeValid()

            if ogr.GT_Flatten(geometry.GetGeometryType()) not in (
                ogr.wkbPolygon, ogr.wkbMultiPolygon
            ):
                continue

            if transformation is not None:
                geometry.Transform(transformation)

            batch.append(geometry)
            polygon_count += 1

            if len(batch) >= 500:
                burn_batch(mosaic, target, batch)
                batch = []

        if batch:
            burn_batch(mosaic, target, batch)

    source = None
    mosaic.FlushCache()
    mosaic = None

    if polygon_count == 0:
        raise ValueError("The water GeoPackage contains no usable polygons.")

    print(f"Applied {polygon_count} water polygons.")


def burn_batch(mosaic, crs, geometries):
    """Rasterize a small in-memory polygon batch into the DEM as NoData."""
    memory = ogr.GetDriverByName("Memory").CreateDataSource("")
    layer = memory.CreateLayer("water", srs=crs, geom_type=ogr.wkbUnknown)

    for geometry in geometries:
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(geometry)
        layer.CreateFeature(feature)
        feature = None

    status = gdal.RasterizeLayer(
        mosaic,
        [1],
        layer,
        burn_values=[NODATA],
        options=["ALL_TOUCHED=TRUE"],
    )
    memory = None

    if status != 0:
        raise RuntimeError("Water polygon rasterization failed.")


def verify_land_complete(mosaic_path):
    """
    Check remaining missing cells.

    After masking, NoData is expected for water. Interpolation completeness
    is checked before masking by calling this function at that stage.
    """
    with rasterio.open(mosaic_path) as dataset:
        missing = 0

        for _, window in dataset.block_windows(1):
            data = dataset.read(1, window=window, masked=True)
            missing += int((
                np.ma.getmaskarray(data) | ~np.isfinite(data.data)
            ).sum())

    if missing:
        raise RuntimeError(
            f"{missing} cells remain missing after interpolation. "
            "No terrain files were generated."
        )


def sample_vertices(dataset, xs, ys):
    """
    Sample the common mosaic at mesh vertex positions.

    All tiles use the same source grid, including samples beyond individual
    tile boundaries. Only the outer project boundary is clamped.
    """
    t = dataset.transform
    columns = np.clip(
        (xs - t.c) / t.a - 0.5, 0, dataset.width - 1
    )
    rows = np.clip(
        (ys - t.f) / t.e - 0.5, 0, dataset.height - 1
    )

    c0 = np.floor(columns).astype(int)
    r0 = np.floor(rows).astype(int)
    c1 = np.minimum(c0 + 1, dataset.width - 1)
    r1 = np.minimum(r0 + 1, dataset.height - 1)

    left, top = int(c0.min()), int(r0.min())
    right, bottom = int(c1.max()) + 1, int(r1.max()) + 1

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
    """Include both boundaries even when the stride does not divide the core."""
    indexes = list(range(0, cell_count, stride)) + [cell_count]
    return np.asarray(indexes, dtype=np.float64) * resolution


def tile_bounds(source, tile):
    """Obtain core bounds, including tiles with no original DEM file."""
    if tile.get("core_bounds"):
        return tuple(tile["core_bounds"])

    if tile.get("dem"):
        with rasterio.open(source / tile["dem"]) as dataset:
            b = dataset.bounds
            return b.left, b.bottom, b.right, b.top

    return None


def prepare_tile(dataset, bounds, destination, levels):
    """
    Write one TRN1 file compatible with the current C++ viewer.

    Each LOD stores:
        uint32 column count, row count
        float32 east offsets
        float32 south offsets
        float32 elevations including a one-sample halo
    """
    xmin, ymin, xmax, ymax = bounds
    resolution = dataset.transform.a

    width = round((xmax - xmin) / resolution)
    height = round((ymax - ymin) / resolution)

    if width < 1 or height < 1:
        return None

    # Determine land elevation bounds from this tile's mosaic cells.
    col = round((xmin - dataset.transform.c) / resolution)
    row = round((dataset.transform.f - ymax) / resolution)

    data = dataset.read(
        1, window=Window(col, row, width, height), masked=True
    ).astype(np.float64).filled(np.nan)
    values = data[np.isfinite(data)]

    # Fully water-covered tiles do not need a terrain file.
    if not len(values):
        return None

    zmin, zmax = float(values.min()), float(values.max())

    with destination.open("wb") as output:
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
                dataset, xmin + east_halo, ymax - south_halo
            )

            output.write(struct.pack("<2I", len(east), len(south)))
            output.write(east.astype("<f4").tobytes())
            output.write(south.astype("<f4").tobytes())
            output.write(elevations.astype("<f4").tobytes())

    return xmin, ymin, xmax, ymax, zmin, zmax


def main():
    parser = argparse.ArgumentParser(
        description="Fill terrain gaps and prepare water-masked viewer tiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input-dir", required=True)
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--levels", type=int, default=5)
    parser.add_argument(
        "--water",
        default="MTK_2009_clipped/mtk-vakavesi_2009_clipped.gpkg",
        help="GeoPackage containing water polygons.",
    )
    args = parser.parse_args()

    if not 1 <= args.levels <= 10:
        parser.error("--levels must be between 1 and 10.")

    gdal.UseExceptions()
    ogr.UseExceptions()
    osr.UseExceptions()

    # Limit GDAL's block cache; this does not cap total process memory.
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
    entries = []

    # Store large temporary rasters on disk, not in memory.
    with tempfile.TemporaryDirectory(
        prefix="_terrain_work_", dir=str(output)
    ) as temporary:
        temporary_dir = Path(temporary)

        mosaic_path = build_filled_mosaic(
            filenames, resolution, bounds, temporary_dir
        )

        # Confirm that all gaps have been filled before introducing water.
        verify_land_complete(mosaic_path)

        print("Removing terrain covered by MTK water polygons...")
        apply_water_polygons(mosaic_path, water_path)

        with rasterio.open(mosaic_path) as dataset:
            for number, tile in enumerate(manifest["tiles"]):
                core = tile_bounds(source, tile)
                if core is None:
                    print(f"Skipping tile without core bounds: {tile['tile']}")
                    continue

                print(
                    f"[{number + 1}/{len(manifest['tiles'])}] "
                    f"Preparing {tile['tile']}"
                )

                filename = f"tile_{number:06d}.trn"
                result = prepare_tile(
                    dataset, core, tiles_dir / filename, args.levels
                )

                if result is not None:
                    entries.append((f"tiles/{filename}", result))

    if not entries:
        raise ValueError("No land terrain tiles were generated.")

    origin_x = min(b[0] for _, b in entries)
    origin_y = max(b[3] for _, b in entries)
    origin_z = math.floor(min(b[4] for _, b in entries))

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
        "tile_count": len(entries),
        "source_manifest": str(manifest_path),
        "water_polygons": str(water_path),
        "gap_filling": "Global GDAL FillNodata interpolation",
        "water_mask": "Polygon rasterization with ALL_TOUCHED",
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"\nPrepared {len(entries)} land tiles in {output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
