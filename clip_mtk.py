#!/usr/bin/env python3
"""
Clip all input GeoPackage vector layers to a reference bounding rectangle.

Usage:
    python clip_mtk.py -i mtk-tie_2009.gpkg -r aerial.tif
    python clip_mtk.py -i mtk-tie_2009.gpkg \
        --extent-gpkg espoo2008_area.gpkg --overwrite

Arguments:
    -i, --input       Input GeoPackage to clip.
    -r, --raster      Raster defining the full rectangular extent.
    --extent-gpkg     Alternatively, a GeoPackage defining the extent.
    --extent-layer    Optional reference GeoPackage layer.
                      Default: all nonempty spatial layers.
    --overwrite       Replace an existing output GeoPackage.

Output:
    <working directory>/MTK_2009_clipped/<input_stem>_clipped.gpkg

The reference defines a RECTANGLE, not a polygon-shaped clipping mask.
Raster NoData areas are included.

All output layers use EPSG:3067. Other CRSs produce warnings and are
transformed. Missing CRS metadata causes an error.

GeoPackage bounds are calculated from actual saved geometries.
Save QGIS layer edits before running this script.

Requires GDAL Python bindings.
"""

import argparse
import math
import sys
import tempfile
from pathlib import Path

from osgeo import gdal, ogr, osr


def target_crs():
    """Create EPSG:3067 with conventional easting/northing axis order."""
    crs = osr.SpatialReference()
    crs.ImportFromEPSG(3067)
    crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return crs


def normalized_crs(crs):
    """Clone a CRS before normalizing its axis order."""
    result = crs.Clone()
    result.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return result


def transform_rectangle_bounds(bounds, source_crs):
    """
    Transform a source bounding rectangle and return its EPSG:3067 bounds.

    Boundary densification approximates curvature introduced by reprojection.
    The final output remains an axis-aligned rectangle in EPSG:3067.
    """
    xmin, ymin, xmax, ymax = bounds

    if not all(math.isfinite(value) for value in bounds):
        raise ValueError("Reference extent contains invalid coordinates.")

    destination_crs = target_crs()

    if source_crs.IsSame(destination_crs):
        return bounds

    # Use a line around the rectangle, which also handles a degenerate
    # per-layer extent such as a layer containing a single point.
    corners = [
        (xmin, ymin),
        (xmax, ymin),
        (xmax, ymax),
        (xmin, ymax),
        (xmin, ymin),
    ]

    boundary = ogr.Geometry(ogr.wkbLineString)

    for edge in range(4):
        start = corners[edge]
        end = corners[edge + 1]

        for step in range(100):
            fraction = step / 100.0
            boundary.AddPoint_2D(
                start[0] + fraction * (end[0] - start[0]),
                start[1] + fraction * (end[1] - start[1]),
            )

    boundary.AddPoint_2D(*corners[-1])

    transformation = osr.CoordinateTransformation(
        source_crs, destination_crs
    )

    if boundary.Transform(transformation) != 0:
        raise RuntimeError("Reference extent transformation failed.")

    # OGR envelope order differs from the order used in this script.
    left, right, bottom, top = boundary.GetEnvelope()
    result = (left, bottom, right, top)

    if not all(math.isfinite(value) for value in result):
        raise ValueError("Transformed reference extent is invalid.")

    return result


def raster_extent(filename):
    """Read the raster's full bounding rectangle, ignoring pixel values."""
    raster = gdal.Open(str(filename), gdal.GA_ReadOnly)

    if raster is None:
        raise ValueError(f"Cannot open raster: {filename}")

    projection = raster.GetProjection()
    if not projection:
        raise ValueError("Reference raster has no CRS.")

    crs = osr.SpatialReference()
    crs.ImportFromWkt(projection)
    crs = normalized_crs(crs)

    transform = raster.GetGeoTransform()
    corners = []

    # Include all four outer pixel-grid corners.
    for column, row in (
        (0, 0),
        (raster.RasterXSize, 0),
        (raster.RasterXSize, raster.RasterYSize),
        (0, raster.RasterYSize),
    ):
        corners.append((
            transform[0] + column * transform[1] + row * transform[2],
            transform[3] + column * transform[4] + row * transform[5],
        ))

    bounds = (
        min(point[0] for point in corners),
        min(point[1] for point in corners),
        max(point[0] for point in corners),
        max(point[1] for point in corners),
    )

    if transform[2] != 0 or transform[4] != 0:
        print(
            "WARNING: Reference raster is rotated. "
            "Using its axis-aligned bounding extent."
        )

    if not crs.IsSame(target_crs()):
        print(
            "WARNING: Reference raster is not EPSG:3067. "
            "Transforming its extent."
        )

    raster = None
    return transform_rectangle_bounds(bounds, crs)


def geopackage_extent(filename, layer_name):
    """
    Calculate reference bounds from saved feature geometries.

    Do not use layer.GetExtent(): stored GeoPackage bounds can remain larger
    than the actual geometry after editing or removing features.
    """
    dataset = ogr.Open(str(filename), 0)

    if dataset is None:
        raise ValueError(f"Cannot open reference GeoPackage: {filename}")

    if layer_name:
        layer = dataset.GetLayerByName(layer_name)
        if layer is None:
            raise ValueError(
                f"Reference layer {layer_name!r} not found."
            )
        layers = [layer]
    else:
        layers = [
            dataset.GetLayerByIndex(index)
            for index in range(dataset.GetLayerCount())
        ]

    extents = []

    for layer in layers:
        if layer.GetGeomType() == ogr.wkbNone:
            continue

        layer.SetSpatialFilter(None)
        layer.SetAttributeFilter(None)
        layer.ResetReading()

        xmin = ymin = float("inf")
        xmax = ymax = float("-inf")
        feature_count = 0

        # Scan actual geometries; their shapes are never used as clip masks.
        for feature in layer:
            geometry = feature.GetGeometryRef()

            if geometry is None or geometry.IsEmpty():
                continue

            left, right, bottom, top = geometry.GetEnvelope()

            if not all(math.isfinite(value) for value in (
                left, right, bottom, top
            )):
                raise ValueError(
                    f"Invalid geometry bounds in {layer.GetName()!r}."
                )

            xmin = min(xmin, left)
            ymin = min(ymin, bottom)
            xmax = max(xmax, right)
            ymax = max(ymax, top)
            feature_count += 1

        if feature_count == 0:
            continue

        crs = layer.GetSpatialRef()
        if crs is None:
            raise ValueError(
                f"Reference layer {layer.GetName()!r} has no CRS."
            )

        crs = normalized_crs(crs)

        if not crs.IsSame(target_crs()):
            print(
                f"WARNING: Reference layer {layer.GetName()!r} is not "
                "EPSG:3067. Transforming its extent."
            )

        extents.append(transform_rectangle_bounds(
            (xmin, ymin, xmax, ymax), crs
        ))

        print(
            f"Reference layer: {layer.GetName()} "
            f"({feature_count} saved geometries)"
        )

    layers = None
    dataset = None

    if not extents:
        raise ValueError(
            "Reference GeoPackage contains no usable spatial geometries."
        )

    # Combine layer extents into one rectangular clipping area.
    return (
        min(bounds[0] for bounds in extents),
        min(bounds[1] for bounds in extents),
        max(bounds[2] for bounds in extents),
        max(bounds[3] for bounds in extents),
    )


def input_layers(filename):
    """Validate all spatial input layers and collect their names."""
    dataset = ogr.Open(str(filename), 0)

    if dataset is None:
        raise ValueError(f"Cannot open input GeoPackage: {filename}")

    names = []

    for index in range(dataset.GetLayerCount()):
        layer = dataset.GetLayerByIndex(index)
        name = layer.GetName()

        if layer.GetGeomType() == ogr.wkbNone:
            print(f"Skipping nonspatial table: {name}")
            continue

        crs = layer.GetSpatialRef()

        if crs is None:
            raise ValueError(f"Input layer {name!r} has no CRS.")

        if not normalized_crs(crs).IsSame(target_crs()):
            print(
                f"WARNING: Input layer {name!r} is not EPSG:3067. "
                "It will be reprojected."
            )

        names.append(name)

    dataset = None

    if not names:
        raise ValueError("Input GeoPackage has no spatial vector layers.")

    return names


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Clip all GeoPackage vector layers to a raster or "
            "GeoPackage bounding rectangle."
        )
    )

    parser.add_argument(
        "-i", "--input", required=True,
        help="Input GeoPackage to clip.",
    )

    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument(
        "-r", "--raster",
        help="Raster defining the full rectangular extent.",
    )
    reference.add_argument(
        "--extent-gpkg",
        help="GeoPackage defining the rectangular bounding extent.",
    )

    parser.add_argument(
        "--extent-layer",
        help="Reference GeoPackage layer; default all spatial layers.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace an existing output GeoPackage.",
    )

    args = parser.parse_args()

    gdal.UseExceptions()
    ogr.UseExceptions()
    osr.UseExceptions()

    if args.extent_layer and not args.extent_gpkg:
        parser.error("--extent-layer requires --extent-gpkg.")

    input_path = Path(args.input).resolve()
    reference_path = Path(args.raster or args.extent_gpkg).resolve()

    if not input_path.is_file():
        parser.error(f"Input GeoPackage does not exist: {input_path}")
    if not reference_path.is_file():
        parser.error(f"Reference file does not exist: {reference_path}")

    output_dir = Path.cwd() / "MTK_2009_clipped"
    output_path = output_dir / f"{input_path.stem}_clipped.gpkg"

    if output_path.resolve() in (input_path, reference_path):
        parser.error("Output cannot replace an input or reference file.")

    if output_path.exists() and not args.overwrite:
        parser.error(
            f"Output already exists: {output_path}\n"
            "Use --overwrite to replace it."
        )

    if args.raster:
        bounds = raster_extent(reference_path)
    else:
        bounds = geopackage_extent(
            reference_path, args.extent_layer
        )

    xmin, ymin, xmax, ymax = bounds

    if xmin >= xmax or ymin >= ymax:
        raise ValueError("Reference rectangle has zero width or height.")

    print(
        "\nClipping rectangle in EPSG:3067:\n"
        f"  xmin: {xmin:.3f}\n"
        f"  ymin: {ymin:.3f}\n"
        f"  xmax: {xmax:.3f}\n"
        f"  ymax: {ymax:.3f}"
    )

    layers = input_layers(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preserve any existing output until all layers have been processed.
    with tempfile.TemporaryDirectory(
        prefix="_mtk_clip_", dir=str(output_dir)
    ) as temporary:
        temporary_output = Path(temporary) / "result.gpkg"

        for index, name in enumerate(layers):
            print(f"[{index + 1}/{len(layers)}] Clipping {name}")

            options = [
                "-f", "GPKG",
                "-t_srs", "EPSG:3067",

                # Destination clipping occurs AFTER reprojection.
                # Therefore this is an exact axis-aligned EPSG:3067
                # rectangle, not the reference polygon geometry.
                "-clipdst",
                str(xmin), str(ymin), str(xmax), str(ymax),

                "-nln", name,
                "-nlt", "PROMOTE_TO_MULTI",
                "-lco", "SPATIAL_INDEX=YES",
            ]

            if index > 0:
                options.append("-update")

            result = gdal.VectorTranslate(
                str(temporary_output),
                str(input_path),
                options=gdal.VectorTranslateOptions(
                    options=options,
                    layers=[name],
                ),
            )

            if result is None:
                raise RuntimeError(f"Failed to clip layer: {name}")

            result = None

        # Atomic replacement is attempted after closing all output handles.
        # Close the old output in QGIS if Windows reports it is locked.
        temporary_output.replace(output_path)

    print(f"\nFinished: {output_path}")
    print(f"Spatial layers processed: {len(layers)}")
    print("Output CRS: EPSG:3067")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
