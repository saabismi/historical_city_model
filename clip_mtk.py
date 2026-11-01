#!/usr/bin/env python3
"""
Clip MTK GeoPackages to a raster or GeoPackage bounding rectangle.

Examples:
    python clip_mtk.py -i MTK_2009 --extent-gpkg area.gpkg
    python clip_mtk.py -i MTK_2009 --files tie rakennus \
        --extent-gpkg area.gpkg --overwrite
    python clip_mtk.py -i MTK-tie_2009.gpkg -r aerial.tif

Arguments:
    -i, --input       Folder containing MTK-<dataset>_<year>.gpkg files,
                      or a single GeoPackage.
    --files           Optional dataset names or exact filenames to select.
                      Default: all matching files in the input folder.
    -r, --raster      Raster defining the full bounding rectangle.
    --extent-gpkg     Alternatively, a reference GeoPackage.
    --extent-layer    Optional reference GeoPackage layer.
    --output-dir      Override the output folder.
    --overwrite       Replace existing output GeoPackages.

Default output:
    <working directory>/MTK_<year>_clipped/<input_stem>_clipped.gpkg

The reference defines a RECTANGLE, not a polygon-shaped clipping mask.
Raster NoData areas are included. GeoPackage reference bounds are calculated
from actual saved geometries.

All spatial layers are clipped and output in EPSG:3067.
Layer names and attributes are retained, but FIDs may be regenerated.
Nonspatial tables and application-specific metadata are not copied.

Requires GDAL Python bindings.
"""

import argparse
import math
import re
import sys
import tempfile
from pathlib import Path

from osgeo import gdal, ogr, osr


# Match the CSC archive convention, including uppercase/lowercase variants.
MTK_PATTERN = re.compile(
    r"^MTK-(?P<dataset>.+)_(?P<year>\d{4})\.gpkg$",
    re.IGNORECASE,
)


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
    Transform a bounding rectangle and return its EPSG:3067 envelope.

    Densification approximates curved boundaries introduced by reprojection.
    The resulting clipping region remains rectangular.
    """
    xmin, ymin, xmax, ymax = bounds

    if not all(math.isfinite(value) for value in bounds):
        raise ValueError("Reference extent contains invalid coordinates.")

    destination_crs = target_crs()

    if source_crs.IsSame(destination_crs):
        return bounds

    # A line also supports a degenerate individual layer extent, such as
    # one point, before the extents of multiple layers are combined.
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

    # OGR envelope order is xmin, xmax, ymin, ymax.
    left, right, bottom, top = boundary.GetEnvelope()
    result = (left, bottom, right, top)

    if not all(math.isfinite(value) for value in result):
        raise ValueError("Transformed reference extent is invalid.")

    return result


def raster_extent(filename):
    """Read the full raster bounding rectangle without inspecting pixels."""
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

    # Calculate all four outer pixel-grid corners, including rotation.
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
            "WARNING: Reference raster is rotated; "
            "using its axis-aligned bounding extent."
        )

    if not crs.IsSame(target_crs()):
        print(
            "WARNING: Reference raster is not EPSG:3067; "
            "transforming its extent."
        )

    raster = None
    return transform_rectangle_bounds(bounds, crs)


def geopackage_extent(filename, layer_name):
    """
    Calculate reference bounds from actual saved geometries.

    The geometries define bounds only; they are not used as clipping masks.
    Scanning avoids depending on potentially stale stored layer extents.
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
                f"WARNING: Reference layer {layer.GetName()!r} "
                "is not EPSG:3067; transforming its extent."
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

    return (
        min(bounds[0] for bounds in extents),
        min(bounds[1] for bounds in extents),
        max(bounds[2] for bounds in extents),
        max(bounds[3] for bounds in extents),
    )


def select_inputs(input_path, selectors):
    """
    Select matching MTK files from a folder, or accept one GeoPackage.

    A selector can be:
        tie
        MTK-tie_2009.gpkg

    Dataset names select all matching years if the folder contains more
    than one year. Exact filenames select only that file.
    """
    if input_path.is_file():
        if input_path.suffix.lower() != ".gpkg":
            raise ValueError("Single input must be a GeoPackage.")

        if selectors:
            raise ValueError(
                "--files is intended for a folder input. "
                "Omit it when supplying one GeoPackage."
            )

        return [input_path]

    if not input_path.is_dir():
        raise ValueError(f"Input does not exist: {input_path}")

    # Scan only the supplied folder, not its subfolders.
    candidates = sorted(
        (
            path.resolve()
            for path in input_path.iterdir()
            if path.is_file() and MTK_PATTERN.fullmatch(path.name)
        ),
        key=lambda path: path.name.casefold(),
    )

    if not candidates:
        raise ValueError(
            "No files matching MTK-<dataset>_<year>.gpkg found in:\n"
            f"{input_path}"
        )

    if not selectors:
        return candidates

    selected = set()

    for selector in selectors:
        normalized = selector.strip().casefold()

        matches = [
            path
            for path in candidates
            if path.name.casefold() == normalized
            or MTK_PATTERN.fullmatch(path.name).group(
                "dataset"
            ).casefold() == normalized
        ]

        if not matches:
            available = ", ".join(path.name for path in candidates)
            raise ValueError(
                f"No input file matches --files value {selector!r}.\n"
                f"Available files: {available}"
            )

        selected.update(matches)

    # Preserve deterministic processing order and remove duplicates.
    return [path for path in candidates if path in selected]


def input_layers(filename):
    """Validate spatial input layers and collect their names."""
    dataset = ogr.Open(str(filename), 0)

    if dataset is None:
        raise ValueError(f"Cannot open input GeoPackage: {filename}")

    names = []

    for index in range(dataset.GetLayerCount()):
        layer = dataset.GetLayerByIndex(index)
        name = layer.GetName()

        if layer.GetGeomType() == ogr.wkbNone:
            print(f"Skipping nonspatial table: {filename.name}:{name}")
            continue

        crs = layer.GetSpatialRef()

        if crs is None:
            raise ValueError(
                f"Input layer {filename.name}:{name!r} has no CRS."
            )

        if not normalized_crs(crs).IsSame(target_crs()):
            print(
                f"WARNING: {filename.name}:{name} is not EPSG:3067; "
                "it will be reprojected."
            )

        names.append(name)

    dataset = None

    if not names:
        raise ValueError(
            f"Input has no spatial vector layers: {filename}"
        )

    return names


def default_output_dir(filename):
    """Use a year-specific output folder when the filename follows MTK naming."""
    match = MTK_PATTERN.fullmatch(filename.name)

    if match:
        folder_name = f"MTK_{match.group('year')}_clipped"
    else:
        # Preserve generic single-GeoPackage support.
        folder_name = "MTK_clipped"

    return Path.cwd() / folder_name


def clip_file(input_path, output_path, layer_names, bounds):
    """
    Clip one input GeoPackage into one output GeoPackage.

    Build into a temporary file first. Existing output is replaced only
    after every spatial layer in this input has completed successfully.
    """
    xmin, ymin, xmax, ymax = bounds
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="_mtk_clip_", dir=str(output_path.parent)
    ) as temporary:
        temporary_output = Path(temporary) / "result.gpkg"

        for index, name in enumerate(layer_names):
            print(
                f"  [{index + 1}/{len(layer_names)}] Clipping {name}",
                flush=True,
            )

            options = [
                "-f", "GPKG",
                "-t_srs", "EPSG:3067",

                # Clip AFTER reprojection so the clipping region is exactly
                # the requested EPSG:3067 bounding rectangle.
                "-clipdst",
                str(xmin), str(ymin), str(xmax), str(ymax),

                "-nln", name,
                "-nlt", "PROMOTE_TO_MULTI",
                "-lco", "SPATIAL_INDEX=YES",
            ]

            if index > 0:
                options.append("-update")

            # Do not skip failed geometries silently.
            result = gdal.VectorTranslate(
                str(temporary_output),
                str(input_path),
                options=gdal.VectorTranslateOptions(
                    options=options,
                    layers=[name],
                ),
            )

            if result is None:
                raise RuntimeError(
                    f"Failed to clip {input_path.name}:{name}"
                )

            result = None

        # Close output in QGIS if Windows reports that replacement is locked.
        temporary_output.replace(output_path)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Clip MTK GeoPackages in a folder to a reference "
            "bounding rectangle."
        )
    )

    parser.add_argument(
        "-i", "--input", required=True,
        help="MTK input folder or a single GeoPackage.",
    )
    parser.add_argument(
        "--files", nargs="+",
        help="Dataset names or exact filenames to select from the folder.",
    )

    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument(
        "-r", "--raster",
        help="Raster defining the full bounding rectangle.",
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
        "--output-dir",
        help="Override the default year-specific output folder.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace existing output GeoPackages.",
    )

    args = parser.parse_args()

    gdal.UseExceptions()
    ogr.UseExceptions()
    osr.UseExceptions()

    if args.extent_layer and not args.extent_gpkg:
        parser.error("--extent-layer requires --extent-gpkg.")

    input_path = Path(args.input).resolve()
    reference_path = Path(
        args.raster or args.extent_gpkg
    ).resolve()

    if not reference_path.is_file():
        parser.error(
            f"Reference file does not exist: {reference_path}"
        )

    inputs = select_inputs(input_path, args.files)

    if args.raster:
        bounds = raster_extent(reference_path)
    else:
        bounds = geopackage_extent(
            reference_path, args.extent_layer
        )

    xmin, ymin, xmax, ymax = bounds

    if xmin >= xmax or ymin >= ymax:
        raise ValueError(
            "Reference rectangle has zero width or height."
        )

    print(
        "\nClipping rectangle in EPSG:3067:\n"
        f"  xmin: {xmin:.3f}\n"
        f"  ymin: {ymin:.3f}\n"
        f"  xmax: {xmax:.3f}\n"
        f"  ymax: {ymax:.3f}"
    )

    explicit_output = (
        Path(args.output_dir).resolve()
        if args.output_dir else None
    )

    # Preflight every selected file before clipping any of them.
    # This avoids doing lengthy work before discovering an existing output.
    jobs = []
    output_keys = set()
    protected_inputs = set(inputs) | {reference_path}

    for filename in inputs:
        output_dir = (
            explicit_output
            if explicit_output
            else default_output_dir(filename)
        )
        output_path = (
            output_dir / f"{filename.stem}_clipped.gpkg"
        ).resolve()

        if output_path in protected_inputs:
            raise ValueError(
                f"Output would replace an input/reference: {output_path}"
            )

        # Case-insensitive comparison protects Windows output paths.
        key = str(output_path).casefold()
        if key in output_keys:
            raise ValueError(
                f"Two selected files have the same output path: {output_path}"
            )
        output_keys.add(key)

        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_path}\n"
                "Use --overwrite to replace existing outputs."
            )

        jobs.append((
            filename,
            output_path,
            input_layers(filename),
        ))

    print(f"\nSelected {len(jobs)} GeoPackage(s):")
    for filename, output_path, _ in jobs:
        print(f"  {filename.name} -> {output_path}")

    # Process sequentially. Each input produces its own clipped GeoPackage.
    for index, (filename, output_path, layers) in enumerate(jobs):
        print(
            f"\n[{index + 1}/{len(jobs)}] Processing {filename.name}",
            flush=True,
        )
        clip_file(filename, output_path, layers, bounds)
        print(f"  Saved: {output_path}")

    print(f"\nFinished: {len(jobs)} GeoPackage(s) clipped.")
    print("Output CRS: EPSG:3067")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nInterrupted. Previously completed outputs remain available.",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
