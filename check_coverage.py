#!/usr/bin/env python3
"""
Check whether downloaded point clouds cover a proposed model extent.

Examples:
    python check_pointcloud_coverage.py ^
        -i new_extent.gpkg ^
        --downloads espoo1965_clouds

    python check_pointcloud_coverage.py ^
        -i new_extent.tif ^
        --downloads espoo1965_clouds ^
        --buffer 30

Arguments:
    -i, --input       GeoPackage or TIFF defining a RECTANGULAR extent.
    --downloads       Parent folder containing download_manifest.csv,
                      _index/2008_latest.shp, and pointclouds/.
    --buffer          Extra coverage required on EACH side, in metres.
                      Default: 20.
    --layer           Optional GeoPackage layer. By default, combine the
                      extents of all nonempty spatial layers.
    --index           Optional override for 2008_latest.shp.

Exit codes:
    0  Buffered area has indexed coverage and all required oldest files.
    1  Missing coverage or required files.
    2  Input/configuration error.

This checks index polygons and local file presence. It does not inspect
actual LAS/LAZ points or their classifications.
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote, urlsplit

from osgeo import gdal, ogr, osr
from shapely import from_wkb, make_valid, union_all
from shapely.geometry import GeometryCollection, box


ARCHIVE_PREFIX = "mml/laserkeilaus/2008_latest/"


def normalized_srs(srs):
    """Clone an OGR CRS using conventional X/Y coordinate order."""
    result = srs.Clone()
    result.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return result


def target_srs():
    """Return EPSG:3067, whose horizontal coordinates are metres."""
    result = osr.SpatialReference()
    result.ImportFromEPSG(3067)
    result.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return result


def transform_bounds(bounds, source_crs, destination_crs):
    """
    Transform a rectangle's boundary and return its enclosing rectangle.

    Densifying before transformation handles CRS changes more reliably
    than transforming only four corners.
    """
    xmin, ymin, xmax, ymax = bounds

    if not all(math.isfinite(v) for v in bounds):
        raise ValueError("Extent contains non-finite coordinates.")
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("Extent is empty or invalid.")

    if source_crs.IsSame(destination_crs):
        return bounds

    corners = [
        (xmin, ymin),
        (xmax, ymin),
        (xmax, ymax),
        (xmin, ymax),
    ]

    ring = ogr.Geometry(ogr.wkbLinearRing)

    for edge in range(4):
        a = corners[edge]
        b = corners[(edge + 1) % 4]

        for step in range(101):
            fraction = step / 100.0
            ring.AddPoint_2D(
                a[0] + fraction * (b[0] - a[0]),
                a[1] + fraction * (b[1] - a[1]),
            )

    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(ring)

    if polygon.Transform(
        osr.CoordinateTransformation(source_crs, destination_crs)
    ) != 0:
        raise RuntimeError("Could not transform extent to EPSG:3067.")

    result = polygon.GetEnvelope()  # xmin, xmax, ymin, ymax
    return result[0], result[2], result[1], result[3]


def raster_extent(path):
    """Get the full TIFF extent, including NoData areas."""
    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    if dataset is None:
        raise ValueError(f"Cannot open raster: {path}")

    projection = dataset.GetProjection()
    if not projection:
        raise ValueError(f"Raster has no CRS: {path}")

    source_crs = osr.SpatialReference()
    source_crs.ImportFromWkt(projection)
    source_crs = normalized_srs(source_crs)

    gt = dataset.GetGeoTransform()

    # Match the north-up raster requirement of make_tiled_dem.py.
    if gt[2] != 0 or gt[4] != 0 or gt[1] <= 0 or gt[5] >= 0:
        raise ValueError(
            "Reference TIFF must be north-up without rotation, "
            "as required by the current DEM generator."
        )

    bounds = (
        gt[0],
        gt[3] + dataset.RasterYSize * gt[5],
        gt[0] + dataset.RasterXSize * gt[1],
        gt[3],
    )

    dataset = None
    return transform_bounds(bounds, source_crs, target_srs())


def gpkg_extent(path, layer_name):
    """Combine the rectangular extents of selected spatial layers."""
    dataset = ogr.Open(str(path), 0)
    if dataset is None:
        raise ValueError(f"Cannot open GeoPackage: {path}")

    if layer_name:
        layer = dataset.GetLayerByName(layer_name)
        if layer is None:
            raise ValueError(
                f"Layer {layer_name!r} not found in {path}"
            )
        layers = [layer]
    else:
        layers = [
            dataset.GetLayerByIndex(i)
            for i in range(dataset.GetLayerCount())
        ]

    extents = []

    for layer in layers:
        if layer.GetGeomType() == ogr.wkbNone:
            continue

        if layer.GetFeatureCount() == 0:
            continue

        crs = layer.GetSpatialRef()
        if crs is None:
            raise ValueError(
                f"GeoPackage layer {layer.GetName()!r} has no CRS."
            )

        extent = layer.GetExtent(force=1)
        if extent is None:
            continue

        # OGR order: xmin, xmax, ymin, ymax.
        bounds = (
            extent[0], extent[2],
            extent[1], extent[3],
        )

        extents.append(
            transform_bounds(
                bounds,
                normalized_srs(crs),
                target_srs(),
            )
        )

        print(f"Using GeoPackage layer: {layer.GetName()}")

    dataset = None

    if not extents:
        raise ValueError(
            "No nonempty spatial layer found in the GeoPackage."
        )

    return (
        min(e[0] for e in extents),
        min(e[1] for e in extents),
        max(e[2] for e in extents),
        max(e[3] for e in extents),
    )


def archive_relative(value):
    """Normalize an index or manifest path to YEAR/.../file.laz."""
    text = unquote(str(value).strip()).replace("\\", "/").lstrip("/")

    if text.startswith(ARCHIVE_PREFIX):
        text = text[len(ARCHIVE_PREFIX):]

    return text


def manifest_paths(path):
    """Read archive-relative paths listed in download_manifest.csv."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    result = set()

    for row in rows:
        relative = archive_relative(row.get("path") or "")

        if not relative:
            url_path = urlsplit(row.get("url") or "").path
            marker = "/2008_latest/"

            if marker not in url_path:
                raise ValueError(
                    f"Cannot parse manifest URL: {row.get('url')!r}"
                )

            relative = archive_relative(
                url_path.split(marker, 1)[1]
            )

        result.add(relative)

    return result


def indexed_footprints(index_path, region):
    """
    Get index polygons intersecting the buffered area.

    Index polygons are reprojected into EPSG:3067 before comparison.
    All indexed years are included, even if their files are not downloaded.
    """
    dataset = ogr.Open(str(index_path), 0)
    if dataset is None:
        raise ValueError(f"Cannot open coverage index: {index_path}")

    layer = dataset.GetLayer(0)
    if layer is None:
        raise ValueError("Coverage index contains no readable layer.")

    source_crs = layer.GetSpatialRef()
    if source_crs is None:
        raise ValueError(
            "Coverage index has no CRS. Check that its .prj file "
            "is present beside the .shp."
        )

    source_crs = normalized_srs(source_crs)
    destination_crs = target_srs()

    definition = layer.GetLayerDefn()
    fields = {
        definition.GetFieldDefn(i).GetName().lower():
        definition.GetFieldDefn(i).GetName()
        for i in range(definition.GetFieldCount())
    }

    if "path" not in fields:
        raise ValueError("Coverage index has no 'path' attribute.")

    # Use a native-CRS spatial filter to avoid reading unrelated features.
    query = region
    if not source_crs.IsSame(destination_crs):
        query_bounds = transform_bounds(
            region.bounds,
            destination_crs,
            source_crs,
        )
        query = box(*query_bounds)

    layer.SetSpatialFilterRect(*query.bounds)

    transformation = None
    if not source_crs.IsSame(destination_crs):
        transformation = osr.CoordinateTransformation(
            source_crs, destination_crs
        )

    parts = defaultdict(list)

    for feature in layer:
        path_value = feature.GetField(fields["path"])
        geometry = feature.GetGeometryRef()

        if path_value is None or geometry is None or geometry.IsEmpty():
            continue

        relative = archive_relative(path_value)
        year_part = relative.split("/", 1)[0]

        if not (len(year_part) == 4 and year_part.isdigit()):
            continue

        geometry = geometry.Clone()

        if transformation is not None:
            if geometry.Transform(transformation) != 0:
                raise RuntimeError(
                    f"Cannot transform footprint: {relative}"
                )

        footprint = make_valid(
            from_wkb(bytes(geometry.ExportToWkb()))
        )

        if not footprint.intersects(region):
            continue

        clipped = footprint.intersection(region)

        if clipped.area > 0:
            parts[relative].append(clipped)

    layer = None
    dataset = None

    return {
        relative: make_valid(union_all(geometries))
        for relative, geometries in parts.items()
    }


def check_coverage(footprints, region, listed, pointcloud_dir):
    """
    Identify the oldest files required at each location.

    Later years are evaluated only against coverage from earlier years.
    Files from the same year can all contribute.
    """
    by_year = defaultdict(list)

    for relative, footprint in footprints.items():
        year = int(relative.split("/", 1)[0])
        by_year[year].append((relative, footprint))

    older_coverage = GeometryCollection()
    required = []
    missing_manifest = []
    missing_file = []

    for year in sorted(by_year):
        entries = by_year[year]

        for relative, footprint in entries:
            owned = footprint.difference(older_coverage)

            if owned.is_empty or owned.area <= 0:
                continue

            required.append((year, relative))

            if relative not in listed:
                missing_manifest.append(relative)

            local = pointcloud_dir / Path(relative).name

            if not local.is_file():
                missing_file.append((relative, str(local)))

        older_coverage = make_valid(union_all([
            older_coverage,
            *[footprint for _, footprint in entries],
        ]))

    uncovered = region.difference(older_coverage)

    return required, missing_manifest, missing_file, uncovered


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Check downloaded oldest point-cloud coverage for a "
            "GeoPackage or TIFF extent plus processing buffer."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="GeoPackage, .tif, or .tiff defining the output rectangle.",
    )
    parser.add_argument(
        "--downloads", required=True,
        help="Parent folder containing manifest, _index/, and pointclouds/.",
    )
    parser.add_argument(
        "--buffer", type=float, default=20.0,
        help="Required extra coverage on each side in metres.",
    )
    parser.add_argument(
        "--layer",
        help="GeoPackage layer; default: all nonempty spatial layers.",
    )
    parser.add_argument(
        "--index",
        help="Override the default _index/2008_latest.shp.",
    )
    args = parser.parse_args()

    if args.buffer < 0:
        parser.error("--buffer cannot be negative.")

    input_path = Path(args.input).resolve()
    downloads = Path(args.downloads).resolve()
    index_path = (
        Path(args.index).resolve()
        if args.index
        else downloads / "_index" / "2008_latest.shp"
    )
    manifest_path = downloads / "download_manifest.csv"
    pointcloud_dir = downloads / "pointclouds"

    if not input_path.is_file():
        parser.error(f"Input does not exist: {input_path}")
    if not index_path.is_file():
        parser.error(f"Coverage index does not exist: {index_path}")
    if not manifest_path.is_file():
        parser.error(f"Download manifest does not exist: {manifest_path}")
    if not pointcloud_dir.is_dir():
        parser.error(
            f"Point-cloud folder does not exist: {pointcloud_dir}"
        )

    suffix = input_path.suffix.lower()

    if suffix == ".gpkg":
        bounds = gpkg_extent(input_path, args.layer)
    elif suffix in {".tif", ".tiff"}:
        if args.layer:
            parser.error("--layer applies only to a GeoPackage.")
        bounds = raster_extent(input_path)
    else:
        parser.error("Input must be .gpkg, .tif, or .tiff.")

    xmin, ymin, xmax, ymax = bounds

    buffered = box(
        xmin - args.buffer,
        ymin - args.buffer,
        xmax + args.buffer,
        ymax + args.buffer,
    )

    gdal.UseExceptions()
    ogr.UseExceptions()
    osr.UseExceptions()

    print(
        "\nOutput rectangle (EPSG:3067): "
        f"{xmin:.2f}, {ymin:.2f}, {xmax:.2f}, {ymax:.2f}"
    )
    print(
        f"Checked rectangle with {args.buffer:g} m on each side: "
        + ", ".join(f"{value:.2f}" for value in buffered.bounds)
    )

    listed = manifest_paths(manifest_path)
    footprints = indexed_footprints(index_path, buffered)

    required, missing_manifest, missing_file, uncovered = (
        check_coverage(
            footprints,
            buffered,
            listed,
            pointcloud_dir,
        )
    )

    print(f"\nOldest-acquisition files required: {len(required)}")

    counts = defaultdict(int)
    for year, _ in required:
        counts[year] += 1

    for year, count in sorted(counts.items()):
        print(f"  {year}: {count} file(s)")

    print(
        f"Uncovered by index: "
        f"{uncovered.area:.2f} m² "
        f"({100 * uncovered.area / buffered.area:.4f}%)"
    )

    if missing_manifest:
        print(
            f"\nFiles absent from download_manifest.csv "
            f"({len(missing_manifest)}):"
        )
        for relative in missing_manifest:
            print(f"  {relative}")

    if missing_file:
        print(
            f"\nRequired local point-cloud files missing "
            f"({len(missing_file)}):"
        )
        for relative, expected in missing_file:
            print(f"  {relative}")
            print(f"    expected: {expected}")

    # Tiny polygon slivers can arise from geometry precision.
    uncovered_tolerance_m2 = 0.01

    passed = (
        uncovered.area <= uncovered_tolerance_m2
        and not missing_manifest
        and not missing_file
        and len(required) > 0
    )

    if passed:
        print(
            "\nPASS: Indexed oldest-acquisition coverage and "
            "required files are available for the buffered extent."
        )
        return 0

    print(
        "\nFAIL: Do not assume the currently downloaded files "
        "fully support this buffered extent."
    )
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(2)
