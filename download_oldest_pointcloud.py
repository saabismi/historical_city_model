#!/usr/bin/env python3
"""
Download oldest available point clouds covering a TIFF or GeoPackage extent.

Usage:
    python download_oldest_lidar.py -i aerial.tif
    python download_oldest_lidar.py -i area.gpkg -o downloads
    python download_oldest_lidar.py -i area.gpkg \
        --extent-layer area --list-only

Arguments:
    -i, --input       Reference TIFF or GeoPackage.
    -o, --output-dir  Default: <input_stem>_pointcloud next to the input.
    --extent-layer    GeoPackage reference layer; default all spatial layers.
    --buffer          Extra coverage on EACH side, in metres; default 20.
    --index           Local 2008_latest.shp; otherwise fetch from CSC.
    --list-only       Write selection reports without downloading point clouds.

Outputs:
    download_manifest.csv
    coverage_report.txt
    pointclouds/*.las or *.laz
    _index/*          Automatically downloaded index components.

The reference defines a RECTANGLE, not a polygon-shaped mask.
GeoPackage bounds are calculated from actual saved geometries.
Raster NoData areas are included.

Extent calculations and coverage comparisons use EPSG:3067.
Files are selected oldest year first, retaining all relevant same-year files.
Whole point-cloud files are downloaded; points are not clipped here.

Existing manifest entries are preserved when an output folder is reused.
Existing point-cloud files are skipped, not checksum-verified.

Requires GDAL Python bindings and Requests:
    pip install requests
"""

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote

import requests
from osgeo import gdal, ogr, osr


# The index's path attribute may already include this prefix.
ARCHIVE_PREFIX = "mml/laserkeilaus/2008_latest/"

# Plain HTTP matches the working CSC access method used earlier.
# HTTP does not authenticate the server or protect download integrity.
ARCHIVE_URL = (
    "http://ftp.csc.fi/pub/sci/geo/geodata/"
    "mml/laserkeilaus/2008_latest/"
)


def target_crs():
    """Create EPSG:3067 with conventional easting/northing axis order."""
    crs = osr.SpatialReference()
    crs.ImportFromEPSG(3067)
    crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return crs


def normalized_crs(crs):
    """Clone a spatial reference before changing axis-order settings."""
    result = crs.Clone()
    result.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return result


def rectangle(bounds, subdivisions=100):
    """Create a rectangular polygon with densified edges."""
    xmin, ymin, xmax, ymax = bounds
    corners = [
        (xmin, ymin), (xmax, ymin),
        (xmax, ymax), (xmin, ymax),
    ]

    ring = ogr.Geometry(ogr.wkbLinearRing)

    for edge in range(4):
        start = corners[edge]
        end = corners[(edge + 1) % 4]

        for step in range(subdivisions):
            fraction = step / subdivisions
            ring.AddPoint_2D(
                start[0] + fraction * (end[0] - start[0]),
                start[1] + fraction * (end[1] - start[1]),
            )

    ring.AddPoint_2D(*corners[0])
    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(ring)
    return polygon


def bounds_in_3067(bounds, source_crs):
    """
    Transform a source extent into its enclosing EPSG:3067 rectangle.

    A densified boundary also handles individual layers with degenerate
    extents, such as one point, before multiple layer bounds are combined.
    """
    if not all(math.isfinite(value) for value in bounds):
        raise ValueError("Reference extent contains invalid coordinates.")

    source_crs = normalized_crs(source_crs)
    destination = target_crs()

    if source_crs.IsSame(destination):
        return bounds

    xmin, ymin, xmax, ymax = bounds
    corners = [
        (xmin, ymin), (xmax, ymin),
        (xmax, ymax), (xmin, ymax), (xmin, ymin),
    ]
    boundary = ogr.Geometry(ogr.wkbLineString)

    for edge in range(4):
        start, end = corners[edge], corners[edge + 1]
        for step in range(100):
            fraction = step / 100
            boundary.AddPoint_2D(
                start[0] + fraction * (end[0] - start[0]),
                start[1] + fraction * (end[1] - start[1]),
            )

    boundary.AddPoint_2D(*corners[-1])

    if boundary.Transform(
        osr.CoordinateTransformation(source_crs, destination)
    ) != 0:
        raise RuntimeError("Reference extent transformation failed.")

    left, right, bottom, top = boundary.GetEnvelope()
    result = left, bottom, right, top

    if not all(math.isfinite(value) for value in result):
        raise ValueError("Transformed reference extent is invalid.")

    return result


def raster_bounds(filename):
    """Read the full rectangle of a north-up raster, including NoData."""
    dataset = gdal.Open(str(filename), gdal.GA_ReadOnly)
    if dataset is None:
        raise ValueError(f"Cannot open raster: {filename}")

    projection = dataset.GetProjection()
    if not projection:
        raise ValueError("Reference raster has no CRS.")

    crs = osr.SpatialReference()
    crs.ImportFromWkt(projection)

    gt = dataset.GetGeoTransform()
    if gt[2] != 0 or gt[4] != 0 or gt[1] <= 0 or gt[5] >= 0:
        raise ValueError("Reference raster must be north-up without rotation.")

    bounds = (
        gt[0],
        gt[3] + dataset.RasterYSize * gt[5],
        gt[0] + dataset.RasterXSize * gt[1],
        gt[3],
    )

    dataset = None
    return bounds_in_3067(bounds, crs)


def geopackage_bounds(filename, layer_name):
    """
    Calculate bounds from actual saved GeoPackage geometries.

    Geometry shapes are used only to calculate their envelopes. The
    eventual selection area is a rectangle.
    """
    dataset = ogr.Open(str(filename), 0)
    if dataset is None:
        raise ValueError(f"Cannot open reference GeoPackage: {filename}")

    if layer_name:
        layer = dataset.GetLayerByName(layer_name)
        if layer is None:
            raise ValueError(f"Reference layer not found: {layer_name}")
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

        layer.SetSpatialFilter(None)
        layer.SetAttributeFilter(None)
        layer.ResetReading()

        xmin = ymin = float("inf")
        xmax = ymax = float("-inf")
        count = 0

        # Scan saved geometries instead of relying on cached layer bounds.
        for feature in layer:
            geometry = feature.GetGeometryRef()
            if geometry is None or geometry.IsEmpty():
                continue

            left, right, bottom, top = geometry.GetEnvelope()
            if not all(math.isfinite(value) for value in (
                left, right, bottom, top
            )):
                raise ValueError(
                    f"Invalid geometry in reference layer {layer.GetName()}."
                )

            xmin, ymin = min(xmin, left), min(ymin, bottom)
            xmax, ymax = max(xmax, right), max(ymax, top)
            count += 1

        if not count:
            continue

        crs = layer.GetSpatialRef()
        if crs is None:
            raise ValueError(
                f"Reference layer {layer.GetName()} has no CRS."
            )

        extents.append(bounds_in_3067(
            (xmin, ymin, xmax, ymax), crs
        ))
        print(f"Reference layer: {layer.GetName()} ({count} geometries)")

    layers = None
    dataset = None

    if not extents:
        raise ValueError("Reference GeoPackage has no usable geometries.")

    return (
        min(e[0] for e in extents),
        min(e[1] for e in extents),
        max(e[2] for e in extents),
        max(e[3] for e in extents),
    )


def download_file(session, url, destination, optional=False):
    """Stream a download to a temporary file, then rename after success."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")

    try:
        with session.get(
            url, stream=True, timeout=(30, 180)
        ) as response:
            if optional and response.status_code == 404:
                # Do not retain an old optional component when refreshing.
                destination.unlink(missing_ok=True)
                return False

            response.raise_for_status()
            expected = response.headers.get("Content-Length")
            written = 0

            with temporary.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                        written += len(chunk)

            if expected and not response.headers.get("Content-Encoding"):
                if written != int(expected):
                    raise RuntimeError(f"Incomplete download: {url}")

            if written == 0:
                raise RuntimeError(f"Empty download: {url}")

        temporary.replace(destination)
        return True

    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def fetch_index(session, output_dir):
    """Fetch the shapefile and companion files together."""
    index_dir = output_dir / "_index"
    index_dir.mkdir(parents=True, exist_ok=True)

    for extension in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
        filename = "2008_latest" + extension
        print(f"Fetching index: {filename}")

        download_file(
            session,
            ARCHIVE_URL + filename,
            index_dir / filename,
            optional=(extension == ".cpg"),
        )

    return index_dir / "2008_latest.shp"


def normalize_archive_path(value):
    """Validate and normalize an index path to YEAR/.../filename."""
    raw = unquote(str(value).strip()).replace("\\", "/").lstrip("/")

    if raw.startswith(ARCHIVE_PREFIX):
        raw = raw[len(ARCHIVE_PREFIX):]

    path = PurePosixPath(raw)

    if not raw or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe index path: {value!r}")

    if len(path.parts) < 2 or not re.fullmatch(r"\d{4}", path.parts[0]):
        raise ValueError(f"Cannot determine year from index path: {value!r}")

    year = int(path.parts[0])
    if year < 2008:
        raise ValueError(f"Unexpected acquisition year: {value!r}")

    return path.as_posix(), year


def polygon_intersection(first, second):
    """Return positive-area overlap; ignore boundary-only contact."""
    result = first.Intersection(second)

    if result is None:
        raise RuntimeError("Geometry intersection failed.")

    if result.IsEmpty() or result.GetArea() <= 0:
        return None

    return result


def read_candidates(index_filename, extent):
    """Read coverage polygons, transforming them into EPSG:3067."""
    dataset = ogr.Open(str(index_filename), 0)
    if dataset is None:
        raise ValueError(f"Cannot open index: {index_filename}")

    layer = dataset.GetLayer(0)
    if layer is None:
        raise ValueError("Coverage index has no layer.")

    index_crs = layer.GetSpatialRef()
    if index_crs is None:
        raise ValueError("Coverage index has no CRS; check its .prj file.")

    index_crs = normalized_crs(index_crs)
    destination = target_crs()

    definition = layer.GetLayerDefn()
    fields = {
        definition.GetFieldDefn(i).GetName().lower():
        definition.GetFieldDefn(i).GetName()
        for i in range(definition.GetFieldCount())
    }
    if "path" not in fields:
        raise ValueError("Coverage index has no path field.")

    # Express a densified spatial filter in the index's native CRS.
    query = extent.Clone()
    transformation = None

    if not index_crs.IsSame(destination):
        query.Transform(osr.CoordinateTransformation(
            destination, index_crs
        ))
        transformation = osr.CoordinateTransformation(
            index_crs, destination
        )

    # Use the transformed filter's bounding box to avoid excluding candidates
    # solely because of small reprojection approximation differences.
    left, right, bottom, top = query.GetEnvelope()
    layer.SetSpatialFilterRect(left, bottom, right, top)

    candidates = defaultdict(dict)

    for feature in layer:
        geometry = feature.GetGeometryRef()
        if geometry is None or geometry.IsEmpty():
            continue

        geometry = geometry.Clone()
        if transformation:
            if geometry.Transform(transformation) != 0:
                raise RuntimeError("Index polygon transformation failed.")

        if not geometry.IsValid():
            geometry = geometry.MakeValid()

        clipped = polygon_intersection(geometry, extent)
        if clipped is None:
            continue

        value = feature.GetField(fields["path"])
        if value is None or not str(value).strip():
            raise ValueError("Intersecting index polygon has an empty path.")

        relative, year = normalize_archive_path(value)
        previous = candidates[year].get(relative)

        if previous is None:
            candidates[year][relative] = clipped
        else:
            combined = previous.Union(clipped)
            if combined is None:
                raise RuntimeError("Combining index footprints failed.")
            candidates[year][relative] = combined

    layer = None
    dataset = None
    return candidates


def select_oldest(extent, candidates):
    """Select all relevant same-year files before considering later years."""
    remaining = extent.Clone()
    selected = []

    for year in sorted(candidates):
        if remaining.IsEmpty() or remaining.GetArea() <= 0:
            break

        yearly_coverage = None

        for relative, footprint in sorted(candidates[year].items()):
            contribution = polygon_intersection(footprint, remaining)
            if contribution is None:
                continue

            selected.append((year, relative))

            if yearly_coverage is None:
                yearly_coverage = contribution
            else:
                yearly_coverage = yearly_coverage.Union(contribution)
                if yearly_coverage is None:
                    raise RuntimeError("Combining yearly coverage failed.")

        if yearly_coverage is not None:
            remaining = remaining.Difference(yearly_coverage)
            if remaining is None:
                raise RuntimeError("Calculating remaining coverage failed.")

    return selected, remaining


def write_manifest(output_dir, downloads):
    """
    Merge the current selection into an existing manifest.

    This preserves records for files downloaded for previous areas. All
    recorded local paths are updated to the current flat pointclouds folder.
    """
    manifest = output_dir / "download_manifest.csv"
    records = {}

    if manifest.exists():
        with manifest.open(
            "r", newline="", encoding="utf-8-sig"
        ) as handle:
            reader = csv.DictReader(handle)

            required = {"year", "path", "url", "local_path"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(
                    "Existing manifest has an unsupported format. "
                    "Use a new output folder or migrate that manifest."
                )

            for row in reader:
                relative, year = normalize_archive_path(row["path"])
                records[relative] = (
                    year, relative,
                    ARCHIVE_URL + quote(relative, safe="/"),
                    output_dir / "pointclouds" / PurePosixPath(relative).name,
                )

    for entry in downloads:
        records[entry[1]] = entry

    # Protect flat storage against duplicate remote filenames.
    filenames = {}
    for _, relative, _, destination in records.values():
        key = destination.name.casefold()
        if key in filenames and filenames[key] != relative:
            raise ValueError(
                "Two archive files share the same local filename:\n"
                f"  {filenames[key]}\n  {relative}"
            )
        filenames[key] = relative

    temporary = manifest.with_name(manifest.name + ".tmp")

    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["year", "path", "url", "local_path"])

        for entry in sorted(records.values(), key=lambda item: (item[0], item[1])):
            year, relative, url, destination = entry
            writer.writerow([year, relative, url, str(destination)])

    temporary.replace(manifest)
    return manifest, len(records)


def main():
    parser = argparse.ArgumentParser(
        description="Download oldest LiDAR for a TIFF or GeoPackage rectangle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output-dir")
    parser.add_argument(
        "--extent-layer",
        help="Reference GeoPackage layer; default all spatial layers.",
    )
    parser.add_argument(
        "--buffer", type=float, default=20.0,
        help="Additional extent on each side in metres.",
    )
    parser.add_argument("--index", help="Local 2008_latest.shp.")
    parser.add_argument(
        "--list-only", action="store_true",
        help="Write selection reports without downloading point clouds.",
    )
    args = parser.parse_args()

    gdal.UseExceptions()
    ogr.UseExceptions()
    osr.UseExceptions()

    if not math.isfinite(args.buffer) or args.buffer < 0:
        parser.error("--buffer must be a finite, nonnegative number.")

    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        parser.error(f"Reference file does not exist: {input_path}")

    if input_path.suffix.lower() == ".gpkg":
        bounds = geopackage_bounds(input_path, args.extent_layer)
    elif input_path.suffix.lower() in {".tif", ".tiff"}:
        if args.extent_layer:
            parser.error("--extent-layer applies only to GeoPackages.")
        bounds = raster_bounds(input_path)
    else:
        parser.error("Input must be .gpkg, .tif, or .tiff.")

    xmin, ymin, xmax, ymax = bounds
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("Reference rectangle has zero width or height.")

    buffered_bounds = (
        xmin - args.buffer,
        ymin - args.buffer,
        xmax + args.buffer,
        ymax + args.buffer,
    )
    extent = rectangle(buffered_bounds)

    print("Reference rectangle in EPSG:3067:")
    print("  " + ", ".join(f"{v:.3f}" for v in bounds))
    print(f"Selection rectangle with {args.buffer:g} m buffer:")
    print("  " + ", ".join(f"{v:.3f}" for v in buffered_bounds))

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else input_path.parent / f"{input_path.stem}_pointcloud"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pointclouds").mkdir(exist_ok=True)

    with requests.Session() as session:
        session.headers.update({
            "User-Agent": "OldestMMLPointCloudDownloader/2.0",
            "Accept-Encoding": "identity",
        })

        if args.index:
            index_path = Path(args.index).resolve()
            if not index_path.is_file():
                parser.error(f"Local index does not exist: {index_path}")
        else:
            index_path = fetch_index(session, output_dir)

        print("Checking indexed oldest-acquisition coverage...")
        candidates = read_candidates(index_path, extent)
        selected, remaining = select_oldest(extent, candidates)

        downloads = [
            (
                year,
                relative,
                ARCHIVE_URL + quote(relative, safe="/"),
                output_dir / "pointclouds" / PurePosixPath(relative).name,
            )
            for year, relative in selected
        ]

        manifest, manifest_count = write_manifest(output_dir, downloads)

        missing_area = 0.0 if remaining.IsEmpty() else remaining.GetArea()
        missing_percent = 100 * missing_area / extent.GetArea()

        report = (
            f"Reference file: {input_path}\n"
            f"Reference layer: {args.extent_layer or 'all spatial layers / raster'}\n"
            f"Reference extent EPSG:3067: {bounds}\n"
            f"Buffer metres per side: {args.buffer}\n"
            f"Selection extent EPSG:3067: {buffered_bounds}\n"
            f"Coverage index: {index_path}\n"
            f"Files selected for this run: {len(downloads)}\n"
            f"Total manifest entries: {manifest_count}\n"
            f"Uncovered indexed area: {missing_area:.3f} m²\n"
            f"Uncovered indexed percentage: {missing_percent:.6f}%\n"
            "Coverage is based on index polygons, not actual ground points.\n"
            "Whole files are downloaded; points are not clipped here.\n"
        )
        (output_dir / "coverage_report.txt").write_text(
            report, encoding="utf-8"
        )

        counts = defaultdict(int)
        for year, _, _, _ in downloads:
            counts[year] += 1

        for year, count in sorted(counts.items()):
            print(f"  {year}: {count} file(s)")

        print(f"Selected for this extent: {len(downloads)} files")
        print(f"Uncovered by index: {missing_percent:.6f}%")
        print(f"Manifest: {manifest}")

        if missing_area > 0.01:
            print(
                "WARNING: Some of the buffered rectangle has no indexed "
                "coverage. Downloads cannot supply data absent from the index."
            )

        if args.list_only:
            print("List-only mode: no point clouds downloaded.")
            return

        for number, (_, _, url, destination) in enumerate(downloads, 1):
            if destination.exists():
                print(
                    f"[{number}/{len(downloads)}] "
                    f"Exists, skipping: {destination.name}"
                )
                continue

            print(f"[{number}/{len(downloads)}] Downloading {url}")
            download_file(session, url, destination)

    print(f"Finished: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDownload interrupted.", file=sys.stderr)
        sys.exit(1)
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
