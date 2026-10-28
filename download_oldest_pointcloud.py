#!/usr/bin/env python3
"""
Download the oldest available point clouds covering an aerial raster.

Requirements:
    GDAL Python bindings and Requests (pip install requests).

Usage:
    python download_oldest_lidar.py -i aerial.tif
    python download_oldest_lidar.py -i aerial.tif --index /path/2008_latest.shp
    python download_oldest_lidar.py -i aerial.tif -o downloads --list-only

Arguments:
    -i, --input       Input georeferenced, north-up raster (required).
    -o, --output-dir  Output folder; default: <input_stem>_pointcloud,
                      next to the input raster.
    --index           Local coverage shapefile; otherwise fetch from CSC.
    --list-only       Write the manifest without downloading point clouds.
    -h, --help        Show help.

Outputs:
    download_manifest.csv, coverage_report.txt, and point-cloud files.
    Automatically fetched index files are stored in an "_index" subfolder.

Selection:
    Use the raster's FULL extent, including NoData areas.
    Select all same-year files intersecting coverage still needed at the
    start of that year. Later years are used only to fill remaining gaps.

Whole files are downloaded. Your subsequent merge must remove newer
points where older coverage exists; this script does not clip point clouds.
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote

import requests
from osgeo import gdal, ogr, osr


# The path attribute may include this prefix already.
ARCHIVE_PREFIX = "mml/laserkeilaus/2008_latest/"

# This public archive is accessed over plain HTTP.
# HTTP does not verify server identity or protect download integrity.
ARCHIVE_URL = (
    "http://ftp.csc.fi/pub/sci/geo/geodata/"
    "mml/laserkeilaus/2008_latest/"
)


def download_file(session, url, destination, optional=False):
    """Download atomically; return False for an optional missing resource."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Write incomplete data under a different name.
    temporary = destination.with_name(destination.name + ".part")

    try:
        with session.get(
            url, stream=True, timeout=(30, 180)
        ) as response:
            # The .cpg file is useful but not required.
            if optional and response.status_code == 404:
                return False

            # Fail rather than silently ignoring other HTTP errors.
            response.raise_for_status()

            expected = response.headers.get("Content-Length")
            written = 0

            # Stream large point clouds rather than loading them into RAM.
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                        written += len(chunk)

            # Check the response length when it represents unencoded bytes.
            if expected and not response.headers.get("Content-Encoding"):
                if written != int(expected):
                    raise RuntimeError(f"Incomplete download: {url}")

        # Rename only once the download has completed successfully.
        temporary.replace(destination)
        return True

    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def fetch_index(session, output_dir):
    """Download the shapefile and companion files into the output folder."""
    index_dir = output_dir / "_index"
    index_dir.mkdir(parents=True, exist_ok=True)

    # These files provide geometry, geometry indexing, attributes, and CRS.
    required_extensions = [".shp", ".shx", ".dbf", ".prj"]

    for extension in required_extensions + [".cpg"]:
        filename = "2008_latest" + extension
        destination = index_dir / filename

        print(f"Fetching index: {filename}")

        # Refresh all components together rather than using a potentially
        # stale mixture of cached shapefile components.
        download_file(
            session,
            ARCHIVE_URL + filename,
            destination,
            optional=(extension == ".cpg"),
        )

    return index_dir / "2008_latest.shp"


def raster_extent_in_crs(filename, target_crs):
    """
    Construct the full raster rectangle and transform it to the index CRS.

    Edges are densified before transformation because straight edges can
    become curved when changing coordinate reference systems.
    """
    raster = gdal.Open(str(filename), gdal.GA_ReadOnly)
    if raster is None:
        raise RuntimeError(f"Cannot open raster: {filename}")

    projection = raster.GetProjection()
    if not projection:
        raise ValueError("The input raster has no CRS.")

    gt = raster.GetGeoTransform()

    # This script expects the conventional north-up raster layout.
    if gt[2] != 0 or gt[4] != 0 or gt[1] <= 0 or gt[5] >= 0:
        raise ValueError(
            "The raster must be north-up without rotation. "
            "Use gdalwarp to create a north-up raster first."
        )

    # Calculate the full extent; no pixel values or NoData masks are read.
    xmin = gt[0]
    xmax = xmin + raster.RasterXSize * gt[1]
    ymax = gt[3]
    ymin = ymax + raster.RasterYSize * gt[5]

    corners = [
        (xmin, ymin),
        (xmax, ymin),
        (xmax, ymax),
        (xmin, ymax),
    ]

    ring = ogr.Geometry(ogr.wkbLinearRing)

    # Sample every edge to approximate its shape after reprojection.
    subdivisions = 100
    for index in range(4):
        start = corners[index]
        end = corners[(index + 1) % 4]

        for step in range(subdivisions):
            fraction = step / subdivisions
            ring.AddPoint_2D(
                start[0] + fraction * (end[0] - start[0]),
                start[1] + fraction * (end[1] - start[1]),
            )

    ring.AddPoint_2D(*corners[0])

    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(ring)

    source_crs = osr.SpatialReference()
    source_crs.ImportFromWkt(projection)
    source_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    # Use x/y order consistently, including with GDAL 3+.
    target_crs = target_crs.Clone()
    target_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    polygon.Transform(
        osr.CoordinateTransformation(source_crs, target_crs)
    )

    raster = None
    return polygon


def normalize_archive_path(value):
    """Convert the index's path attribute into a safe archive-relative path."""
    raw = str(value).strip().replace("\\", "/").lstrip("/")

    # Remove the prefix already represented by ARCHIVE_URL.
    if raw.startswith(ARCHIVE_PREFIX):
        raw = raw[len(ARCHIVE_PREFIX):]

    # Decode escapes before checking for unsafe path components.
    raw = unquote(raw)
    path = PurePosixPath(raw)

    if not raw or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe or empty index path: {value!r}")

    # Expected result: 2008/L413/1/L4131H1.laz
    if not path.parts or not re.fullmatch(r"\d{4}", path.parts[0]):
        raise ValueError(
            f"Cannot determine year from index path: {value!r}"
        )

    year = int(path.parts[0])
    if year < 2008:
        raise ValueError(f"Unexpected year in index path: {value!r}")

    return path.as_posix(), year


def polygon_intersection(first, second):
    """Return a positive-area intersection, or None for boundary-only contact."""
    result = first.Intersection(second)

    if result is None:
        raise RuntimeError("Geometry intersection failed.")

    if result.IsEmpty() or result.GetArea() <= 0:
        return None

    return result


def read_candidates(index_filename, raster_filename):
    """Read intersecting coverage polygons and group them by year and file."""
    dataset = ogr.Open(str(index_filename))
    if dataset is None:
        raise RuntimeError(f"Cannot open index: {index_filename}")

    layer = dataset.GetLayer(0)
    if layer is None:
        raise ValueError("The index has no readable layer.")

    # Locate the path field case-insensitively.
    definition = layer.GetLayerDefn()
    fields = {
        definition.GetFieldDefn(i).GetName().lower():
        definition.GetFieldDefn(i).GetName()
        for i in range(definition.GetFieldCount())
    }

    if "path" not in fields:
        raise ValueError(
            f"The index has no 'path' field. Fields: {list(fields.values())}"
        )

    index_crs = layer.GetSpatialRef()
    if index_crs is None:
        raise ValueError("The coverage index has no CRS; check its .prj file.")

    # Requiring a projected CRS makes area-based coverage reporting meaningful.
    if not index_crs.IsProjected():
        raise ValueError("The coverage index must use a projected CRS.")

    extent = raster_extent_in_crs(raster_filename, index_crs)

    if extent.IsEmpty() or extent.GetArea() <= 0:
        raise ValueError("The transformed raster extent has no area.")

    # Filter unrelated index features before checking exact intersections.
    layer.SetSpatialFilter(extent)

    # A file may have multiple features; combine those footprints so it is
    # downloaded only once.
    candidates = defaultdict(dict)

    for feature in layer:
        geometry = feature.GetGeometryRef()
        if geometry is None or geometry.IsEmpty():
            continue

        # Repair invalid polygons where GDAL supports MakeValid.
        geometry = geometry.Clone()
        if not geometry.IsValid():
            geometry = geometry.MakeValid()

        clipped = polygon_intersection(geometry, extent)
        if clipped is None:
            continue

        value = feature.GetField(fields["path"])
        if value is None or not str(value).strip():
            raise ValueError(
                f"Intersecting feature {feature.GetFID()} has an empty path."
            )

        relative_path, year = normalize_archive_path(value)
        existing = candidates[year].get(relative_path)

        if existing is None:
            candidates[year][relative_path] = clipped
        else:
            combined = existing.Union(clipped)
            if combined is None:
                raise RuntimeError("Combining index footprints failed.")
            candidates[year][relative_path] = combined

    dataset = None
    return extent, candidates


def select_oldest(extent, candidates):
    """Select files oldest first, retaining all relevant files within each year."""
    remaining = extent.Clone()
    selected = []

    for year in sorted(candidates):
        if remaining.IsEmpty() or remaining.GetArea() <= 0:
            break

        # Do not reduce 'remaining' while testing files of the same year.
        # This preserves all same-year split/overlapping files needed for
        # the area that was uncovered at the start of that year.
        year_coverage = None

        for relative_path, footprint in sorted(candidates[year].items()):
            contribution = polygon_intersection(footprint, remaining)
            if contribution is None:
                continue

            selected.append((year, relative_path))

            if year_coverage is None:
                year_coverage = contribution
            else:
                year_coverage = year_coverage.Union(contribution)
                if year_coverage is None:
                    raise RuntimeError("Combining yearly coverage failed.")

        # Later years may contribute only outside this year's coverage.
        if year_coverage is not None:
            remaining = remaining.Difference(year_coverage)
            if remaining is None:
                raise RuntimeError("Calculating remaining coverage failed.")

    return selected, remaining


def main():
    parser = argparse.ArgumentParser(
        description="Download the oldest point clouds covering a raster.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i", "--input", required=True,
        help="Input georeferenced, north-up aerial raster.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        help="Output folder; default: <input_stem>_pointcloud next to input.",
    )
    parser.add_argument(
        "--index",
        help="Local 2008_latest.shp; otherwise download the index from CSC.",
    )
    parser.add_argument(
        "--list-only", action="store_true",
        help="Write reports without downloading point clouds.",
    )
    args = parser.parse_args()

    # Raise exceptions instead of silently returning failed GDAL operations.
    gdal.UseExceptions()
    ogr.UseExceptions()

    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        parser.error(f"Input raster does not exist: {input_path}")

    # For /data/aerial.tif, default output is /data/aerial_pointcloud.
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else input_path.parent / f"{input_path.stem}_pointcloud"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # A session reuses network connections and closes them on exit.
    with requests.Session() as session:
        session.headers.update({
            "User-Agent": "OldestMMLPointCloudDownloader/1.0",
            "Accept-Encoding": "identity",
        })

        if args.index:
            index_path = Path(args.index).resolve()
            if not index_path.is_file():
                parser.error(f"Local index does not exist: {index_path}")
        else:
            index_path = fetch_index(session, output_dir)

        print("Reading coverage index and checking the raster extent...")
        extent, candidates = read_candidates(index_path, input_path)
        selected, remaining = select_oldest(extent, candidates)

        # URL-encode filename/path characters but preserve directory slashes.
        downloads = []
        for year, relative_path in selected:
            url = ARCHIVE_URL + quote(relative_path, safe="/")
            # Save all point clouds directly in the output folder.
            # The remote directory structure is still retained in the URL.
            filename = PurePosixPath(relative_path).name
            destination = output_dir / "pointclouds" / filename
            downloads.append((year, relative_path, url, destination))

        # Detect duplicate filenames before downloading anything.
        # Case-insensitive comparison also protects Windows output folders.
        destinations = {}

        for _, relative_path, _, destination in downloads:
            key = destination.name.casefold()

            if key in destinations:
                raise ValueError(
                    "Two selected files have the same output filename:\n"
                    f"  {destinations[key]}\n"
                    f"  {relative_path}\n"
                    "They cannot both be saved in one folder without renaming."
                )

            destinations[key] = relative_path


        # Always write the manifest, including in preview mode.
        manifest = output_dir / "download_manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["year", "path", "url", "local_path"])

            for year, relative_path, url, destination in downloads:
                writer.writerow([
                    year,
                    relative_path,
                    url,
                    str(destination),
                ])

        # Report any portion of the raster rectangle lacking indexed coverage.
        total_area = extent.GetArea()
        missing_area = 0 if remaining.IsEmpty() else remaining.GetArea()
        missing_percent = 100 * missing_area / total_area

        report = (
            f"Input raster: {input_path}\n"
            f"Coverage index: {index_path}\n"
            f"Selected files: {len(downloads)}\n"
            f"Uncovered extent: {missing_percent:.6f}%\n"
            "Coverage percentage is based on polygon area in the index CRS.\n"
            "Selection uses index footprints, not actual point locations.\n"
        )
        (output_dir / "coverage_report.txt").write_text(
            report, encoding="utf-8"
        )

        # Summarize how many files were selected from each year.
        counts = defaultdict(int)
        for year, _, _, _ in downloads:
            counts[year] += 1

        for year, count in sorted(counts.items()):
            print(f"  {year}: {count} file(s)")

        print(f"Selected files: {len(downloads)}")
        print(f"Manifest: {manifest}")
        print(f"Uncovered extent: {missing_percent:.6f}%")

        if missing_area > 0:
            print(
                "WARNING: Some raster coverage is missing from the index "
                "(very small values may be geometry precision effects)."
            )

        if args.list_only:
            print("List-only mode: point clouds were not downloaded.")
            return

        # Download whole files, preserving year/subdirectory structure.
        for number, (_, _, url, destination) in enumerate(downloads, 1):
            # Skip existing files to allow rerunning an interrupted job.
            # Their contents are not checksum-verified, so use this folder
            # only for downloads managed by this script.
            if destination.exists():
                print(
                    f"[{number}/{len(downloads)}] "
                    f"Already exists, skipping: {destination.name}"
                )
                continue

            print(f"[{number}/{len(downloads)}] Downloading {url}")
            download_file(session, url, destination)

        print(f"Finished. Output folder: {output_dir}")


if __name__ == "__main__":
    main()
