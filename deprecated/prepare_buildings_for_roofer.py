#!/usr/bin/env python3
"""
Prepare building footprints and point-cloud data for a Roofer test.

Usage:
    python prepare_buildings_for_roofer.py ^
        --buildings MTK_2009_clipped/mtk-rakennus_2009_clipped.gpkg ^
        --downloads espoo1965_clouds ^
        --output roofer_test ^
        --assume-epsg-3067

Options:
    --buildings       Input GeoPackage.
    --layer           Footprint layer; default: rakennus.
    --downloads       Folder containing download_manifest.csv and
                      pointclouds/.
    --index           Coverage shapefile; default:
                      <downloads>/_index/2008_latest.shp
    --output          New or empty output folder.
    --count           Valid footprints to select; default: 100.
    --ground-buffer   Ground context radius in metres; default: 40.
    --chunk-size      LAS points read per chunk; default: 1,000,000.
    --assume-epsg-3067 assume that the data is in epsg 3067 CRS

Output:
    buildings.gpkg          Selected footprints with building_id.
    points.laz              Derived point cloud for Roofer.
    building_report.csv     Roof-candidate counts.
    preparation_report.json Input and coverage information.
    run_roofer.bat          Example Roofer command.

Selection rules:
    - Candidate roof points: original LAS class 1 inside an exact footprint.
      These are written as class 6.
    - Ground: original LAS class 2 within the ground context area.
    - Withheld points are excluded.
    - Older indexed acquisitions take spatial priority over later ones.

Original LAS/LAZ files are never modified.

Requires:
    GDAL Python bindings, NumPy, Shapely >= 2, pyproj, laspy and lazrs.
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote, urlsplit

import laspy
import numpy as np
import shapely
from osgeo import gdal, ogr, osr
from pyproj import CRS
from shapely import from_wkb, intersects_xy, make_valid, union_all
from shapely.geometry import box
from shapely.strtree import STRtree


EPSG = 3067

# Output coordinates are quantized to one millimetre.
OUTPUT_SCALE = 0.001


def target_osr():
    """Return EPSG:3067 using traditional easting/northing axis order."""
    crs = osr.SpatialReference()
    crs.ImportFromEPSG(EPSG)
    crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return crs


def normalized_crs(crs):
    """Clone a CRS before changing its axis-order setting."""
    result = crs.Clone()
    result.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return result


def polygonal_parts(geometry):
    """Extract polygon pieces from repaired polygon/collection geometry."""
    if geometry.is_empty:
        return []

    if geometry.geom_type == "Polygon":
        return [geometry]

    parts = []
    if hasattr(geometry, "geoms"):
        for child in geometry.geoms:
            parts.extend(polygonal_parts(child))
    return parts


def read_buildings(filename, layer_name, limit):
    """
    Select the first valid polygons in the layer's current reading order.

    IDs are saved into the output GeoPackage, which becomes the persistent
    reference for this test. No claim is made that source order is stable
    after rewriting the input GeoPackage.
    """
    dataset = ogr.Open(str(filename), 0)
    if dataset is None:
        raise ValueError(f"Cannot open building GeoPackage: {filename}")

    layer = dataset.GetLayerByName(layer_name)
    if layer is None:
        raise ValueError(f"Building layer not found: {layer_name}")

    crs = layer.GetSpatialRef()
    if crs is None:
        raise ValueError("Building layer has no CRS.")

    source_crs = normalized_crs(crs)
    destination_crs = target_osr()
    transformation = None

    if not source_crs.IsSame(destination_crs):
        print("Reprojecting building footprints to EPSG:3067.")
        transformation = osr.CoordinateTransformation(
            source_crs, destination_crs
        )

    buildings = []
    skipped = 0

    layer.ResetReading()

    for feature in layer:
        geometry = feature.GetGeometryRef()

        if geometry is None or geometry.IsEmpty():
            skipped += 1
            continue

        geometry = geometry.Clone()

        if transformation is not None:
            if geometry.Transform(transformation) != 0:
                raise RuntimeError("Building transformation failed.")

        repaired = make_valid(
            from_wkb(bytes(geometry.ExportToWkb()))
        )
        pieces = polygonal_parts(repaired)

        if not pieces:
            skipped += 1
            continue

        footprint = union_all(pieces)

        if footprint.is_empty or footprint.area <= 0:
            skipped += 1
            continue

        buildings.append({
            "id": f"building_{len(buildings) + 1:06d}",
            "geometry": footprint,
            # Informational only: not used as our building ID.
            "source_fid_at_selection": feature.GetFID(),
        })

        if len(buildings) >= limit:
            break

    dataset = None

    if not buildings:
        raise ValueError("No valid building polygons were selected.")

    if len(buildings) < limit:
        print(
            f"WARNING: Only {len(buildings)} valid buildings were found."
        )

    print(
        f"Selected {len(buildings)} buildings; "
        f"skipped {skipped} invalid/empty features."
    )

    return buildings


def write_buildings(filename, buildings):
    """Save selected footprints and their persistent project IDs."""
    driver = ogr.GetDriverByName("GPKG")
    dataset = driver.CreateDataSource(str(filename))

    if dataset is None:
        raise RuntimeError(f"Cannot create {filename}")

    layer = dataset.CreateLayer(
        "buildings",
        srs=target_osr(),
        geom_type=ogr.wkbMultiPolygon,
    )

    id_field = ogr.FieldDefn("building_id", ogr.OFTString)
    id_field.SetWidth(32)
    layer.CreateField(id_field)

    # This is only a trace for the current input file, not the project ID.
    layer.CreateField(
        ogr.FieldDefn("selection_source_fid", ogr.OFTInteger64)
    )

    for building in buildings:
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetField("building_id", building["id"])
        feature.SetField(
            "selection_source_fid",
            building["source_fid_at_selection"],
        )

        geometry = ogr.CreateGeometryFromWkb(
            shapely.to_wkb(building["geometry"])
        )
        feature.SetGeometry(ogr.ForceToMultiPolygon(geometry))
        layer.CreateFeature(feature)

    layer = None
    dataset = None


def archive_relative(value):
    """Convert an index/manifest path into YEAR/.../filename."""
    text = unquote(str(value).strip()).replace("\\", "/").lstrip("/")
    prefix = "mml/laserkeilaus/2008_latest/"

    if text.startswith(prefix):
        text = text[len(prefix):]

    return text


def manifest_relative(row):
    """Read archive-relative path from either supported manifest format."""
    path = row.get("path") or ""

    if path:
        return archive_relative(path)

    url = row.get("url") or ""
    url_path = urlsplit(url).path
    marker = "/2008_latest/"

    if marker not in url_path:
        raise ValueError(f"Cannot parse manifest URL: {url}")

    return archive_relative(url_path.split(marker, 1)[1])


def read_download_manifest(downloads):
    """Map archive paths to downloaded local point-cloud files."""
    filename = downloads / "download_manifest.csv"

    if not filename.is_file():
        raise ValueError(f"Missing download manifest: {filename}")

    with filename.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError("Download manifest is empty.")

    downloaded = {}

    for row in rows:
        relative = manifest_relative(row)
        year_text = row.get("year")

        if not year_text:
            raise ValueError(
                f"Manifest row has no year: {relative}"
            )

        year = int(year_text)
        first_component = relative.split("/", 1)[0]

        if not re.fullmatch(r"\d{4}", first_component):
            raise ValueError(f"Cannot determine path year: {relative}")

        if year != int(first_component):
            raise ValueError(
                f"Manifest year and path year disagree: {relative}"
            )

        # The downloader flattened files into the pointclouds folder.
        local = downloads / "pointclouds" / Path(relative).name

        # Older manifests may still point to a valid alternative location.
        if not local.is_file():
            recorded = row.get("local_path") or ""
            if recorded:
                local = Path(recorded)

        downloaded[relative] = {
            "path": local,
            "year": year,
        }

    return downloaded


def read_index_coverages(index_path, region):
    """
    Read all indexed acquisitions intersecting the test region.

    Considering index entries even when not downloaded prevents a newer
    acquisition from silently taking priority over an older missing file.
    """
    dataset = ogr.Open(str(index_path), 0)
    if dataset is None:
        raise ValueError(f"Cannot open coverage index: {index_path}")

    layer = dataset.GetLayer(0)
    if layer is None:
        raise ValueError("Coverage index has no readable layer.")

    index_crs = layer.GetSpatialRef()
    if index_crs is None:
        raise ValueError(
            "Coverage index has no CRS. Ensure its .prj file is present."
        )

    index_crs = normalized_crs(index_crs)
    destination_crs = target_osr()
    transformation = None

    if not index_crs.IsSame(destination_crs):
        print("Transforming coverage index polygons to EPSG:3067.")
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
        raise ValueError("Coverage index has no path field.")

    # Combine multiple index features referring to the same file.
    footprints = defaultdict(list)

    layer.ResetReading()

    for feature in layer:
        value = feature.GetField(fields["path"])
        geometry = feature.GetGeometryRef()

        if value is None or geometry is None or geometry.IsEmpty():
            continue

        relative = archive_relative(value)
        year_text = relative.split("/", 1)[0]

        if not re.fullmatch(r"\d{4}", year_text):
            continue

        geometry = geometry.Clone()

        if transformation is not None:
            if geometry.Transform(transformation) != 0:
                raise RuntimeError(
                    f"Cannot transform index geometry: {relative}"
                )

        footprint = make_valid(
            from_wkb(bytes(geometry.ExportToWkb()))
        )

        if footprint.is_empty or not footprint.intersects(region):
            continue

        clipped = make_valid(footprint.intersection(region))

        if not clipped.is_empty and clipped.area > 0:
            footprints[relative].append(clipped)

    dataset = None

    if not footprints:
        raise ValueError(
            "No coverage-index polygons intersect the test region."
        )

    by_year = defaultdict(list)

    for relative, pieces in footprints.items():
        year = int(relative.split("/", 1)[0])
        by_year[year].append({
            "relative": relative,
            "year": year,
            "footprint": make_valid(union_all(pieces)),
        })

    return by_year


def select_owned_sources(by_year, downloaded):
    """
    Apply oldest-year ownership to index footprints.

    All same-year files remain eligible in overlapping coverage.
    Fail if an oldest-owning file is missing rather than silently
    substituting newer data.
    """
    older = shapely.GeometryCollection()
    selected = []
    missing = []

    for year in sorted(by_year):
        entries = by_year[year]

        for entry in entries:
            owner = make_valid(entry["footprint"].difference(older))

            if owner.is_empty or owner.area <= 0:
                continue

            relative = entry["relative"]
            local = downloaded.get(relative)

            if local is None or not local["path"].is_file():
                missing.append(relative)
                continue

            selected.append({
                "relative": relative,
                "year": year,
                "path": local["path"],
                "owner": owner,
            })

        # All indexed coverage counts as older, including missing downloads.
        older = make_valid(union_all([
            older,
            *[entry["footprint"] for entry in entries],
        ]))

    if missing:
        preview = "\n".join(f"  {path}" for path in missing[:20])
        raise ValueError(
            "Some oldest-coverage files needed for the test are not "
            "available locally. Do not substitute newer data silently.\n"
            f"{preview}\n"
            f"Missing file count: {len(missing)}"
        )

    if not selected:
        raise ValueError("No usable downloaded point-cloud coverage.")

    return selected, older


def validate_las_crs(header, filename, assume_missing):
    """Check LAS CRS; missing CRS requires an explicit user assumption."""
    source = header.parse_crs()

    if source is None:
        if not assume_missing:
            raise ValueError(
                f"Point cloud has no CRS metadata: {filename}\n"
                "If you have verified its coordinates are EPSG:3067, "
                "rerun with --assume-epsg-3067."
            )

        print(
            f"WARNING: Assuming EPSG:3067 for {filename.name}"
        )
        return

    if not CRS(source).equals(CRS.from_epsg(EPSG), ignore_axis_order=True):
        raise ValueError(
            f"Point-cloud CRS is not EPSG:3067: {filename}"
        )


def write_derived_points(
    output_path,
    sources,
    buildings,
    region,
    chunk_size,
    assume_missing_crs,
):
    """
    Write a minimal derived LAZ with XYZ and classification.

    Class 1 inside a footprint becomes class 6. Class 2 is kept in the
    ground context region. Other classes and withheld points are excluded.
    """
    footprint_geometries = [
        building["geometry"] for building in buildings
    ]
    footprint_union = union_all(footprint_geometries)
    footprint_tree = STRtree(footprint_geometries)

    counts = np.zeros(len(buildings), dtype=np.int64)
    source_stats = []

    # LAS 1.2 point format 3 provides XYZ and standard classification.
    # Other attributes from the originals are deliberately not copied.
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array(
        [OUTPUT_SCALE, OUTPUT_SCALE, OUTPUT_SCALE]
    )

    # Local offsets keep scaled integer coordinates safely bounded.
    xmin, ymin, xmax, ymax = region.bounds
    header.offsets = np.array([
        np.floor(xmin / 1000) * 1000,
        np.floor(ymin / 1000) * 1000,
        0.0,
    ])
    header.add_crs(CRS.from_epsg(EPSG))

    total_ground = 0
    total_roof = 0

    with laspy.open(str(output_path), mode="w", header=header) as writer:
        for source in sources:
            print(
                f"Reading {source['year']}: {source['path'].name}",
                flush=True,
            )

            source_ground = 0
            source_roof = 0

            with laspy.open(source["path"]) as reader:
                validate_las_crs(
                    reader.header, source["path"], assume_missing_crs
                )

                for points in reader.chunk_iterator(chunk_size):
                    classes = np.asarray(points.classification)
                    candidate = (
                        np.isin(classes, [1, 2])
                        & ~np.asarray(points.withheld, dtype=bool)
                    )

                    if not np.any(candidate):
                        continue

                    x = np.asarray(points.x)[candidate]
                    y = np.asarray(points.y)[candidate]
                    z = np.asarray(points.z)[candidate]
                    original_class = classes[candidate]

                    in_bbox = (
                        np.isfinite(x)
                        & np.isfinite(y)
                        & np.isfinite(z)
                        & (x >= xmin)
                        & (x <= xmax)
                        & (y >= ymin)
                        & (y <= ymax)
                    )

                    x, y, z = x[in_bbox], y[in_bbox], z[in_bbox]
                    original_class = original_class[in_bbox]

                    if not len(x):
                        continue

                    # A source only contributes inside its oldest-owned area.
                    owned = intersects_xy(source["owner"], x, y)
                    in_region = intersects_xy(region, x, y)

                    roof = (
                        (original_class == 1)
                        & owned
                        & intersects_xy(footprint_union, x, y)
                    )
                    ground = (
                        (original_class == 2)
                        & owned
                        & in_region
                    )

                    keep = roof | ground

                    if not np.any(keep):
                        continue

                    # Count candidate roof observations per footprint.
                    # A point on overlapping footprints can count for both.
                    if np.any(roof):
                        roof_points = shapely.points(x[roof], y[roof])
                        pairs = footprint_tree.query(
                            roof_points, predicate="intersects"
                        )

                        if pairs.shape[1]:
                            counts += np.bincount(
                                pairs[1],
                                minlength=len(buildings),
                            )

                    output_points = laspy.ScaleAwarePointRecord.zeros(
                        int(keep.sum()), header=header
                    )
                    output_points.x = x[keep]
                    output_points.y = y[keep]
                    output_points.z = z[keep]
                    output_points.classification = np.where(
                        roof[keep], 6, 2
                    ).astype(np.uint8)

                    writer.write_points(output_points)

                    source_roof += int(roof.sum())
                    source_ground += int(ground.sum())

            total_roof += source_roof
            total_ground += source_ground

            source_stats.append({
                "archive_path": source["relative"],
                "local_path": str(source["path"].resolve()),
                "year": source["year"],
                "roof_candidate_points": source_roof,
                "ground_points": source_ground,
            })

            print(
                f"  Roof candidates: {source_roof}; "
                f"ground: {source_ground}"
            )

    if total_roof == 0:
        raise ValueError(
            "No class-1 points were found inside selected footprints. "
            "Check alignment, classification, and acquisition coverage."
        )

    if total_ground == 0:
        print(
            "WARNING: No ground points were found. "
            "Roofer may be unable to determine building base elevations."
        )

    return counts, source_stats, total_roof, total_ground


def write_roofer_command(output):
    """Write an editable Windows batch file using paths relative to itself."""
    text = """@echo off
cd /d "%~dp0"

REM Roofer 1.0.0 starting configuration.
REM Raw output and diagnostic crop output are preserved in roofer_output.
roofer ^
    --polygon-source-layer buildings ^
    --id-attribute building_id ^
    --bld-class 6 ^
    --grnd-class 2 ^
    --no-clip-terrain ^
    --lod12 ^
    --lod13 ^
    --crop-output ^
    --index ^
    -j 4 ^
    points.laz ^
    buildings.gpkg ^
    roofer_output

pause
"""
    (output / "run_roofer.bat").write_text(
        text, encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Prepare the first building footprints for Roofer."
    )
    parser.add_argument("--buildings", required=True)
    parser.add_argument("--layer", default="rakennus")
    parser.add_argument("--downloads", required=True)
    parser.add_argument("--index")
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--ground-buffer", type=float, default=40)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument(
        "--assume-epsg-3067",
        action="store_true",
        help=(
            "Allow LAS files with missing CRS metadata after manually "
            "verifying that their coordinates are EPSG:3067."
        ),
    )
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be positive.")
    if args.ground_buffer <= 0:
        parser.error("--ground-buffer must be positive.")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive.")

    gdal.UseExceptions()
    ogr.UseExceptions()
    osr.UseExceptions()

    building_path = Path(args.buildings).resolve()
    downloads = Path(args.downloads).resolve()
    output = Path(args.output).resolve()
    index_path = (
        Path(args.index).resolve()
        if args.index
        else downloads / "_index" / "2008_latest.shp"
    )

    if not building_path.is_file():
        parser.error(f"Building GeoPackage missing: {building_path}")
    if not index_path.is_file():
        parser.error(f"Coverage index missing: {index_path}")

    if output.exists() and (
        not output.is_dir() or any(output.iterdir())
    ):
        parser.error("Output folder must be empty or not yet exist.")

    buildings = read_buildings(
        building_path, args.layer, args.count
    )

    # Ground context only. Roof candidates still use exact footprints.
    footprint_union = union_all([
        building["geometry"] for building in buildings
    ])
    region = make_valid(
        footprint_union.buffer(args.ground_buffer)
    )

    downloaded = read_download_manifest(downloads)
    by_year = read_index_coverages(index_path, region)
    sources, indexed_coverage = select_owned_sources(
        by_year, downloaded
    )

    # Report uncovered context; do not pretend that coverage exists there.
    uncovered = region.difference(indexed_coverage)
    uncovered_area = float(uncovered.area)

    if uncovered_area > 0.01:
        print(
            f"WARNING: {uncovered_area:.1f} m² of the ground-context "
            "region has no indexed point-cloud coverage."
        )

    output.mkdir(parents=True, exist_ok=True)

    write_buildings(output / "buildings.gpkg", buildings)

    counts, source_stats, roof_total, ground_total = (
        write_derived_points(
            output / "points.laz",
            sources,
            buildings,
            region,
            args.chunk_size,
            args.assume_epsg_3067,
        )
    )

    with (output / "building_report.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "building_id",
            "selection_source_fid",
            "footprint_area_m2",
            "class_1_roof_candidates",
        ])

        for building, count in zip(buildings, counts):
            writer.writerow([
                building["id"],
                building["source_fid_at_selection"],
                round(building["geometry"].area, 2),
                int(count),
            ])

    report = {
        "input_buildings": str(building_path),
        "input_layer": args.layer,
        "coverage_index": str(index_path),
        "crs": "EPSG:3067",
        "building_count": len(buildings),
        "ground_context_buffer_metres": args.ground_buffer,
        "roof_selection": (
            "Original class 1 inside exact selected footprints, "
            "written as class 6"
        ),
        "ground_selection": (
            "Original class 2 inside ground-context region"
        ),
        "roof_candidate_points": roof_total,
        "ground_points": ground_total,
        "uncovered_context_area_m2": uncovered_area,
        "point_attributes": (
            "XYZ and classification only; other source attributes "
            "are not retained."
        ),
        "sources": source_stats,
        "warnings": [
            "Class-1 points are candidates, not verified roof points.",
            "Same-year overlapping files can both contribute points.",
            "Footprint/point-cloud date and alignment differences remain.",
            "Roof overhangs are not generated at this stage.",
            "Generated building IDs are persistent only if buildings.gpkg "
            "is retained and reused.",
        ],
    }

    (output / "preparation_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    write_roofer_command(output)

    print(f"\nPrepared {len(buildings)} buildings.")
    print(f"Roof candidates: {roof_total}")
    print(f"Ground points: {ground_total}")
    print(f"Output: {output}")
    print("Review building_report.csv, then run run_roofer.bat.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
