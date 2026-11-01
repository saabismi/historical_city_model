#!/usr/bin/env python3
"""
Generate oldest-acquisition ground DEM tiles and optional terrain GLBs.

Typical usage:
    python make_tiled_dem.py \
        -i espoo1965_clouds \
        -o espoo_terrain_corrected \
        --reference espoo1965_merged.tif

Input folder:
    download_manifest.csv
    _index/2008_latest.shp and its companion files
    pointclouds/*.las or *.laz

Arguments:
    -i, --input-dir       Parent download folder described above.
    -o, --output-dir      New or empty output folder.
    --reference           Raster defining the full rectangular output extent.
    --reference-gpkg      Alternatively, use a GeoPackage's bounding extent.
    --reference-layer     Optional layer with --reference-gpkg.
    --index               Override the coverage shapefile location.
    --crs                 Output/point-cloud CRS; default EPSG:3067.
    --tile-size           Core size in metres; default 250.
    --resolution          DEM cell size in metres; default 1.
    --overlap             Processing buffer on each side; default 20 m.
    --max-edge            Maximum TIN triangle edge in metres; default 50.
                          Use 0 to disable.
    --chunk-size          LAS points per read; default 1,000,000.
    --no-glb              Do not generate terrain GLBs.

Outputs:
    dem/*.tif
    glb/*.glb, unless --no-glb
    tile_manifest.json

Selection rule:
    Older indexed acquisitions own their coverage. Later acquisitions supply
    points only outside that older coverage. Same-year files may overlap.

This script does not modify the downloaded point clouds. Index footprints
represent nominal coverage, not necessarily actual ground-point availability.
"""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote, urlsplit

import laspy
import numpy as np
import rasterio
import trimesh
from osgeo import ogr, osr
from pyproj import CRS, Transformer
from rasterio.transform import from_origin
from scipy.spatial import Delaunay, QhullError
from shapely import (
    GeometryCollection,
    from_wkb,
    intersects_xy,
    make_valid,
    union_all,
)
from shapely.geometry import box
from shapely.strtree import STRtree


NODATA = -9999.0
ARCHIVE_PREFIX = "mml/laserkeilaus/2008_latest/"


def archive_relative(value):
    """Normalize an index or manifest path to YEAR/.../filename."""
    text = unquote(str(value).strip()).replace("\\", "/").lstrip("/")

    if text.startswith(ARCHIVE_PREFIX):
        text = text[len(ARCHIVE_PREFIX):]

    return text


def relative_from_manifest_row(row):
    """Support manifests with either a path column or only a URL."""
    if row.get("path"):
        return archive_relative(row["path"])

    url_path = urlsplit(row.get("url", "")).path
    marker = "/2008_latest/"

    if marker not in url_path:
        raise ValueError(
            f"Cannot determine archive path from manifest URL: "
            f"{row.get('url')!r}"
        )

    return archive_relative(url_path.split(marker, 1)[1])


def read_downloads(folder):
    """Read the manifest and locate each downloaded, flattened LAS/LAZ."""
    manifest = folder / "download_manifest.csv"

    if not manifest.is_file():
        raise ValueError(f"Download manifest not found: {manifest}")

    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"Download manifest is empty: {manifest}")

    downloads = {}
    local_names = {}

    for row in rows:
        relative = relative_from_manifest_row(row)
        components = relative.split("/")

        if len(components) < 2 or not re.fullmatch(
            r"\d{4}", components[0]
        ):
            raise ValueError(
                f"Invalid archive path in manifest: {relative!r}"
            )

        year = int(row["year"])

        if year != int(components[0]):
            raise ValueError(
                f"Manifest year differs from path year: {relative}"
            )

        filename = components[-1]

        # This project stores downloaded files in one flat subfolder.
        local = folder / "pointclouds" / filename

        if not local.is_file() and row.get("local_path"):
            recorded = Path(row["local_path"])

            if not recorded.is_absolute():
                recorded = folder / recorded

            if recorded.is_file():
                local = recorded

        if not local.is_file():
            # Keep the entry: ownership checking below will identify
            # whether this missing file is actually needed.
            local = None

        # Two different archive files cannot safely share one flattened
        # local filename. Do not silently use the wrong file.
        key = filename.casefold()
        previous = local_names.get(key)

        if previous is not None and previous != relative:
            raise ValueError(
                "Different archive files share a flattened filename:\n"
                f"  {previous}\n"
                f"  {relative}"
            )

        local_names[key] = relative
        downloads[relative] = {
            "year": year,
            "local": local,
        }

    return downloads


def inspect_downloaded_files(downloads, expected_crs):
    """Read headers for existing point-cloud files and verify their CRS."""
    headers = {}
    missing_crs = 0

    for relative, entry in downloads.items():
        filename = entry["local"]

        if filename is None:
            continue

        with laspy.open(filename) as reader:
            header = reader.header

            if header.point_count == 0:
                continue

            source_crs = header.parse_crs()

            if source_crs is None:
                missing_crs += 1
            elif not CRS(source_crs).equals(
                expected_crs, ignore_axis_order=True
            ):
                raise ValueError(
                    f"Point-cloud CRS differs from --crs: {filename}"
                )

            headers[relative] = (
                float(header.mins[0]),
                float(header.mins[1]),
                float(header.maxs[0]),
                float(header.maxs[1]),
            )

    if missing_crs:
        print(
            f"WARNING: {missing_crs} point-cloud file(s) have no CRS "
            "metadata. Assuming their coordinates use --crs."
        )

    if not headers:
        raise ValueError("No nonempty downloaded LAS/LAZ files found.")

    return headers


def raster_extent(reference, output_crs):
    """Read the full rectangle of a north-up reference raster."""
    with rasterio.open(reference) as dataset:
        if dataset.crs is None:
            raise ValueError("Reference raster has no CRS.")

        if not CRS(dataset.crs).equals(
            output_crs, ignore_axis_order=True
        ):
            raise ValueError(
                "Reference raster must use the selected output CRS."
            )

        t = dataset.transform

        if t.b != 0 or t.d != 0 or t.a <= 0 or t.e >= 0:
            raise ValueError(
                "Reference raster must be north-up without rotation."
            )

        b = dataset.bounds
        return b.left, b.bottom, b.right, b.top


def gpkg_extent(filename, layer_name, output_crs):
    """Use the rectangular bounds of one or all spatial GPKG layers."""
    dataset = ogr.Open(str(filename), 0)

    if dataset is None:
        raise ValueError(f"Cannot open reference GeoPackage: {filename}")

    if layer_name:
        layer = dataset.GetLayerByName(layer_name)

        if layer is None:
            raise ValueError(
                f"Reference layer not found: {layer_name}"
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

        source_srs = layer.GetSpatialRef()

        if source_srs is None:
            raise ValueError(
                f"GeoPackage layer has no CRS: {layer.GetName()}"
            )

        extent = layer.GetExtent(force=1)

        if extent is None:
            continue

        xmin, xmax, ymin, ymax = map(float, extent)

        if xmin >= xmax or ymin >= ymax:
            continue

        source_crs = CRS.from_wkt(source_srs.ExportToWkt())

        if source_crs.equals(output_crs, ignore_axis_order=True):
            extents.append((xmin, ymin, xmax, ymax))
        else:
            print(
                f"WARNING: Reprojecting extent of layer "
                f"{layer.GetName()} to {output_crs.to_string()}."
            )

            transformed = Transformer.from_crs(
                source_crs, output_crs, always_xy=True
            ).transform_bounds(
                xmin, ymin, xmax, ymax, densify_pts=100
            )

            extents.append(transformed)

    dataset = None

    if not extents:
        raise ValueError(
            "Reference GeoPackage has no usable spatial extent."
        )

    return (
        min(e[0] for e in extents),
        min(e[1] for e in extents),
        max(e[2] for e in extents),
        max(e[3] for e in extents),
    )


def requested_extent(args, headers, crs):
    """Choose the explicitly requested rectangle or LAS header union."""
    if args.reference:
        return raster_extent(args.reference, crs)

    if args.reference_gpkg:
        return gpkg_extent(
            args.reference_gpkg, args.reference_layer, crs
        )

    print(
        "WARNING: No reference supplied. Using whole downloaded "
        "point-cloud file extents."
    )

    return (
        min(b[0] for b in headers.values()),
        min(b[1] for b in headers.values()),
        max(b[2] for b in headers.values()),
        max(b[3] for b in headers.values()),
    )


def normalized_srs(srs):
    """Clone an OGR spatial reference with conventional X/Y order."""
    result = srs.Clone()
    result.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return result


def read_index_coverages(index_path, region, output_crs):
    """
    Read all indexed acquisition footprints intersecting region.

    This deliberately considers older index entries even when their files
    were not downloaded. Otherwise a newer local file could incorrectly
    become the apparent oldest source.
    """
    dataset = ogr.Open(str(index_path), 0)

    if dataset is None:
        raise ValueError(f"Cannot open coverage index: {index_path}")

    layer = dataset.GetLayer(0)

    if layer is None:
        raise ValueError("Coverage index contains no layer.")

    source_srs = layer.GetSpatialRef()

    if source_srs is None:
        raise ValueError(
            "Coverage index has no CRS. Check its accompanying .prj file."
        )

    source_srs = normalized_srs(source_srs)

    target_srs = osr.SpatialReference()
    target_srs.ImportFromWkt(output_crs.to_wkt())
    target_srs.SetAxisMappingStrategy(
        osr.OAMS_TRADITIONAL_GIS_ORDER
    )

    to_target = None

    if not source_srs.IsSame(target_srs):
        print(
            "Reprojecting coverage-index polygons to "
            f"{output_crs.to_string()}."
        )

        to_target = osr.CoordinateTransformation(
            source_srs, target_srs
        )

    # Spatially filter the large index in its native CRS.
    query = ogr.CreateGeometryFromWkb(region.wkb)

    if to_target is not None:
        to_source = osr.CoordinateTransformation(
            target_srs, source_srs
        )

        # Densify before transforming the filter to approximate curved
        # transformed boundaries.
        query.Segmentize(max(
            (region.bounds[2] - region.bounds[0]) / 100,
            (region.bounds[3] - region.bounds[1]) / 100,
            1.0,
        ))

        if query.Transform(to_source) != 0:
            raise RuntimeError(
                "Could not transform index spatial filter."
            )

    layer.SetSpatialFilter(query)

    definition = layer.GetLayerDefn()
    field_names = {
        definition.GetFieldDefn(i).GetName().lower():
        definition.GetFieldDefn(i).GetName()
        for i in range(definition.GetFieldCount())
    }

    if "path" not in field_names:
        raise ValueError("Coverage index has no 'path' field.")

    pieces = defaultdict(list)
    layer.ResetReading()

    for feature in layer:
        value = feature.GetField(field_names["path"])
        geometry = feature.GetGeometryRef()

        if value is None or geometry is None or geometry.IsEmpty():
            continue

        relative = archive_relative(value)
        first = relative.split("/", 1)[0]

        if not re.fullmatch(r"\d{4}", first):
            continue

        geometry = geometry.Clone()

        if to_target is not None:
            if geometry.Transform(to_target) != 0:
                raise RuntimeError(
                    f"Cannot reproject index polygon: {relative}"
                )

        footprint = make_valid(
            from_wkb(bytes(geometry.ExportToWkb()))
        )

        if footprint.is_empty or not footprint.intersects(region):
            continue

        clipped = footprint.intersection(region)

        if not clipped.is_empty and clipped.area > 0:
            pieces[relative].append(make_valid(clipped))

    layer = None
    dataset = None

    if not pieces:
        raise ValueError(
            "No coverage-index polygons intersect the processing area."
        )

    return [
        {
            "relative": relative,
            "year": int(relative.split("/", 1)[0]),
            "footprint": make_valid(union_all(geometries)),
        }
        for relative, geometries in pieces.items()
    ]


def build_oldest_sources(coverages, downloads, region):
    """
    Assign each indexed file its oldest-year ownership geometry.

    Only previously processed YEARS remove ownership. All files within
    the same year remain eligible even if their footprints overlap.
    """
    by_year = defaultdict(list)

    for entry in coverages:
        by_year[entry["year"]].append(entry)

    older = GeometryCollection()
    sources = []
    missing = []

    for year in sorted(by_year):
        year_entries = by_year[year]

        for entry in year_entries:
            owner = make_valid(
                entry["footprint"].difference(older)
            )

            if owner.is_empty or owner.area <= 0:
                continue

            relative = entry["relative"]
            downloaded = downloads.get(relative)

            if (
                downloaded is None
                or downloaded["local"] is None
                or not downloaded["local"].is_file()
            ):
                missing.append(relative)
                continue

            sources.append({
                "path": downloaded["local"],
                "year": year,
                "relative": relative,
                "owner": owner,
                "bounds": owner.bounds,
            })

        # Include all indexed footprints, including missing local files.
        # Never let a newer file replace missing older coverage silently.
        older = make_valid(union_all([
            older,
            *[entry["footprint"] for entry in year_entries],
        ]))

    if missing:
        examples = "\n".join(
            f"  {path}" for path in missing[:20]
        )

        raise ValueError(
            "Oldest-acquisition files required for the output extent "
            "and its processing buffer are missing:\n"
            f"{examples}\n"
            f"Total missing: {len(missing)}\n"
            "Download these files before generating the DEM."
        )

    if not sources:
        raise ValueError(
            "No downloaded files own coverage in the processing area."
        )

    uncovered = region.difference(older)

    if not uncovered.is_empty and uncovered.area > 0.01:
        print(
            "WARNING: The coverage index does not cover "
            f"{uncovered.area:.1f} m² of the buffered processing area. "
            "Those locations may remain NoData."
        )

    print(
        f"Using {len(sources)} oldest-owned point-cloud files."
    )

    return sources


def read_ground(sources, tree, bounds, chunk_size):
    """Read only class-2 points owned by intersecting source files."""
    query = box(*bounds)
    arrays = []

    for source_index in tree.query(
        query, predicate="intersects"
    ):
        source = sources[int(source_index)]
        owner = source["owner"].intersection(query)

        if owner.is_empty:
            continue

        with laspy.open(source["path"]) as reader:
            for points in reader.chunk_iterator(chunk_size):
                classification = np.asarray(points.classification)

                selected = (
                    (classification == 2)
                    & ~np.asarray(points.withheld, dtype=bool)
                )

                if not np.any(selected):
                    continue

                x = np.asarray(points.x)[selected]
                y = np.asarray(points.y)[selected]
                z = np.asarray(points.z)[selected]

                inside = (
                    np.isfinite(x)
                    & np.isfinite(y)
                    & np.isfinite(z)
                    & (x >= bounds[0])
                    & (x <= bounds[2])
                    & (y >= bounds[1])
                    & (y <= bounds[3])
                )

                x, y, z = x[inside], y[inside], z[inside]

                if not len(x):
                    continue

                # Boundary-inclusive predicate prevents losing points
                # exactly on an ownership polygon's border.
                owned = intersects_xy(owner, x, y)

                if np.any(owned):
                    arrays.append(np.column_stack((
                        x[owned],
                        y[owned],
                        z[owned],
                    )))

    if not arrays:
        return np.empty((0, 3), dtype=np.float64)

    return np.concatenate(arrays)


def build_tin(points, max_edge):
    """Create a ground TIN; suppress interpolation over large gaps."""
    xy, inverse = np.unique(
        points[:, :2],
        axis=0,
        return_inverse=True,
    )

    sums = np.bincount(
        inverse, weights=points[:, 2]
    )
    counts = np.bincount(inverse)
    elevations = sums / counts

    if len(xy) < 3:
        raise ValueError(
            "Fewer than three unique ground-point positions."
        )

    origin = xy.min(axis=0)
    local_xy = xy - origin
    tin = Delaunay(local_xy)

    valid_triangles = np.ones(
        len(tin.simplices),
        dtype=bool,
    )

    if max_edge > 0:
        vertices = local_xy[tin.simplices]

        edge_lengths = np.stack([
            np.linalg.norm(
                vertices[:, 0] - vertices[:, 1],
                axis=1,
            ),
            np.linalg.norm(
                vertices[:, 1] - vertices[:, 2],
                axis=1,
            ),
            np.linalg.norm(
                vertices[:, 2] - vertices[:, 0],
                axis=1,
            ),
        ], axis=1)

        valid_triangles = (
            edge_lengths.max(axis=1) <= max_edge
        )

    return tin, elevations, origin, valid_triangles


def interpolate(tin_data, x, y):
    """Interpolate TIN elevations onto the supplied regular grid."""
    tin, elevations, origin, valid_triangles = tin_data

    xx, yy = np.meshgrid(x, y)
    query = np.column_stack((
        xx.ravel(), yy.ravel()
    )) - origin

    simplex = tin.find_simplex(query)

    result = np.full(
        len(query), np.nan, dtype=np.float64
    )

    candidates = np.flatnonzero(simplex >= 0)

    usable = candidates[
        valid_triangles[simplex[candidates]]
    ]

    if len(usable):
        triangle_ids = simplex[usable]
        transforms = tin.transform[triangle_ids]

        delta = (
            query[usable] - transforms[:, 2, :]
        )

        first_weights = np.einsum(
            "ijk,ik->ij",
            transforms[:, :2, :],
            delta,
        )

        weights = np.column_stack((
            first_weights,
            1.0 - first_weights.sum(axis=1),
        ))

        triangle_z = elevations[
            tin.simplices[triangle_ids]
        ]

        result[usable] = np.sum(
            weights * triangle_z,
            axis=1,
        )

    return result.reshape(len(y), len(x))


def write_dem(
    filename, elevations, xmin, ymax, resolution, crs
):
    """Write one core-only DEM GeoTIFF."""
    data = np.where(
        np.isfinite(elevations), elevations, NODATA
    ).astype(np.float32)

    with rasterio.open(
        filename,
        "w",
        driver="GTiff",
        width=data.shape[1],
        height=data.shape[0],
        count=1,
        dtype="float32",
        crs=crs,
        transform=from_origin(
            xmin, ymax, resolution, resolution
        ),
        nodata=NODATA,
        tiled=True,
        compress="deflate",
        BIGTIFF="IF_SAFER",
    ) as output:
        output.write(data, 1)


def write_glb(filename, x, y, elevations, z_origin):
    """Write an untextured, tile-local terrain GLB."""
    xx, yy = np.meshgrid(x, y)

    vertices = np.column_stack((
        (xx - x[0]).ravel(),
        (elevations - z_origin).ravel(),
        (y[0] - yy).ravel(),
    ))

    rows, columns = elevations.shape
    indexes = np.arange(
        rows * columns
    ).reshape(rows, columns)

    a = indexes[:-1, :-1].ravel()
    b = indexes[:-1, 1:].ravel()
    c = indexes[1:, :-1].ravel()
    d = indexes[1:, 1:].ravel()

    faces = np.concatenate((
        np.column_stack((a, c, b)),
        np.column_stack((b, c, d)),
    ))

    finite = np.isfinite(elevations.ravel())
    faces = faces[np.all(finite[faces], axis=1)]

    if len(faces) == 0:
        return False

    used, inverse = np.unique(
        faces, return_inverse=True
    )

    mesh = trimesh.Trimesh(
        vertices=vertices[used],
        faces=inverse.reshape(-1, 3),
        process=False,
    )

    mesh.visual.vertex_colors = [180, 180, 180, 255]
    mesh.export(str(filename), file_type="glb")

    return True


def save_manifest(path, manifest):
    """Write the manifest after every tile, including skipped tiles."""
    path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate oldest-acquisition ground DEM tiles "
            "and optional GLBs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i", "--input-dir", required=True,
        help="Parent download folder containing manifest and pointclouds/.",
    )
    parser.add_argument(
        "-o", "--output-dir", required=True,
        help="New or empty output folder.",
    )
    parser.add_argument(
        "--index",
        help="Coverage shapefile; default <input-dir>/_index/2008_latest.shp.",
    )
    parser.add_argument(
        "--crs", default="EPSG:3067",
        help="Point-cloud/output projected CRS.",
    )

    references = parser.add_mutually_exclusive_group()

    references.add_argument(
        "--reference",
        help="Raster defining the full rectangular output extent.",
    )
    references.add_argument(
        "--reference-gpkg",
        help="GeoPackage defining the rectangular output extent.",
    )

    parser.add_argument(
        "--reference-layer",
        help="Layer to use with --reference-gpkg; default all spatial layers.",
    )
    parser.add_argument(
        "--tile-size", type=float, default=250,
    )
    parser.add_argument(
        "--resolution", type=float, default=1,
    )
    parser.add_argument(
        "--overlap", type=float, default=20,
    )
    parser.add_argument(
        "--max-edge", type=float, default=50,
    )
    parser.add_argument(
        "--no-glb", action="store_true",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1_000_000,
    )

    args = parser.parse_args()

    if args.reference_layer and not args.reference_gpkg:
        parser.error(
            "--reference-layer requires --reference-gpkg."
        )

    if args.tile_size <= 0 or args.resolution <= 0:
        parser.error(
            "Tile size and resolution must be positive."
        )

    if args.overlap < 0 or args.max_edge < 0:
        parser.error(
            "Overlap and maximum edge cannot be negative."
        )

    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive.")

    cell_count_float = (
        args.tile_size / args.resolution
    )
    cells_per_tile = round(cell_count_float)

    if (
        cells_per_tile < 1
        or not math.isclose(
            cell_count_float,
            cells_per_tile,
            abs_tol=1e-8,
        )
    ):
        parser.error(
            "Tile size must be an integer multiple of resolution."
        )

    crs = CRS.from_user_input(args.crs)

    if not crs.is_projected or any(
        not math.isclose(
            axis.unit_conversion_factor, 1.0
        )
        for axis in crs.axis_info[:2]
    ):
        parser.error(
            "Use a projected CRS with metre-based horizontal units."
        )

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    index_path = (
        Path(args.index).resolve()
        if args.index
        else input_dir / "_index" / "2008_latest.shp"
    )

    if not input_dir.is_dir():
        parser.error(
            f"Input folder does not exist: {input_dir}"
        )

    if not index_path.is_file():
        parser.error(
            f"Coverage index does not exist: {index_path}"
        )

    if args.reference and not Path(args.reference).is_file():
        parser.error(
            f"Reference raster does not exist: {args.reference}"
        )

    if (
        args.reference_gpkg
        and not Path(args.reference_gpkg).is_file()
    ):
        parser.error(
            f"Reference GeoPackage does not exist: "
            f"{args.reference_gpkg}"
        )

    if output_dir.exists() and (
        not output_dir.is_dir()
        or any(output_dir.iterdir())
    ):
        parser.error(
            "Output folder must be empty or not yet exist."
        )

    ogr.UseExceptions()
    osr.UseExceptions()

    downloads = read_downloads(input_dir)
    headers = inspect_downloaded_files(downloads, crs)

    xmin, ymin, xmax, ymax = requested_extent(
        args, headers, crs
    )

    if (
        not all(
            math.isfinite(v)
            for v in (xmin, ymin, xmax, ymax)
        )
        or xmin >= xmax
        or ymin >= ymax
    ):
        raise ValueError("Invalid requested extent.")

    total_cols = math.ceil(
        (xmax - xmin) / args.resolution
    )
    total_rows = math.ceil(
        (ymax - ymin) / args.resolution
    )

    actual_xmax = (
        xmin + total_cols * args.resolution
    )
    actual_ymin = (
        ymax - total_rows * args.resolution
    )

    # Ownership must cover every DEM tile's interpolation buffer, not
    # just the requested output rectangle.
    ownership_region = box(
        xmin - args.overlap,
        actual_ymin - args.overlap,
        actual_xmax + args.overlap,
        ymax + args.overlap,
    )

    print(
        f"Requested extent: {xmin:.3f}, {ymin:.3f}, "
        f"{xmax:.3f}, {ymax:.3f}"
    )
    print(
        f"Actual DEM extent: {xmin:.3f}, {actual_ymin:.3f}, "
        f"{actual_xmax:.3f}, {ymax:.3f}"
    )

    print("Reading acquisition coverage index...")

    coverages = read_index_coverages(
        index_path, ownership_region, crs
    )

    sources = build_oldest_sources(
        coverages, downloads, ownership_region
    )

    # Shapely 2 STRtree returns indexes into this exact geometry list.
    source_tree = STRtree([
        source["owner"] for source in sources
    ])

    # Create output only after input and ownership validation succeeds.
    dem_dir = output_dir / "dem"
    dem_dir.mkdir(parents=True, exist_ok=True)

    glb_dir = output_dir / "glb"

    if not args.no_glb:
        glb_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "crs": crs.to_string(),
        "vertical_reference": (
            "Inherited from inputs; not transformed"
        ),
        "resolution_metres": args.resolution,
        "tile_size_metres": args.tile_size,
        "processing_buffer_metres": args.overlap,
        "max_tin_edge_metres": args.max_edge,
        "requested_extent": [
            xmin, ymin, xmax, ymax
        ],
        "actual_grid_extent": [
            xmin, actual_ymin, actual_xmax, ymax
        ],
        "extent_source": (
            str(Path(args.reference).resolve())
            if args.reference
            else str(Path(args.reference_gpkg).resolve())
            if args.reference_gpkg
            else "union_of_downloaded_pointcloud_file_bounds"
        ),
        "reference_layer": args.reference_layer,
        "grid_origin": [xmin, ymax],
        "glb_axes": "x=east, y=up, z=south",
        "shared_z_origin": None,
        "acquisition_selection": (
            "Oldest indexed year owns its coverage; "
            "same-year files may overlap"
        ),
        "download_manifest": str(
            input_dir / "download_manifest.csv"
        ),
        "coverage_index": str(index_path),
        "tiles": [],
    }

    z_origin = None
    manifest_path = output_dir / "tile_manifest.json"

    for row in range(
        0, total_rows, cells_per_tile
    ):
        for col in range(
            0, total_cols, cells_per_tile
        ):
            width = min(
                cells_per_tile, total_cols - col
            )
            height = min(
                cells_per_tile, total_rows - row
            )

            left = xmin + col * args.resolution
            top = ymax - row * args.resolution
            right = left + width * args.resolution
            bottom = top - height * args.resolution

            name = (
                f"tile_r{row // cells_per_tile:05d}"
                f"_c{col // cells_per_tile:05d}"
            )

            print(f"Processing {name}", flush=True)

            buffered = (
                left - args.overlap,
                bottom - args.overlap,
                right + args.overlap,
                top + args.overlap,
            )

            points = read_ground(
                sources,
                source_tree,
                buffered,
                args.chunk_size,
            )

            record = {
                "tile": name,
                "core_bounds": [
                    left, bottom, right, top
                ],
                "ground_points": int(len(points)),
                "dem": None,
                "glb": None,
            }

            try:
                tin_data = build_tin(
                    points, args.max_edge
                )
            except (ValueError, QhullError) as error:
                record["status"] = (
                    "insufficient_ground_geometry"
                )
                record["message"] = (
                    str(error).splitlines()[0]
                )
                manifest["tiles"].append(record)
                save_manifest(manifest_path, manifest)

                print(
                    "  Skipped: insufficient "
                    "ground geometry."
                )
                continue

            x_centers = left + (
                np.arange(width) + 0.5
            ) * args.resolution

            y_centers = top - (
                np.arange(height) + 0.5
            ) * args.resolution

            dem = interpolate(
                tin_data, x_centers, y_centers
            )

            if not np.any(np.isfinite(dem)):
                record["status"] = "no_valid_dem"
                manifest["tiles"].append(record)
                save_manifest(manifest_path, manifest)

                print(
                    "  Skipped: no valid DEM cells."
                )
                continue

            dem_path = dem_dir / f"{name}.tif"

            write_dem(
                dem_path,
                dem,
                left,
                top,
                args.resolution,
                crs,
            )

            record["dem"] = (
                dem_path.relative_to(
                    output_dir
                ).as_posix()
            )

            record["valid_cell_fraction"] = float(
                np.isfinite(dem).mean()
            )

            if not args.no_glb:
                x_edges = left + (
                    np.arange(width + 1)
                    * args.resolution
                )
                y_edges = top - (
                    np.arange(height + 1)
                    * args.resolution
                )

                mesh_z = interpolate(
                    tin_data, x_edges, y_edges
                )

                if z_origin is None:
                    z_origin = float(
                        np.floor(np.nanmin(dem))
                    )
                    manifest["shared_z_origin"] = (
                        z_origin
                    )

                glb_path = glb_dir / f"{name}.glb"

                if write_glb(
                    glb_path,
                    x_edges,
                    y_edges,
                    mesh_z,
                    z_origin,
                ):
                    record["glb"] = (
                        glb_path.relative_to(
                            output_dir
                        ).as_posix()
                    )
                    record[
                        "glb_origin_easting_northing_elevation"
                    ] = [
                        left, top, z_origin
                    ]

            record["status"] = "generated"
            manifest["tiles"].append(record)
            save_manifest(manifest_path, manifest)

    save_manifest(manifest_path, manifest)
    print(f"Finished: {output_dir}")


if __name__ == "__main__":
    main()
