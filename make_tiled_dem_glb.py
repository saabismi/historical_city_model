#!/usr/bin/env python3
"""
Generate tiled ground DEMs and optional GLB terrain meshes.

Examples:
    python make_tiled_dem.py -i pointclouds -o terrain --crs EPSG:3067

    python make_tiled_dem.py -i pointclouds -o terrain \
        --crs EPSG:3067 --overlap 30 --no-glb

    python make_tiled_dem.py -i pointclouds -o terrain \
        --reference aerial.tif --crs EPSG:3067

Arguments:
    -i, --input-dir   Folder containing LAS/LAZ files.
    -o, --output-dir  Output folder; must be empty or not yet exist.
    --crs            Input/output projected CRS; default EPSG:3067.
    --reference      Optional raster defining the full output extent.
                      Must have the same CRS and be north-up.
    --tile-size      Core size in metres; default 250.
    --resolution     DEM cell size in metres; default 1.
    --overlap        Processing buffer on EACH side; default 20 m.
    --max-edge       Reject triangles with longer edges; default 50 m.
                      Set 0 to disable this gap protection.
    --no-glb         Generate DEMs only.
    --chunk-size     Points read per chunk; default 1,000,000.

Outputs:
    dem/*.tif
    glb/*.glb                  (unless --no-glb)
    tile_manifest.json

Uses ground classification 2, excludes withheld points, and interpolates
a Delaunay TIN. Does not enforce oldest-acquisition ownership.

GLB coordinates:
    x = easting relative to tile origin
    y = elevation relative to a shared elevation origin
    z = negative northing relative to tile origin
The manifest records placement and CRS information.
"""

import argparse
import json
import math
from pathlib import Path

import laspy
import numpy as np
import rasterio
import trimesh
from pyproj import CRS
from rasterio.transform import from_origin
from scipy.spatial import Delaunay, QhullError


NODATA = -9999.0


def intersects(a, b):
    """Check whether two XY bounding boxes intersect."""
    return (
        a[0] < b[2] and a[2] > b[0]
        and a[1] < b[3] and a[3] > b[1]
    )


def inspect_sources(folder, expected_crs):
    """Read file headers for bounding boxes and any available CRS metadata."""
    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in {".las", ".laz"}
    )
    if not files:
        raise ValueError("No LAS/LAZ files found in the input folder.")

    sources = []
    missing_crs = 0

    for filename in files:
        with laspy.open(filename) as reader:
            header = reader.header
            source_crs = header.parse_crs()

            # Reject known CRS mismatches; missing metadata is handled below.
            if source_crs is not None:
                if not CRS(source_crs).equals(
                    expected_crs, ignore_axis_order=True
                ):
                    raise ValueError(f"CRS mismatch: {filename}")
            else:
                missing_crs += 1

            if header.point_count == 0:
                continue

            sources.append({
                "path": filename,
                "bounds": (
                    float(header.mins[0]),
                    float(header.mins[1]),
                    float(header.maxs[0]),
                    float(header.maxs[1]),
                ),
            })

    if not sources:
        raise ValueError("All source files are empty.")

    if missing_crs:
        print(
            f"WARNING: {missing_crs} file(s) lack CRS metadata. "
            "Assuming they use --crs; verify this yourself."
        )

    return sources


def get_extent(sources, reference, crs):
    """Use the reference raster's full extent, or the union of LAS headers."""
    if reference:
        with rasterio.open(reference) as raster:
            if raster.crs is None:
                raise ValueError("Reference raster has no CRS.")

            if not CRS(raster.crs).equals(crs, ignore_axis_order=True):
                raise ValueError("Reference raster must use the input CRS.")

            transform = raster.transform
            if (
                transform.b != 0 or transform.d != 0
                or transform.a <= 0 or transform.e >= 0
            ):
                raise ValueError("Reference raster must be north-up.")

            bounds = raster.bounds
            return bounds.left, bounds.bottom, bounds.right, bounds.top

    return (
        min(s["bounds"][0] for s in sources),
        min(s["bounds"][1] for s in sources),
        max(s["bounds"][2] for s in sources),
        max(s["bounds"][3] for s in sources),
    )


def read_ground(sources, bounds, chunk_size):
    """Read class-2 points inside a buffered tile, using bounded-size chunks."""
    arrays = []

    for source in sources:
        if not intersects(source["bounds"], bounds):
            continue

        with laspy.open(source["path"]) as reader:
            for points in reader.chunk_iterator(chunk_size):
                x = np.asarray(points.x)
                y = np.asarray(points.y)
                z = np.asarray(points.z)

                # Include boundary points and exclude withheld observations.
                keep = (
                    (np.asarray(points.classification) == 2)
                    & ~np.asarray(points.withheld, dtype=bool)
                    & (x >= bounds[0]) & (x <= bounds[2])
                    & (y >= bounds[1]) & (y <= bounds[3])
                    & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
                )

                if np.any(keep):
                    arrays.append(
                        np.column_stack((x[keep], y[keep], z[keep]))
                    )

    if not arrays:
        return np.empty((0, 3), dtype=np.float64)

    return np.concatenate(arrays)


def build_tin(points, max_edge):
    """Create a TIN; average duplicate XY observations and flag large triangles."""
    # Delaunay requires unique XY positions. Averaging duplicates avoids
    # arbitrarily choosing one Z, but is not an acquisition-priority rule.
    xy, inverse = np.unique(
        points[:, :2], axis=0, return_inverse=True
    )
    sums = np.bincount(inverse, weights=points[:, 2])
    counts = np.bincount(inverse)
    elevations = sums / counts

    if len(xy) < 3:
        raise ValueError("Fewer than three unique ground-point positions.")

    # Shift coordinates locally to improve numerical conditioning.
    origin = xy.min(axis=0)
    local_xy = xy - origin
    tin = Delaunay(local_xy)

    valid_triangles = np.ones(len(tin.simplices), dtype=bool)

    # Prevent interpolation across excessively large gaps.
    if max_edge > 0:
        vertices = local_xy[tin.simplices]
        edge_lengths = np.stack([
            np.linalg.norm(vertices[:, 0] - vertices[:, 1], axis=1),
            np.linalg.norm(vertices[:, 1] - vertices[:, 2], axis=1),
            np.linalg.norm(vertices[:, 2] - vertices[:, 0], axis=1),
        ], axis=1)
        valid_triangles = edge_lengths.max(axis=1) <= max_edge

    return tin, elevations, origin, valid_triangles


def interpolate(tin_data, x, y):
    """Interpolate a regular XY grid using triangle barycentric coordinates."""
    tin, elevations, origin, valid_triangles = tin_data

    xx, yy = np.meshgrid(x, y)
    query = np.column_stack((xx.ravel(), yy.ravel())) - origin

    simplex = tin.find_simplex(query)
    result = np.full(len(query), np.nan, dtype=np.float64)

    # Outside the convex hull, or inside rejected triangles, leave NoData.
    candidates = np.flatnonzero(simplex >= 0)
    usable = candidates[valid_triangles[simplex[candidates]]]

    if len(usable):
        triangle_ids = simplex[usable]
        transforms = tin.transform[triangle_ids]

        delta = query[usable] - transforms[:, 2, :]
        first_weights = np.einsum(
            "ijk,ik->ij", transforms[:, :2, :], delta
        )
        weights = np.column_stack((
            first_weights,
            1.0 - first_weights.sum(axis=1),
        ))

        triangle_z = elevations[tin.simplices[triangle_ids]]
        result[usable] = np.sum(weights * triangle_z, axis=1)

    return result.reshape(len(y), len(x))


def write_dem(filename, elevations, xmin, ymax, resolution, crs):
    """Write core-only DEM cell-center elevations as a georeferenced TIFF."""
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
        transform=from_origin(xmin, ymax, resolution, resolution),
        nodata=NODATA,
        tiled=True,
        compress="deflate",
        BIGTIFF="IF_SAFER",
    ) as output:
        output.write(data, 1)


def write_glb(filename, x, y, elevations, z_origin):
    """Make a y-up mesh with boundary vertices; omit faces crossing NoData."""
    xx, yy = np.meshgrid(x, y)

    # Use tile-local coordinates to avoid large-coordinate precision loss.
    vertices = np.column_stack((
        (xx - x[0]).ravel(),
        (elevations - z_origin).ravel(),
        (y[0] - yy).ravel(),
    ))

    rows, cols = elevations.shape
    indexes = np.arange(rows * cols).reshape(rows, cols)

    # Two upward-facing triangles per grid cell.
    a = indexes[:-1, :-1].ravel()
    b = indexes[:-1, 1:].ravel()
    c = indexes[1:, :-1].ravel()
    d = indexes[1:, 1:].ravel()

    faces = np.concatenate((
        np.column_stack((a, c, b)),
        np.column_stack((b, c, d)),
    ))

    # Do not bridge missing terrain elevations.
    finite = np.isfinite(elevations.ravel())
    faces = faces[np.all(finite[faces], axis=1)]

    if len(faces) == 0:
        return False

    # Remove unused vertices, including all non-finite ones.
    used, inverse = np.unique(faces, return_inverse=True)
    mesh = trimesh.Trimesh(
        vertices=vertices[used],
        faces=inverse.reshape(-1, 3),
        process=False,
    )

    # Neutral color; semantic textures can be added later.
    mesh.visual.vertex_colors = [180, 180, 180, 255]
    mesh.export(str(filename), file_type="glb")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Create buffered TIN DEM tiles and optional GLB meshes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input-dir", required=True)
    parser.add_argument("-o", "--output-dir", required=True)
        
    parser.add_argument(# Default to ETRS-TM35FIN, Finland's metre-based projected CRS.
        "--crs",
        default="EPSG:3067",
        help="Input/output projected CRS.",
    )

    parser.add_argument("--reference", help="Raster defining output extent.")
    parser.add_argument("--tile-size", type=float, default=250)
    parser.add_argument("--resolution", type=float, default=1)
    parser.add_argument("--overlap", type=float, default=20)
    parser.add_argument("--max-edge", type=float, default=50)
    parser.add_argument("--no-glb", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    args = parser.parse_args()

    if args.tile_size <= 0 or args.resolution <= 0:
        parser.error("Tile size and resolution must be positive.")
    if args.overlap < 0 or args.max_edge < 0:
        parser.error("Overlap and maximum edge length cannot be negative.")
    if args.chunk_size <= 0:
        parser.error("Chunk size must be positive.")

    # Require whole cells per core, avoiding inconsistent tile boundaries.
    cells_float = args.tile_size / args.resolution
    cells = round(cells_float)
    if cells < 1 or not math.isclose(cells_float, cells, abs_tol=1e-8):
        parser.error("Tile size must be an integer multiple of resolution.")

    crs = CRS.from_user_input(args.crs)
    if not crs.is_projected:
        parser.error("Use a projected CRS with metre units.")
    if any(
        not math.isclose(axis.unit_conversion_factor, 1.0)
        for axis in crs.axis_info[:2]
    ):
        parser.error("The CRS horizontal units must be metres.")

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        parser.error("Input folder does not exist.")

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            parser.error("Output folder must be empty or not yet exist.")

    dem_dir = output_dir / "dem"
    dem_dir.mkdir(parents=True, exist_ok=True)

    glb_dir = output_dir / "glb"
    if not args.no_glb:
        glb_dir.mkdir(parents=True, exist_ok=True)

    sources = inspect_sources(input_dir, crs)
    xmin, ymin, xmax, ymax = get_extent(
        sources, args.reference, crs
    )

    # Anchor one grid at the upper-left extent corner. Expand right/bottom
    # to complete cells; outermost tiles may have fewer than 'cells' cells.
    total_cols = math.ceil((xmax - xmin) / args.resolution)
    total_rows = math.ceil((ymax - ymin) / args.resolution)

    manifest = {
        "crs": crs.to_string(),
        "vertical_reference": "Inherited from inputs; not transformed",
        "resolution_metres": args.resolution,
        "tile_size_metres": args.tile_size,
        "processing_buffer_metres": args.overlap,
        "grid_origin": [xmin, ymax],
        "glb_axes": "x=east, y=up, z=south",
        "shared_z_origin": None,
        "tiles": [],
    }

    z_origin = None

    for row in range(0, total_rows, cells):
        for col in range(0, total_cols, cells):
            width = min(cells, total_cols - col)
            height = min(cells, total_rows - row)

            left = xmin + col * args.resolution
            top = ymax - row * args.resolution
            right = left + width * args.resolution
            bottom = top - height * args.resolution

            name = f"tile_r{row // cells:05d}_c{col // cells:05d}"
            print(f"Processing {name}")

            buffered = (
                left - args.overlap,
                bottom - args.overlap,
                right + args.overlap,
                top + args.overlap,
            )

            points = read_ground(sources, buffered, args.chunk_size)

            record = {
                "tile": name,
                "core_bounds": [left, bottom, right, top],
                "ground_points": len(points),
                "dem": None,
                "glb": None,
            }

            try:
                tin_data = build_tin(points, args.max_edge)
            except (ValueError, QhullError) as error:
                record["status"] = "insufficient_ground_geometry"
                record["message"] = str(error).splitlines()[0]
                manifest["tiles"].append(record)
                print("  Skipped: insufficient ground geometry.")
                continue

            # DEM samples represent cell centers.
            x_centers = left + (
                np.arange(width) + 0.5
            ) * args.resolution
            y_centers = top - (
                np.arange(height) + 0.5
            ) * args.resolution

            dem = interpolate(tin_data, x_centers, y_centers)

            if not np.any(np.isfinite(dem)):
                record["status"] = "no_valid_dem"
                manifest["tiles"].append(record)
                print("  Skipped: no valid interpolated DEM cells.")
                continue

            dem_path = dem_dir / f"{name}.tif"
            write_dem(
                dem_path, dem, left, top, args.resolution, crs
            )
            record["dem"] = dem_path.relative_to(output_dir).as_posix()
            record["valid_cell_fraction"] = float(
                np.isfinite(dem).mean()
            )

            if not args.no_glb:
                # Include both boundary edges: N cells require N+1 vertices.
                # Adjacent tiles therefore evaluate the same boundary XYs.
                x_edges = left + np.arange(width + 1) * args.resolution
                y_edges = top - np.arange(height + 1) * args.resolution
                mesh_z = interpolate(tin_data, x_edges, y_edges)

                if z_origin is None:
                    z_origin = float(np.floor(np.nanmin(dem)))
                    manifest["shared_z_origin"] = z_origin

                glb_path = glb_dir / f"{name}.glb"
                if write_glb(
                    glb_path, x_edges, y_edges, mesh_z, z_origin
                ):
                    record["glb"] = (
                        glb_path.relative_to(output_dir).as_posix()
                    )
                    record["glb_origin_easting_northing_elevation"] = [
                        left, top, z_origin
                    ]

            record["status"] = "generated"
            manifest["tiles"].append(record)

            # Save progress after each successfully generated tile.
            (output_dir / "tile_manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )

    # Also save records for trailing skipped tiles or an entirely empty area.
    (output_dir / "tile_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Finished: {output_dir}")


if __name__ == "__main__":
    main()
