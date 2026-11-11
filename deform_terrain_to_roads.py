#!/usr/bin/env python3
"""
Lower prepared TRN1 terrain beneath already-generated ROAD1 meshes.

This is a post-processing step for the legacy viewer pipeline. It does not
modify the input terrain or road folders. Instead, it writes a complete,
viewer-compatible terrain folder to a new location, preserving the existing
terrain tile names and format.

Usage:
    python deform_terrain_to_roads.py ^
        --terrain Historyviewer\\espoo2008_terrain_quick ^
        --roads Historyviewer\\espoo2008_roads ^
        -o Historyviewer\\espoo2008_terrain_roads

The ROAD1 geometry is used as the source of truth for road widths, shoulders,
slopes, intersections, and elevation. Terrain vertices under a road, or
within the configurable edge buffer, are lowered to just below the road
surface. Existing terrain is never raised. Road triangles are rasterized to
terrain cells in compiled rasterio code, so the expensive Python
cell-by-triangle search is avoided.

Outputs:
    terrain_index.txt
    tiles/*.trn
    metadata.json (when present in the input terrain folder)
    terrain_road_deformation.json

The output can be supplied to the current viewer in place of the original
terrain folder. No C++ changes or road regeneration are required.
"""

import argparse
import json
import math
import shutil
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import Polygon, mapping


TRN_MAGIC = b"TRN1"
ROAD_MAGIC = b"ROAD1"


def read_terrain_index(folder):
    """Read the viewer terrain index and preserve its tile ordering."""
    path = folder / "terrain_index.txt"
    with path.open("r", encoding="utf-8") as handle:
        if handle.readline().strip() != "TERRAIN_INDEX_V1":
            raise ValueError(f"Invalid terrain index: {path}")

        origin = handle.readline().strip()
        count = int(handle.readline().strip())
        tiles = []

        for _ in range(count):
            parts = handle.readline().split()
            if len(parts) != 7:
                raise ValueError(f"Invalid terrain tile record in {path}")

            tiles.append({
                "terrain_file": parts[0],
                "bounds": tuple(float(value) for value in parts[1:5]),
                "zmin": float(parts[5]),
                "zmax": float(parts[6]),
            })

    return origin, tiles


def read_road_index(folder):
    """Map terrain tile basenames to their ROAD1 files."""
    path = folder / "road_index.txt"
    with path.open("r", encoding="utf-8") as handle:
        if handle.readline().strip() != "ROAD_INDEX_V1":
            raise ValueError(f"Invalid road index: {path}")

        count = int(handle.readline().strip())
        result = {}

        for _ in range(count):
            terrain_name = Path(handle.readline().strip()).name.lower()
            road_file = handle.readline().strip()
            result[terrain_name] = folder / road_file

    return result


def read_exact(handle, size, description):
    """Read an exact byte count from a binary file."""
    data = handle.read(size)
    if len(data) != size:
        raise ValueError(f"Truncated {description}.")
    return data


def read_road_triangles(path):
    """
    Read ROAD1 groups as absolute XY triangles with absolute elevations.

    The ROAD1 local axes are X=east, Y=up, Z=south. The road origin stores
    absolute easting, northing, and elevation.
    """
    triangles = []

    with path.open("rb") as handle:
        if read_exact(handle, 5, "ROAD1 header") != ROAD_MAGIC:
            raise ValueError(f"Invalid ROAD1 file: {path}")

        road_count = struct.unpack("<I", read_exact(
            handle, 4, "ROAD1 road count"
        ))[0]

        for _ in range(road_count):
            identifier_length = struct.unpack("<H", read_exact(
                handle, 2, "ROAD1 identifier length"
            ))[0]
            read_exact(handle, identifier_length, "ROAD1 identifier")

            origin_x, origin_y, origin_z = struct.unpack(
                "<3d", read_exact(handle, 24, "ROAD1 origin")
            )
            read_exact(handle, 24, "ROAD1 bounds")
            read_exact(handle, 8, "ROAD1 road attributes")

            group_count = struct.unpack("<I", read_exact(
                handle, 4, "ROAD1 group count"
            ))[0]

            for _ in range(group_count):
                category, material, vertex_count = struct.unpack(
                    "<3I", read_exact(handle, 12, "ROAD1 group header")
                )
                del category, material

                raw = np.frombuffer(
                    read_exact(
                        handle,
                        vertex_count * 32,
                        "ROAD1 vertex data",
                    ),
                    dtype="<f4",
                ).reshape((-1, 8))

                if vertex_count % 3:
                    raise ValueError(f"Invalid triangle group in {path}")

                for triangle in raw.reshape((-1, 3, 8)):
                    points = np.column_stack((
                        origin_x + triangle[:, 0],
                        origin_y - triangle[:, 2],
                        origin_z + triangle[:, 1],
                    ))
                    triangles.append(points)

    return triangles


def point_segment_distance_and_z(point, first, second):
    """Return distance and interpolated Z to the closest segment point."""
    vector = second[:2] - first[:2]
    length_squared = float(np.dot(vector, vector))

    if length_squared <= 1e-20:
        parameter = 0.0
    else:
        parameter = float(np.dot(point - first[:2], vector) / length_squared)
        parameter = max(0.0, min(1.0, parameter))

    closest = first[:2] + parameter * vector
    z = first[2] + parameter * (second[2] - first[2])
    return float(np.linalg.norm(point - closest)), float(z)


def triangle_elevation(point, triangle, edge_buffer):
    """
    Return interpolated road elevation when point is on/near a triangle.

    The edge buffer ensures narrow roads are still covered by coarse terrain
    LODs whose vertices do not land exactly inside the road polygon.
    """
    a, b, c = triangle
    v0 = b[:2] - a[:2]
    v1 = c[:2] - a[:2]
    v2 = point - a[:2]

    denominator = (
        v0[0] * v1[1] - v1[0] * v0[1]
    )

    if abs(denominator) > 1e-12:
        u = (
            v2[0] * v1[1] - v1[0] * v2[1]
        ) / denominator
        v = (
            v0[0] * v2[1] - v2[0] * v0[1]
        ) / denominator
        if u >= 0.0 and v >= 0.0 and u + v <= 1.0:
            return float(a[2] + u * (b[2] - a[2]) + v * (c[2] - a[2]))

    if edge_buffer <= 0.0:
        return None

    candidates = [
        point_segment_distance_and_z(point, a, b),
        point_segment_distance_and_z(point, b, c),
        point_segment_distance_and_z(point, c, a),
    ]
    distance, elevation = min(candidates, key=lambda item: item[0])
    return elevation if distance <= edge_buffer else None


def terrain_vertex_positions(east, south, xmin, ymax):
    """Return absolute XY coordinates for core and one-cell halo vertices."""
    east_spacing = east[1] - east[0] if len(east) > 1 else 0.0
    south_spacing = south[1] - south[0] if len(south) > 1 else 0.0

    all_east = np.concatenate((
        [east[0] - east_spacing],
        east,
        [east[-1] + east_spacing],
    ))
    all_south = np.concatenate((
        [south[0] - south_spacing],
        south,
        [south[-1] + south_spacing],
    ))

    return np.meshgrid(
        xmin + all_east,
        ymax - all_south,
    )


def build_triangle_index(triangles, cell_size):
    """Build a coarse XY bucket index for fast terrain-vertex queries."""
    index = defaultdict(list)

    for triangle in triangles:
        minimum_x = float(triangle[:, 0].min())
        maximum_x = float(triangle[:, 0].max())
        minimum_y = float(triangle[:, 1].min())
        maximum_y = float(triangle[:, 1].max())

        first_x = math.floor(minimum_x / cell_size)
        last_x = math.floor(maximum_x / cell_size)
        first_y = math.floor(minimum_y / cell_size)
        last_y = math.floor(maximum_y / cell_size)

        for bucket_x in range(first_x, last_x + 1):
            for bucket_y in range(first_y, last_y + 1):
                index[(bucket_x, bucket_y)].append(triangle)

    return index


def indexed_triangles(index, point, edge_buffer, cell_size):
    """Yield triangles whose indexed buckets could contain the point."""
    first_x = math.floor((point[0] - edge_buffer) / cell_size)
    last_x = math.floor((point[0] + edge_buffer) / cell_size)
    first_y = math.floor((point[1] - edge_buffer) / cell_size)
    last_y = math.floor((point[1] + edge_buffer) / cell_size)
    seen = set()

    for bucket_x in range(first_x, last_x + 1):
        for bucket_y in range(first_y, last_y + 1):
            for triangle in index.get((bucket_x, bucket_y), ()):
                identifier = id(triangle)
                if identifier not in seen:
                    seen.add(identifier)
                    yield triangle


def point_in_triangle(point, triangle):
    """Return whether an XY point is inside or on a triangle."""
    def cross(first, second):
        return first[0] * second[1] - first[1] * second[0]

    a, b, c = triangle[:, :2]
    ab = b - a
    bc = c - b
    ca = a - c
    ap = point - a
    bp = point - b
    cp = point - c
    cross_values = (
        cross(ab, ap),
        cross(bc, bp),
        cross(ca, cp),
    )
    return all(value >= -1e-9 for value in cross_values) or all(
        value <= 1e-9 for value in cross_values
    )


def segments_intersect(first, second, third, fourth):
    """Return whether two closed XY line segments intersect."""
    def orientation(a, b, c):
        vector_ab = b - a
        vector_ac = c - a
        value = vector_ab[0] * vector_ac[1] - vector_ab[1] * vector_ac[0]
        if abs(value) <= 1e-9:
            return 0
        return 1 if value > 0 else -1

    def on_segment(a, b, point):
        return (
            min(a[0], b[0]) - 1e-9 <= point[0] <= max(a[0], b[0]) + 1e-9
            and min(a[1], b[1]) - 1e-9 <= point[1] <= max(a[1], b[1]) + 1e-9
        )

    first = first[:2]
    second = second[:2]
    third = third[:2]
    fourth = fourth[:2]
    orientations = (
        orientation(first, second, third),
        orientation(first, second, fourth),
        orientation(third, fourth, first),
        orientation(third, fourth, second),
    )

    if orientations[0] != orientations[1] and orientations[2] != orientations[3]:
        return True

    return (
        orientations[0] == 0 and on_segment(first, second, third)
    ) or (
        orientations[1] == 0 and on_segment(first, second, fourth)
    ) or (
        orientations[2] == 0 and on_segment(third, fourth, first)
    ) or (
        orientations[3] == 0 and on_segment(third, fourth, second)
    )


def triangle_intersects_rectangle(triangle, minimum, maximum):
    """Return whether an XY triangle intersects an axis-aligned rectangle."""
    rectangle = np.asarray([
        [minimum[0], minimum[1]],
        [maximum[0], minimum[1]],
        [maximum[0], maximum[1]],
        [minimum[0], maximum[1]],
    ], dtype=float)
    triangle_xy = triangle[:, :2]

    if any(point_in_triangle(point, triangle) for point in rectangle):
        return True

    if (
        minimum[0] <= triangle_xy[:, 0].min()
        and triangle_xy[:, 0].max() <= maximum[0]
        and minimum[1] <= triangle_xy[:, 1].min()
        and triangle_xy[:, 1].max() <= maximum[1]
    ):
        return True

    for index in range(3):
        first = triangle_xy[index]
        second = triangle_xy[(index + 1) % 3]
        for rectangle_index in range(4):
            third = rectangle[rectangle_index]
            fourth = rectangle[(rectangle_index + 1) % 4]
            if segments_intersect(first, second, third, fourth):
                return True

    return False


def deform_tile(
    source,
    destination,
    triangles,
    clearance,
    edge_buffer,
    index_cell_size,
    tile_number,
    tile_count,
):
    """Copy one TRN1 tile while lowering terrain under road triangles.

    ROAD1 triangles are converted to polygons and rasterized once per LOD.
    This keeps the expensive spatial work in rasterio/GDAL-backed code rather
    than checking every terrain vertex against every road triangle in Python.
    """
    changed = 0
    all_values = []
    started = time.monotonic()
    last_report = started

    with source.open("rb") as input_file, destination.open("wb") as output:
        if read_exact(input_file, 4, "TRN1 header") != TRN_MAGIC:
            raise ValueError(f"Invalid terrain tile: {source}")

        levels = struct.unpack("<I", read_exact(
            input_file, 4, "TRN1 level count"
        ))[0]
        bounds = struct.unpack("<4d", read_exact(
            input_file, 32, "TRN1 bounds"
        ))
        read_exact(input_file, 8, "TRN1 elevation bounds")

        output.write(TRN_MAGIC)
        output.write(struct.pack("<I", levels))
        output.write(struct.pack("<4d", *bounds))
        output.write(b"\x00" * 8)

        xmin, ymin, xmax, ymax = bounds

        for level in range(levels):
            columns, rows = struct.unpack("<2I", read_exact(
                input_file, 8, "TRN1 level dimensions"
            ))
            print(
                f"Tile {tile_number}/{tile_count}: LOD {level + 1}/{levels} "
                f"({columns}x{rows}) starting",
                flush=True,
            )
            east = np.frombuffer(
                read_exact(input_file, columns * 4, "TRN1 east coordinates"),
                dtype="<f4",
            ).copy()
            south = np.frombuffer(
                read_exact(input_file, rows * 4, "TRN1 south coordinates"),
                dtype="<f4",
            ).copy()
            heights = np.frombuffer(
                read_exact(
                    input_file,
                    (columns + 2) * (rows + 2) * 4,
                    "TRN1 elevations",
                ),
                dtype="<f4",
            ).copy().reshape((rows + 2, columns + 2))

            if triangles:
                # Rasterize all road triangles in compiled code. This avoids
                # the previous terrain-cell x road-triangle Python loops.
                east_spacing = (
                    float(east[1] - east[0]) if len(east) > 1 else 0.0
                )
                south_spacing = (
                    float(south[1] - south[0]) if len(south) > 1 else 0.0
                )
                shapes = []
                for triangle in triangles:
                    polygon = Polygon(triangle[:, :2])
                    if polygon.is_empty or polygon.area <= 1e-9:
                        continue
                    if edge_buffer > 0:
                        polygon = polygon.buffer(edge_buffer)
                    shapes.append((mapping(polygon), float(triangle[:, 2].min())))

                if shapes and east_spacing > 0 and south_spacing > 0:
                    road_heights = rasterize(
                        shapes,
                        out_shape=(rows + 1, columns + 1),
                        transform=from_origin(
                            xmin + east[0] - east_spacing,
                            ymax - south[0] + south_spacing,
                            east_spacing,
                            south_spacing,
                        ),
                        fill=np.nan,
                        dtype="float32",
                        all_touched=True,
                    )

                    for row, column in zip(*np.nonzero(np.isfinite(road_heights))):
                        target = float(road_heights[row, column]) - clearance
                        for vertex_row, vertex_column in (
                            (row, column),
                            (row + 1, column),
                            (row, column + 1),
                            (row + 1, column + 1),
                        ):
                            current = float(heights[vertex_row, vertex_column])
                            if math.isfinite(current) and target < current:
                                heights[vertex_row, vertex_column] = target
                                changed += 1

            all_values.append(heights[np.isfinite(heights)])

            output.write(struct.pack("<2I", columns, rows))
            output.write(east.astype("<f4").tobytes())
            output.write(south.astype("<f4").tobytes())
            output.write(heights.astype("<f4").tobytes())
            print(
                f"Tile {tile_number}/{tile_count}: LOD {level + 1}/{levels} "
                f"done, changed vertices {changed}, elapsed "
                f"{time.monotonic() - started:.1f}s",
                flush=True,
            )

    values = np.concatenate(all_values) if all_values else np.empty(0)
    elevation_bounds = (
        float(values.min()) if len(values) else 0.0,
        float(values.max()) if len(values) else 0.0,
    )

    with destination.open("r+b") as output:
        output.seek(40)
        output.write(struct.pack("<2f", *elevation_bounds))

    print(
        f"Tile {tile_number}/{tile_count}: finished, changed vertices "
        f"{changed}, elapsed {time.monotonic() - started:.1f}s",
        flush=True,
    )
    return changed, elevation_bounds


def main():
    parser = argparse.ArgumentParser(
        description="Lower TRN1 terrain beneath generated ROAD1 geometry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--terrain", required=True)
    parser.add_argument("--roads", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument(
        "--clearance",
        type=float,
        default=0.03,
        help="Keep terrain this many metres below the road mesh.",
    )
    parser.add_argument(
        "--edge-buffer",
        type=float,
        default=0.25,
        help="Cover nearby terrain vertices around narrow road edges.",
    )
    parser.add_argument(
        "--index-cell-size",
        type=float,
        default=32.0,
        help=(
            "Retained for command-line compatibility; rasterization no "
            "longer uses Python spatial buckets."
        ),
    )
    args = parser.parse_args()

    terrain = Path(args.terrain).resolve()
    roads = Path(args.roads).resolve()
    output = Path(args.output).resolve()

    if (
        args.clearance < 0
        or args.edge_buffer < 0
        or args.index_cell_size <= 0
    ):
        parser.error(
            "--clearance and --edge-buffer must be non-negative; "
            "--index-cell-size must be positive."
        )
    if not (terrain / "terrain_index.txt").is_file():
        parser.error(f"Missing terrain_index.txt: {terrain}")
    if not (roads / "road_index.txt").is_file():
        parser.error(f"Missing road_index.txt: {roads}")
    if output.exists() and (
        not output.is_dir() or any(output.iterdir())
    ):
        parser.error("Output folder must be empty or not yet exist.")

    origin, terrain_tiles = read_terrain_index(terrain)
    road_files = read_road_index(roads)
    output.mkdir(parents=True, exist_ok=True)
    (output / "tiles").mkdir()

    for filename in ("metadata.json",):
        source = terrain / filename
        if source.is_file():
            shutil.copy2(source, output / filename)

    index_entries = []
    total_changed = 0
    tiles_with_roads = 0
    started = time.monotonic()
    print(
        f"Preparing {len(terrain_tiles)} terrain tiles; "
        f"{len(road_files)} road index entries.",
        flush=True,
    )

    for tile_number, tile in enumerate(terrain_tiles, 1):
        source = terrain / tile["terrain_file"]
        destination = output / tile["terrain_file"]
        destination.parent.mkdir(parents=True, exist_ok=True)

        road_file = road_files.get(Path(tile["terrain_file"]).name.lower())
        triangles = []
        if road_file is not None:
            if not road_file.is_file():
                raise ValueError(f"Missing indexed ROAD1 file: {road_file}")
            print(
                f"Tile {tile_number}/{len(terrain_tiles)}: reading roads "
                f"{road_file.name}",
                flush=True,
            )
            triangles = read_road_triangles(road_file)
            print(
                f"Tile {tile_number}/{len(terrain_tiles)}: loaded "
                f"{len(triangles)} road triangles.",
                flush=True,
            )
        else:
            print(
                f"Tile {tile_number}/{len(terrain_tiles)}: no road file; "
                "copying terrain unchanged.",
                flush=True,
            )

        changed, elevation_bounds = deform_tile(
            source,
            destination,
            triangles,
            args.clearance,
            args.edge_buffer,
            args.index_cell_size,
            tile_number,
            len(terrain_tiles),
        )
        total_changed += changed
        if triangles:
            tiles_with_roads += 1

        index_entries.append((
            tile["terrain_file"],
            tile["bounds"],
            elevation_bounds,
        ))

    with (output / "terrain_index.txt").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write("TERRAIN_INDEX_V1\n")
        handle.write(origin + "\n")
        handle.write(f"{len(index_entries)}\n")
        for filename, bounds, elevation_bounds in index_entries:
            handle.write(
                f"{filename} "
                f"{bounds[0]:.9f} {bounds[1]:.9f} "
                f"{bounds[2]:.9f} {bounds[3]:.9f} "
                f"{elevation_bounds[0]:.6f} {elevation_bounds[1]:.6f}\n"
            )

    (output / "terrain_road_deformation.json").write_text(
        json.dumps({
            "format": "terrain_road_deformation_v1",
            "source_terrain": str(terrain),
            "source_roads": str(roads),
            "clearance_metres": args.clearance,
            "edge_buffer_metres": args.edge_buffer,
            "tiles_with_roads": tiles_with_roads,
            "changed_vertices": total_changed,
            "notes": [
                "Input terrain tiles were not modified.",
                "ROAD1 geometry is the source of truth for road placement.",
                "Terrain is lowered only; it is never raised.",
                "The output remains compatible with TERRAIN_INDEX_V1/TRN1.",
            ],
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"Finished: lowered {total_changed} terrain vertices in "
        f"{tiles_with_roads} road tiles; total elapsed "
        f"{time.monotonic() - started:.1f}s.",
        flush=True,
    )
    print(f"Output: {output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
