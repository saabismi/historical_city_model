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
surface. Existing terrain is never raised.

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
from collections import defaultdict
from pathlib import Path

import numpy as np


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


def deform_tile(
    source,
    destination,
    triangle_index,
    clearance,
    edge_buffer,
    index_cell_size,
):
    """Copy one TRN1 tile while lowering terrain under road triangles."""
    changed = 0
    all_values = []

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

        for _ in range(levels):
            columns, rows = struct.unpack("<2I", read_exact(
                input_file, 8, "TRN1 level dimensions"
            ))
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

            xs, ys = terrain_vertex_positions(east, south, xmin, ymax)

            if triangle_index:
                for row in range(rows + 2):
                    for column in range(columns + 2):
                        point = np.array([xs[row, column], ys[row, column]])
                        current = float(heights[row, column])
                        target = None

                        for triangle in indexed_triangles(
                            triangle_index,
                            point,
                            edge_buffer,
                            index_cell_size,
                        ):
                            minimum_x = triangle[:, 0].min() - edge_buffer
                            maximum_x = triangle[:, 0].max() + edge_buffer
                            minimum_y = triangle[:, 1].min() - edge_buffer
                            maximum_y = triangle[:, 1].max() + edge_buffer

                            if not (
                                minimum_x <= point[0] <= maximum_x
                                and minimum_y <= point[1] <= maximum_y
                            ):
                                continue

                            elevation = triangle_elevation(
                                point, triangle, edge_buffer
                            )
                            if elevation is None:
                                continue

                            candidate = elevation - clearance
                            target = candidate if target is None else min(
                                target, candidate
                            )

                        if target is not None and math.isfinite(current):
                            if target < current:
                                heights[row, column] = target
                                changed += 1

            all_values.append(heights[np.isfinite(heights)])

            output.write(struct.pack("<2I", columns, rows))
            output.write(east.astype("<f4").tobytes())
            output.write(south.astype("<f4").tobytes())
            output.write(heights.astype("<f4").tobytes())

    values = np.concatenate(all_values) if all_values else np.empty(0)
    elevation_bounds = (
        float(values.min()) if len(values) else 0.0,
        float(values.max()) if len(values) else 0.0,
    )

    with destination.open("r+b") as output:
        output.seek(40)
        output.write(struct.pack("<2f", *elevation_bounds))

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
        help="XY size of the internal road-triangle spatial-index buckets.",
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

    for tile in terrain_tiles:
        source = terrain / tile["terrain_file"]
        destination = output / tile["terrain_file"]
        destination.parent.mkdir(parents=True, exist_ok=True)

        road_file = road_files.get(Path(tile["terrain_file"]).name.lower())
        triangle_index = {}
        if road_file is not None:
            if not road_file.is_file():
                raise ValueError(f"Missing indexed ROAD1 file: {road_file}")
            triangles = read_road_triangles(road_file)
            triangle_index = build_triangle_index(
                triangles,
                args.index_cell_size,
            )

        changed, elevation_bounds = deform_tile(
            source,
            destination,
            triangle_index,
            args.clearance,
            args.edge_buffer,
            args.index_cell_size,
        )
        total_changed += changed
        if triangle_index:
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
        f"{tiles_with_roads} road tiles."
    )
    print(f"Output: {output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
