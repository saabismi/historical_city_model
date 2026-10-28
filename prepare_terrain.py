#!/usr/bin/env python3
"""
Prepare compact terrain tiles for the legacy C++ viewer.

Usage:
    python prepare_terrain.py -i espoo_terrain -o viewer_terrain

Arguments:
    -i, --input-dir   Folder containing tile_manifest.json and DEM files.
    -o, --output-dir Output folder; must be empty or not yet exist.
    --levels         Number of detail levels; default 5.

Outputs:
    terrain_index.txt       Simple text index for the C++ viewer.
    tiles/*.trn             Little-endian binary elevation grids.

The original DEM files are not modified.
Requires: numpy, rasterio.
"""

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


# Binary format version. The viewer must validate this.
MAGIC = b"TRN1"

# Missing samples are represented by NaN in the binary grids.
# The viewer must omit triangles containing missing samples.


def sample_vertices(dataset, xs, ys):
    """
    Bilinearly sample DEM cell-center elevations at mesh vertex positions.

    Read only the raster window needed for this query. At the outer raster
    edge, clamp to its last available cell center. This means independently
    processed tiles can still have slightly different boundary elevations.
    """
    transform = dataset.transform

    # Convert map coordinates into continuous cell-center indexes.
    columns = (xs - transform.c) / transform.a - 0.5
    rows = (ys - transform.f) / transform.e - 0.5

    columns = np.clip(columns, 0, dataset.width - 1)
    rows = np.clip(rows, 0, dataset.height - 1)

    c0 = np.floor(columns).astype(int)
    r0 = np.floor(rows).astype(int)
    c1 = np.minimum(c0 + 1, dataset.width - 1)
    r1 = np.minimum(r0 + 1, dataset.height - 1)

    left = int(c0.min())
    top = int(r0.min())
    right = int(c1.max()) + 1
    bottom = int(r1.max()) + 1

    data = dataset.read(
        1,
        window=Window(left, top, right - left, bottom - top),
        masked=True,
    ).astype(np.float64).filled(np.nan)

    # Convert indexes into the local read window.
    c0 -= left
    c1 -= left
    r0 -= top
    r1 -= top

    fx = columns - np.floor(columns)
    fy = rows - np.floor(rows)

    # Meshgrid-compatible interpolation from four surrounding samples.
    a = data[np.ix_(r0, c0)]
    b = data[np.ix_(r0, c1)]
    c = data[np.ix_(r1, c0)]
    d = data[np.ix_(r1, c1)]

    weights = [
        (1 - fy[:, None]) * (1 - fx[None, :]),
        (1 - fy[:, None]) * fx[None, :],
        fy[:, None] * (1 - fx[None, :]),
        fy[:, None] * fx[None, :],
    ]

    result = np.zeros_like(a)
    valid = np.ones(a.shape, dtype=bool)

    # A missing neighbor matters only when its interpolation weight is
    # nonzero; this avoids NaN * 0 invalidating an exact sample.
    for values, weight in zip((a, b, c, d), weights):
        needed = weight > 1e-12
        valid &= ~needed | np.isfinite(values)
        result += np.where(np.isfinite(values), values, 0) * weight

    result[~valid] = np.nan
    return result.astype(np.float32)


def axis_vertices(cell_count, stride, resolution):
    """Include both tile boundaries, even if stride does not divide its size."""
    indexes = list(range(0, cell_count, stride)) + [cell_count]
    return np.asarray(indexes, dtype=np.float64) * resolution


def prepare_tile(dem_path, destination, levels):
    """
    Write one terrain file.

    Header:
        4 bytes magic: TRN1
        uint32 level count
        4 float64 values: xmin, ymin, xmax, ymax
        2 float32 values: minimum and maximum valid elevation

    For each level:
        uint32 vertex-column count, vertex-row count
        float32[column count] east offsets from xmin
        float32[row count] south offsets from ymax
        float32[(rows+2)*(columns+2)] elevation grid, row-major

    The elevation grid includes one halo sample on every side.
    Core vertex elevations occupy [1:-1, 1:-1].
    """
    with rasterio.open(dem_path) as dataset:
        t = dataset.transform
        if t.b != 0 or t.d != 0 or t.a <= 0 or t.e >= 0:
            raise ValueError(f"DEM must be north-up: {dem_path}")
        if not math.isclose(t.a, -t.e, rel_tol=1e-8):
            raise ValueError(f"DEM must have square cells: {dem_path}")

        resolution = t.a
        bounds = dataset.bounds

        # Read this tile's DEM only; never load the whole project.
        data = dataset.read(1, masked=True)
        values = np.asarray(data.compressed(), dtype=np.float64)
        values = values[np.isfinite(values)]

        if not len(values):
            return None

        zmin = float(values.min())
        zmax = float(values.max())

        with destination.open("wb") as output:
            output.write(MAGIC)
            output.write(struct.pack("<I", levels))
            output.write(struct.pack(
                "<4d2f",
                bounds.left, bounds.bottom, bounds.right, bounds.top,
                zmin, zmax,
            ))

            for level in range(levels):
                stride = 2 ** level

                east = axis_vertices(dataset.width, stride, resolution)
                south = axis_vertices(dataset.height, stride, resolution)

                # Halo samples support normal estimation. At the source
                # edge they clamp; cross-tile normal reconciliation is
                # left to the viewer or a subsequent processing pass.
                halo_distance = stride * resolution
                east_halo = np.concatenate((
                    [east[0] - halo_distance],
                    east,
                    [east[-1] + halo_distance],
                ))
                south_halo = np.concatenate((
                    [south[0] - halo_distance],
                    south,
                    [south[-1] + halo_distance],
                ))

                elevations = sample_vertices(
                    dataset,
                    bounds.left + east_halo,
                    bounds.top - south_halo,
                )

                output.write(struct.pack("<2I", len(east), len(south)))
                output.write(east.astype("<f4").tobytes())
                output.write(south.astype("<f4").tobytes())
                output.write(elevations.astype("<f4").tobytes())

        return (
            bounds.left, bounds.bottom, bounds.right, bounds.top,
            zmin, zmax,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Prepare lightweight multi-detail terrain tiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input-dir", required=True)
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--levels", type=int, default=5)
    args = parser.parse_args()

    if not 1 <= args.levels <= 10:
        parser.error("--levels must be between 1 and 10.")

    source = Path(args.input_dir).resolve()
    manifest_path = source / "tile_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    output = Path(args.output_dir).resolve()
    if output.exists() and (
        not output.is_dir() or any(output.iterdir())
    ):
        parser.error("Output folder must be empty or not yet exist.")

    tiles_dir = output / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    entries = []

    for number, tile in enumerate(manifest["tiles"]):
        if not tile.get("dem"):
            continue

        # Generated names do not depend on the source filename and contain
        # no spaces, simplifying the legacy viewer's index parsing.
        filename = f"tile_{number:06d}.trn"
        destination = tiles_dir / filename

        print(f"Preparing {tile['tile']}")

        result = prepare_tile(
            source / tile["dem"], destination, args.levels
        )
        if result is not None:
            entries.append((f"tiles/{filename}", result))

    if not entries:
        raise ValueError("No valid terrain tiles were generated.")

    # Choose one shared model origin. Rendering can additionally use a
    # floating camera origin without altering these stored map coordinates.
    origin_x = min(bounds[0] for _, bounds in entries)
    origin_y = max(bounds[3] for _, bounds in entries)
    origin_z = math.floor(min(bounds[4] for _, bounds in entries))

    # The runtime uses this intentionally simple index instead of needing
    # a JSON library. Bounds and placement come from the DEM manifest.
    with (output / "terrain_index.txt").open(
        "w", encoding="utf-8", newline="\n"
    ) as index:
        index.write("TERRAIN_INDEX_V1\n")
        index.write(f"{origin_x:.9f} {origin_y:.9f} {origin_z:.9f}\n")
        index.write(f"{len(entries)}\n")

        for filename, bounds in entries:
            index.write(
                filename + " "
                + " ".join(f"{value:.9f}" for value in bounds)
                + "\n"
            )

    # Preserve GIS metadata separately for inspection and future tooling.
    metadata = {
        "crs": manifest.get("crs"),
        "vertical_reference": manifest.get("vertical_reference"),
        "model_origin_easting_northing_elevation": [
            origin_x, origin_y, origin_z
        ],
        "levels": args.levels,
        "tile_count": len(entries),
        "source_manifest": str(manifest_path),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"Prepared {len(entries)} tiles in {output}")


if __name__ == "__main__":
    main()
