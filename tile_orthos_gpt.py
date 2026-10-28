#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import numpy as np
from osgeo import gdal, osr


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Tile a single-band raster into metre-sized cores with "
            "an overlapping buffer on each side."
        )
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Input raster filename.",
    )
    parser.add_argument(
        "-o", "--output-dir", required=True,
        help="Output folder; must be empty or not yet exist.",
    )
    parser.add_argument(
        "--tile-width", type=float, default=250,
        help="Core width in metres (default: 250).",
    )
    parser.add_argument(
        "--tile-height", type=float, default=250,
        help="Core height in metres (default: 250).",
    )
    parser.add_argument(
        "--overlap", type=float, default=10,
        help="Buffer on EACH side in metres (default: 10).",
    )
    parser.add_argument(
        "--nodata", type=float, default=255,
        help="NoData value (default: 255).",
    )
    parser.add_argument(
        "--keep-empty", action="store_true",
        help="Keep tiles whose core contains no valid pixels.",
    )
    args = parser.parse_args()

    if args.tile_width <= 0 or args.tile_height <= 0:
        parser.error("Tile dimensions must be positive.")
    if args.overlap < 0:
        parser.error("Overlap must be nonnegative.")

    gdal.UseExceptions()
    source = gdal.Open(args.input, gdal.GA_ReadOnly)

    if source is None:
        raise RuntimeError(f"Cannot open: {args.input}")
    if source.RasterCount != 1:
        parser.error("Input must be single-band.")

    projection = source.GetProjection()
    if not projection:
        parser.error("Input must have a coordinate reference system.")

    crs = osr.SpatialReference()
    crs.ImportFromWkt(projection)

    if not crs.IsProjected():
        parser.error(
            "Use a projected CRS suitable for local distances, such as UTM. "
            "Reproject the raster first."
        )

    metres_per_unit = crs.GetLinearUnits()
    if metres_per_unit <= 0:
        parser.error("Cannot determine CRS linear units.")

    gt = source.GetGeoTransform()
    if gt[2] != 0 or gt[4] != 0 or gt[1] <= 0 or gt[5] >= 0:
        parser.error("Input must be north-up without rotation.")

    pixel_width_m = gt[1] * metres_per_unit
    pixel_height_m = abs(gt[5]) * metres_per_unit

    core_width = max(1, math.ceil(args.tile_width / pixel_width_m))
    core_height = max(1, math.ceil(args.tile_height / pixel_height_m))

    buffer_x = math.ceil(args.overlap / pixel_width_m)
    buffer_y = math.ceil(args.overlap / pixel_height_m)

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            parser.error("Output folder must be empty or not yet exist.")
    output_dir.mkdir(parents=True, exist_ok=True)

    band = source.GetRasterBand(1)
    mask_band = band.GetMaskBand()

    written = 0
    skipped = 0

    for row, core_y in enumerate(
        range(0, source.RasterYSize, core_height)
    ):
        for col, core_x in enumerate(
            range(0, source.RasterXSize, core_width)
        ):
            # Core dimensions may be smaller at the raster's outer edges.
            cw = min(core_width, source.RasterXSize - core_x)
            ch = min(core_height, source.RasterYSize - core_y)

            # Add buffer, clipped to the input raster extent.
            x0 = max(0, core_x - buffer_x)
            y0 = max(0, core_y - buffer_y)
            x1 = min(source.RasterXSize, core_x + cw + buffer_x)
            y1 = min(source.RasterYSize, core_y + ch + buffer_y)

            width = x1 - x0
            height = y1 - y0

            data = band.ReadAsArray(x0, y0, width, height)
            mask = mask_band.ReadAsArray(x0, y0, width, height)

            valid = (
                (data != args.nodata)
                & (mask != 0)
                & np.isfinite(data)
            )

            # Core position within the buffered output tile.
            local_x = core_x - x0
            local_y = core_y - y0
            core_valid = valid[
                local_y:local_y + ch,
                local_x:local_x + cw,
            ]

            # Skip based on the CORE, not the overlapping buffer.
            if not args.keep_empty and not np.any(core_valid):
                skipped += 1
                continue

            filename = output_dir / f"tile_r{row:05d}_c{col:05d}.tif"

            tile = gdal.Translate(
                str(filename),
                source,
                options=gdal.TranslateOptions(
                    format="GTiff",
                    srcWin=[x0, y0, width, height],
                    noData=args.nodata,
                    creationOptions=[
                        "TILED=YES",
                        "COMPRESS=LZW",
                        "BIGTIFF=IF_SAFER",
                    ],
                ),
            )

            if tile is None:
                raise RuntimeError(f"Failed to write: {filename}")

            data[~valid] = args.nodata
            tile.GetRasterBand(1).WriteArray(data)

            # Core bounds in the raster CRS's map units.
            xmin = gt[0] + core_x * gt[1]
            xmax = gt[0] + (core_x + cw) * gt[1]
            ymax = gt[3] + core_y * gt[5]
            ymin = gt[3] + (core_y + ch) * gt[5]

            tile.SetMetadata({
                "CORE_X_OFFSET_PIXELS": str(local_x),
                "CORE_Y_OFFSET_PIXELS": str(local_y),
                "CORE_WIDTH_PIXELS": str(cw),
                "CORE_HEIGHT_PIXELS": str(ch),
                "CORE_XMIN": str(xmin),
                "CORE_YMIN": str(ymin),
                "CORE_XMAX": str(xmax),
                "CORE_YMAX": str(ymax),
                "REQUESTED_BUFFER_METRES": str(args.overlap),
            }, "TILING")

            tile = None
            written += 1

    source = None

    print(
        f"Core: {core_width * pixel_width_m:g} × "
        f"{core_height * pixel_height_m:g} m"
    )
    print(
        f"Buffer per side: {buffer_x * pixel_width_m:g} m horizontally, "
        f"{buffer_y * pixel_height_m:g} m vertically"
    )
    print(f"Written: {written}; empty cores skipped: {skipped}")
    print(f"Output: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
