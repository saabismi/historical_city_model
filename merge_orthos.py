#!/usr/bin/env python3

import argparse
import glob
import tempfile
from pathlib import Path

from osgeo import gdal


def main():
    parser = argparse.ArgumentParser(
        description="Merge single-band rasters, ignoring pixels valued 255."
    )
    parser.add_argument(
        "-i", "--inputs",
        nargs="+",
        required=True,
        help="Input raster filenames or wildcard patterns.",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output GeoTIFF filename.",
    )
    args = parser.parse_args()

    gdal.UseExceptions()

    output = Path(args.output).resolve()
    if output.suffix.lower() not in {".tif", ".tiff"}:
        parser.error("Output filename must end in .tif or .tiff.")

    inputs = []
    seen = set()

    for pattern in args.inputs:
        matches = sorted(glob.glob(pattern))
        if not matches:
            parser.error(f"No files match: {pattern}")

        for filename in matches:
            path = Path(filename).resolve()

            if not path.is_file():
                parser.error(f"Not a file: {path}")
            if path == output:
                parser.error("The output file cannot also be an input.")

            if path not in seen:
                inputs.append(str(path))
                seen.add(path)

    for filename in inputs:
        dataset = gdal.Open(filename)
        if dataset is None:
            raise RuntimeError(f"Cannot open: {filename}")
        if dataset.RasterCount != 1:
            raise ValueError(f"Expected a single-band raster: {filename}")
        dataset = None

    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="aerial_mosaic_") as temp_dir:
        vrt = gdal.BuildVRT(
            str(Path(temp_dir) / "mosaic.vrt"),
            inputs,
            options=gdal.BuildVRTOptions(
                srcNodata=255,
                VRTNodata=255,
                resolution="highest",
            ),
        )
        if vrt is None:
            raise RuntimeError("Failed to build mosaic.")

        result = gdal.Translate(
            str(output),
            vrt,
            options=gdal.TranslateOptions(
                format="GTiff",
                noData=255,
                creationOptions=[
                    "TILED=YES",
                    "COMPRESS=LZW",
                    "BIGTIFF=IF_SAFER",
                ],
            ),
        )
        if result is None:
            raise RuntimeError("Failed to write output.")

        result = None
        vrt = None

    print(f"Merged {len(inputs)} rasters into {output}")


if __name__ == "__main__":
    main()
