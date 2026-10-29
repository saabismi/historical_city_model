#!/usr/bin/env python3
"""
Clip all GeoPackage vector layers to a raster's full rectangular extent.

Usage:
    python clip_mtk.py -i MTK-tie_2009.gpkg -r aerial.tif
    python clip_mtk.py -i MTK-tie_2009.gpkg -r aerial.tif --overwrite

Arguments:
    -i, --input       Input GeoPackage.
    -r, --raster      Raster defining the full clipping extent.
    --overwrite       Replace an existing output GeoPackage.

Output:
    <working directory>/MTK_2009_clipped/<input_stem>_clipped.gpkg

Layer names and attributes are retained. All output layers use EPSG:3067.
Non-3067 inputs produce a warning and are reprojected. Missing CRS metadata
causes an error rather than guessing.

Requires GDAL Python bindings.
"""

import argparse
import sys
import tempfile
from pathlib import Path

from osgeo import gdal, ogr, osr


def target_crs():
    """Create ETRS-TM35FIN with conventional easting/northing axis order."""
    crs = osr.SpatialReference()
    crs.ImportFromEPSG(3067)
    crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return crs


def normalized_crs(crs):
    """Clone a CRS and use traditional GIS x/y axis order."""
    result = crs.Clone()
    result.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return result


def raster_rectangle(filename):
    """
    Build the raster's full bounding rectangle in its own CRS.

    No raster pixel values are inspected. For rotated rasters, use the
    axis-aligned bounding rectangle of their four corners.
    """
    raster = gdal.Open(str(filename), gdal.GA_ReadOnly)
    if raster is None:
        raise ValueError(f"Cannot open raster: {filename}")

    projection = raster.GetProjection()
    if not projection:
        raise ValueError("Raster has no coordinate reference system.")

    crs = osr.SpatialReference()
    crs.ImportFromWkt(projection)
    crs = normalized_crs(crs)

    transform = raster.GetGeoTransform()

    # Convert each outer pixel-grid corner to map coordinates.
    corners = []
    for column, row in (
        (0, 0),
        (raster.RasterXSize, 0),
        (raster.RasterXSize, raster.RasterYSize),
        (0, raster.RasterYSize),
    ):
        corners.append((
            transform[0] + column * transform[1] + row * transform[2],
            transform[3] + column * transform[4] + row * transform[5],
        ))

    xmin = min(point[0] for point in corners)
    xmax = max(point[0] for point in corners)
    ymin = min(point[1] for point in corners)
    ymax = max(point[1] for point in corners)

    if xmin >= xmax or ymin >= ymax:
        raise ValueError("Raster extent is empty or invalid.")

    if transform[2] != 0 or transform[4] != 0:
        print(
            "WARNING: Raster is rotated; using its axis-aligned "
            "bounding rectangle."
        )

    # Densify rectangle edges before any reprojection. This approximates
    # curved transformed boundaries rather than transforming only corners.
    rectangle_corners = [
        (xmin, ymin),
        (xmax, ymin),
        (xmax, ymax),
        (xmin, ymax),
    ]
    ring = ogr.Geometry(ogr.wkbLinearRing)
    subdivisions = 100

    for index in range(4):
        start = rectangle_corners[index]
        end = rectangle_corners[(index + 1) % 4]

        for step in range(subdivisions):
            fraction = step / subdivisions
            ring.AddPoint_2D(
                start[0] + fraction * (end[0] - start[0]),
                start[1] + fraction * (end[1] - start[1]),
            )

    ring.AddPoint_2D(*rectangle_corners[0])

    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(ring)

    raster = None
    return polygon, crs


def write_clip_file(filename, geometry, crs):
    """Write one clipping polygon into a temporary GeoPackage."""
    driver = ogr.GetDriverByName("GPKG")
    dataset = driver.CreateDataSource(str(filename))
    if dataset is None:
        raise RuntimeError("Could not create temporary clipping dataset.")

    layer = dataset.CreateLayer(
        "clip_extent", srs=crs, geom_type=ogr.wkbPolygon
    )
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetGeometry(geometry)
    layer.CreateFeature(feature)

    feature = None
    dataset = None


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Clip all GeoPackage vector layers to a raster extent "
            "and output EPSG:3067."
        )
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Input GeoPackage."
    )
    parser.add_argument(
        "-r", "--raster", required=True,
        help="Raster defining the full rectangular clipping extent."
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace an existing output GeoPackage."
    )
    args = parser.parse_args()

    gdal.UseExceptions()
    ogr.UseExceptions()
    osr.UseExceptions()

    input_path = Path(args.input).resolve()
    raster_path = Path(args.raster).resolve()

    if not input_path.is_file():
        parser.error(f"Input GeoPackage does not exist: {input_path}")
    if not raster_path.is_file():
        parser.error(f"Raster does not exist: {raster_path}")

    output_dir = Path.cwd() / "MTK_2009_clipped"
    output_path = output_dir / f"{input_path.stem}_clipped.gpkg"

    if output_path.resolve() == input_path:
        parser.error("Output cannot be the input GeoPackage.")

    if output_path.exists() and not args.overwrite:
        parser.error(
            f"Output already exists: {output_path}\n"
            "Use --overwrite to replace it."
        )

    source = ogr.Open(str(input_path), 0)
    if source is None:
        raise ValueError(f"Cannot open GeoPackage: {input_path}")

    destination_crs = target_crs()
    rectangle, raster_crs = raster_rectangle(raster_path)

    if not raster_crs.IsSame(destination_crs):
        print(
            "WARNING: Raster CRS is not EPSG:3067. "
            "Its clipping extent will be reprojected."
        )

    # Validate every layer before starting output creation.
    layers = []
    for index in range(source.GetLayerCount()):
        layer = source.GetLayerByIndex(index)
        name = layer.GetName()
        geometry_type = layer.GetGeomType()

        # Nonspatial tables cannot be spatially clipped.
        if geometry_type == ogr.wkbNone:
            print(f"Skipping nonspatial table: {name}")
            continue

        crs = layer.GetSpatialRef()
        if crs is None:
            raise ValueError(f"Layer {name!r} has no CRS.")

        crs = normalized_crs(crs)

        if not crs.IsSame(destination_crs):
            print(
                f"WARNING: Layer {name!r} is not EPSG:3067. "
                "It will be reprojected."
            )

        layers.append((name, crs))

    source = None

    if not layers:
        raise ValueError("No spatial vector layers found.")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build into a temporary file first. An existing output is retained
    # until all layers have been successfully processed.
    with tempfile.TemporaryDirectory(
        prefix="_mtk_clip_", dir=str(output_dir)
    ) as temporary:
        temporary_dir = Path(temporary)
        temporary_output = temporary_dir / "result.gpkg"

        for index, (name, layer_crs) in enumerate(layers):
            print(f"[{index + 1}/{len(layers)}] Clipping {name}")

            # Source clipping occurs before reprojection, so express the
            # clipping polygon in this particular source layer's CRS.
            clip_polygon = rectangle.Clone()

            if not raster_crs.IsSame(layer_crs):
                clip_polygon.Transform(
                    osr.CoordinateTransformation(raster_crs, layer_crs)
                )

            clip_file = temporary_dir / f"clip_{index:05d}.gpkg"
            write_clip_file(clip_file, clip_polygon, layer_crs)

            options = [
                "-f", "GPKG",
                "-t_srs", "EPSG:3067",
                "-clipsrc", str(clip_file),
                "-clipsrclayer", "clip_extent",
                "-nln", name,

                # Clipping can split a polygon or line into several parts.
                "-nlt", "PROMOTE_TO_MULTI",

                # Build an output spatial index for efficient GIS access.
                "-lco", "SPATIAL_INDEX=YES",
            ]

            if index > 0:
                # Add subsequent layers to the same output GeoPackage.
                options.append("-update")

            # Do not use skip-failures: geometry or transformation errors
            # should stop processing instead of silently dropping features.
            result = gdal.VectorTranslate(
                str(temporary_output),
                str(input_path),
                options=gdal.VectorTranslateOptions(
                    options=options,
                    layers=[name],
                ),
            )

            if result is None:
                raise RuntimeError(f"Failed to clip layer: {name}")

            result = None

        # Replace only after successful completion. This will fail clearly
        # if the destination is locked, for example by an open QGIS session.
        temporary_output.replace(output_path)

    print(f"\nFinished: {output_path}")
    print(f"Spatial layers processed: {len(layers)}")
    print("Output CRS: EPSG:3067")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
