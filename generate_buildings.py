#!/usr/bin/env python3
"""
Generate textured buildings directly from MTK footprints and oldest LiDAR.

Initial selection (with GLB for inspection):
python generate_buildings.py ^
    --buildings MTK_2009_clipped\mtk-rakennus_2009_clipped.gpkg ^
    --downloads espoo1965_clouds ^
    --terrain Historyviewer\2008_terrain ^
    --materials Historyviewer\assets\buildings\materials.json ^
    -o Historyviewer\2008_buildings_textured ^
    --count 100 ^
    --glb ^
    --assume-epsg-3067

Reuse persistent IDs:
    python generate_buildings.py \
        --selection buildings_test/selected_buildings.gpkg \
        --downloads espoo1965_clouds \
        --terrain Historyviewer/2008_terrain \
        --materials assets/buildings/materials.json \
        -o buildings_test_v2 --glb

Arguments:
    --buildings     Original footprint GeoPackage.
    --selection     Previously generated selected_buildings.gpkg instead.
    --layer         Original layer name; default rakennus.
    --count         Initial selection count; default 100; 0 means all.
    --downloads     Folder containing download_manifest.csv and pointclouds/.
    --index         Coverage shapefile override.
    --terrain       Viewer terrain folder containing terrain_index.txt.
    --materials     Shared materials.json, including per-building overrides.
    -o, --output    New or empty output folder.
    --glb           Export textured inspection GLBs.
    --assume-epsg-3067
                    Accept LAS files without CRS metadata.

Outputs:
    selected_buildings.gpkg
    roof_fit_report.csv
    building_index.txt
    building_manifest.json
    building_materials.json
    building_materials.txt
    textures/*.tga
    tiles/*.bld       BLD2 format
    glb/*.glb        Optional

Roofer is NOT used.
Roof candidates are original class 1 (and class 6 if present).
Ground is class 2. Roof selection uses exact footprints, without a buffer.
Simple roofs are approximations; complex roofs are flagged as uncertain.
The current C++ viewer needs a BLD2 loader before these buildings can render.

Requires:
    GDAL, numpy, scipy, laspy+lazrs, shapely>=2, pyproj,
    trimesh, mapbox-earcut, Pillow.
"""

import argparse
import csv
import json
import math
import shutil
import struct
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote, urlsplit

import laspy
import numpy as np
import trimesh
from PIL import Image
from osgeo import ogr, osr
from pyproj import CRS
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from shapely import (
    from_wkb, to_wkb, make_valid, union_all, intersects_xy,
    points as geometry_points,
)
from shapely.geometry import box, LineString
from shapely.geometry.polygon import orient
from shapely.ops import split
from shapely.strtree import STRtree
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals


# ============================================================================
# EDITABLE SETTINGS
# ============================================================================

CONFIG = {
    "roof_classes": [1, 6],
    "ground_class": 2,
    "chunk_size": 1_000_000,

    # Buildings are processed in spatial batches to bound point memory.
    # Source LAS/LAZ files may be reread for multiple batches.
    "processing_cell_metres": 1000.0,
    "ground_context_metres": 60.0,
    "ground_neighbors": 24,

    "minimum_roof_points": 20,
    "maximum_fit_points": 4000,
    "robust_scale": 0.35,
    
    # Buildings with insufficient roof observations remain in the world.
    "fallback_floor_height_metres": 3.0,
    "fallback_floors": 1,


    "flat_max_slope_degrees": 5.0,
    "flat_error_allowance_metres": 0.08,

    "gable_min_slope_degrees": 8.0,
    "gable_max_slope_degrees": 65.0,
    "gable_min_points_per_side": 8,
    "gable_min_fraction_per_side": 0.15,
    "gable_min_rise_metres": 0.60,
    "gable_min_relative_improvement": 0.20,
    "gable_min_absolute_improvement_metres": 0.12,
    "gable_max_error_metres": 0.90,
    "uncertain_error_metres": 0.90,

    "wall_below_ground_metres": 0.50,
    "gable_overhang_metres": 0.30,

    "ridge_angle_samples": 16,
    "ridge_start_positions": [-0.20, 0.0, 0.20],
    "seed": 12345,
}

WALL, ROOF, GROUND = 1, 2, 3
NO_MATERIAL = 0xFFFFFFFF


# ============================================================================
# GIS INPUT AND PERSISTENT BUILDING IDS
# ============================================================================

def epsg3067():
    """Return EPSG:3067 with conventional GIS axis order."""
    crs = osr.SpatialReference()
    crs.ImportFromEPSG(3067)
    crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return crs


def normalized(crs):
    result = crs.Clone()
    result.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return result


def polygon_parts(geometry):
    """Yield polygon components from repaired geometry."""
    if geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
    elif hasattr(geometry, "geoms"):
        for child in geometry.geoms:
            yield from polygon_parts(child)


def read_buildings(path, layer_name, count, reuse_ids):
    """
    Read footprints, reprojecting to EPSG:3067 when necessary.

    Initial IDs follow the selected reading order. They become persistent
    when saved and subsequently reused through --selection.
    """
    dataset = ogr.Open(str(path))
    if dataset is None:
        raise ValueError(f"Cannot open {path}")
    layer = dataset.GetLayerByName(layer_name)
    if layer is None:
        raise ValueError(f"Layer missing: {layer_name}")

    source = layer.GetSpatialRef()
    if source is None:
        raise ValueError("Building layer has no CRS.")

    transformation = None
    if not normalized(source).IsSame(epsg3067()):
        print("Reprojecting building footprints to EPSG:3067.")
        transformation = osr.CoordinateTransformation(
            normalized(source), epsg3067()
        )

    if reuse_ids and layer.GetLayerDefn().GetFieldIndex("building_id") < 0:
        raise ValueError("Selection file has no building_id field.")

    result, seen = [], set()

    for feature in layer:
        geometry = feature.GetGeometryRef()
        if geometry is None or geometry.IsEmpty():
            continue

        geometry = geometry.Clone()
        if transformation and geometry.Transform(transformation) != 0:
            raise ValueError("Footprint transformation failed.")

        pieces = list(polygon_parts(make_valid(
            from_wkb(bytes(geometry.ExportToWkb()))
        )))
        if not pieces:
            continue

        footprint = union_all(pieces)
        if footprint.area <= 0:
            continue

        identifier = (
            feature.GetField("building_id") if reuse_ids
            else f"building_{len(result) + 1:06d}"
        )
        if identifier is None or not str(identifier).strip():
            raise ValueError("Empty building_id in selection.")
        identifier = str(identifier)
        if identifier in seen:
            raise ValueError(f"Duplicate building_id: {identifier}")
        seen.add(identifier)

        anchor = footprint.representative_point()
        result.append({
            "id": identifier,
            "footprint": footprint,
            "origin": np.array([anchor.x, anchor.y]),
        })

        if not reuse_ids and count and len(result) >= count:
            break

    dataset = None
    if not result:
        raise ValueError("No valid building footprints found.")
    return result


def save_selection(path, buildings):
    """Save the authoritative footprint-to-ID mapping."""
    dataset = ogr.GetDriverByName("GPKG").CreateDataSource(str(path))
    layer = dataset.CreateLayer(
        "buildings", epsg3067(), ogr.wkbMultiPolygon
    )
    layer.CreateField(ogr.FieldDefn("building_id", ogr.OFTString))

    for building in buildings:
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetField("building_id", building["id"])
        feature.SetGeometry(ogr.ForceToMultiPolygon(
            ogr.CreateGeometryFromWkb(to_wkb(building["footprint"]))
        ))
        layer.CreateFeature(feature)

    layer = None
    dataset = None


def read_terrain(folder):
    """Read actual viewer tile bounds and shared origin."""
    with (folder / "terrain_index.txt").open(encoding="utf-8") as handle:
        if handle.readline().strip() != "TERRAIN_INDEX_V1":
            raise ValueError("Invalid terrain index.")
        origin = list(map(float, handle.readline().split()))
        count = int(handle.readline())
        tiles = []

        for _ in range(count):
            parts = handle.readline().split()
            if len(parts) != 7:
                raise ValueError("Invalid terrain tile record.")
            tiles.append({
                "file": parts[0],
                "name": Path(parts[0]).stem,
                "bounds": tuple(map(float, parts[1:5])),

                # Used only as a last-resort building base estimate.
                "zmin": float(parts[5]),
                "zmax": float(parts[6]),
            })


    return origin, tiles


def owner_tile(origin, tiles, tree):
    """Choose one owner tile, including an outer boundary if necessary."""
    point = geometry_points(origin[0], origin[1])
    candidates = sorted(map(int, tree.query(point, predicate="intersects")))
    for index in candidates:
        xmin, ymin, xmax, ymax = tiles[index]["bounds"]
        if xmin <= origin[0] < xmax and ymin <= origin[1] < ymax:
            return tiles[index]
    return tiles[candidates[0]] if candidates else None


# ============================================================================
# DOWNLOAD MANIFEST AND OLDEST COVERAGE
# ============================================================================

def relative_path(value):
    text = unquote(str(value)).strip().replace("\\", "/").lstrip("/")
    prefix = "mml/laserkeilaus/2008_latest/"
    return text[len(prefix):] if text.startswith(prefix) else text


def downloaded_files(folder):
    """Locate flattened source files using the download manifest."""
    with (folder / "download_manifest.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    result = {}
    for row in rows:
        relative = relative_path(row.get("path") or "")
        if not relative:
            url = urlsplit(row.get("url", "")).path
            marker = "/2008_latest/"
            if marker not in url:
                raise ValueError(f"Invalid manifest URL: {url}")
            relative = relative_path(url.split(marker, 1)[1])

        local = folder / "pointclouds" / Path(relative).name
        if not local.is_file() and row.get("local_path"):
            local = Path(row["local_path"])

        result[relative] = local

    return result


def read_coverages(index_path, region):
    """Read indexed footprints and reproject them to EPSG:3067."""
    dataset = ogr.Open(str(index_path))
    if dataset is None:
        raise ValueError(f"Cannot open coverage index: {index_path}")
    layer = dataset.GetLayer(0)
    source = layer.GetSpatialRef()
    if source is None:
        raise ValueError("Coverage index has no CRS; check its .prj.")

    transformation = None
    if not normalized(source).IsSame(epsg3067()):
        transformation = osr.CoordinateTransformation(
            normalized(source), epsg3067()
        )

    definition = layer.GetLayerDefn()
    fields = {
        definition.GetFieldDefn(i).GetName().lower():
        definition.GetFieldDefn(i).GetName()
        for i in range(definition.GetFieldCount())
    }
    if "path" not in fields:
        raise ValueError("Coverage index has no path field.")

    # Restrict reading in the index's own CRS.
    query = ogr.CreateGeometryFromWkb(to_wkb(region))
    if transformation:
        reverse = osr.CoordinateTransformation(
            epsg3067(), normalized(source)
        )
        query.Transform(reverse)
    layer.SetSpatialFilter(query)

    pieces = defaultdict(list)
    for feature in layer:
        value = feature.GetField(fields["path"])
        geometry = feature.GetGeometryRef()
        if value is None or geometry is None:
            continue

        relative = relative_path(value)
        if not relative.split("/", 1)[0].isdigit():
            continue

        geometry = geometry.Clone()
        if transformation and geometry.Transform(transformation) != 0:
            raise ValueError(f"Index transformation failed: {relative}")

        footprint = make_valid(
            from_wkb(bytes(geometry.ExportToWkb()))
        ).intersection(region)
        if not footprint.is_empty and footprint.area > 0:
            pieces[relative].append(footprint)

    dataset = None
    return [
        {
            "relative": relative,
            "year": int(relative.split("/", 1)[0]),
            "footprint": make_valid(union_all(parts)),
        }
        for relative, parts in pieces.items()
    ]


def owned_sources(coverages, region, downloads):
    """
    Construct oldest ownership for one processing batch.

    Missing oldest files are an error, rather than silently using newer data.
    Same-year overlaps remain eligible.
    """
    years = defaultdict(list)
    for entry in coverages:
        if entry["footprint"].intersects(region):
            years[entry["year"]].append(entry)

    older = shapely_empty()
    sources = []

    for year in sorted(years):
        entries = years[year]
        for entry in entries:
            footprint = entry["footprint"].intersection(region)
            owner = make_valid(footprint.difference(older))
            if owner.is_empty or owner.area <= 0:
                continue

            local = downloads.get(entry["relative"])
            if local is None or not local.is_file():
                raise ValueError(
                    "Missing oldest source required for this batch: "
                    + entry["relative"]
                )
            sources.append((local, owner))

        older = union_all([
            older,
            *[entry["footprint"].intersection(region) for entry in entries],
        ])

    return sources, older


def shapely_empty():
    from shapely.geometry import GeometryCollection
    return GeometryCollection()


def read_batch_points(buildings, sources, region, assume_crs):
    """Read candidate roofs and nearby ground for one spatial batch."""
    footprints = [b["footprint"] for b in buildings]
    tree = STRtree(footprints)
    roof_arrays = [[] for _ in buildings]
    ground_arrays = []
    xmin, ymin, xmax, ymax = region.bounds

    for filename, owner in sources:
        print(f"  Reading {filename.name}", flush=True)
        with laspy.open(filename) as reader:
            source_crs = reader.header.parse_crs()
            if source_crs is None:
                if not assume_crs:
                    raise ValueError(
                        f"No LAS CRS: {filename}. Use --assume-epsg-3067 "
                        "only after verifying the coordinates."
                    )
            elif not CRS(source_crs).equals(
                CRS.from_epsg(3067), ignore_axis_order=True
            ):
                raise ValueError(f"LAS CRS mismatch: {filename}")

            for chunk in reader.chunk_iterator(CONFIG["chunk_size"]):
                classes = np.asarray(chunk.classification)
                selected = (
                    np.isin(classes, [
                        *CONFIG["roof_classes"], CONFIG["ground_class"]
                    ])
                    & ~np.asarray(chunk.withheld, dtype=bool)
                )
                if not np.any(selected):
                    continue

                xyz = np.column_stack((
                    np.asarray(chunk.x)[selected],
                    np.asarray(chunk.y)[selected],
                    np.asarray(chunk.z)[selected],
                ))
                classes = classes[selected]
                inside = (
                    np.isfinite(xyz).all(axis=1)
                    & (xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax)
                    & (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax)
                )
                xyz, classes = xyz[inside], classes[inside]
                if not len(xyz):
                    continue

                keep = (
                    intersects_xy(owner, xyz[:, 0], xyz[:, 1])
                    & intersects_xy(region, xyz[:, 0], xyz[:, 1])
                )
                xyz, classes = xyz[keep], classes[keep]

                ground = xyz[classes == CONFIG["ground_class"]]
                if len(ground):
                    ground_arrays.append(ground)

                roof = xyz[np.isin(classes, CONFIG["roof_classes"])]
                if len(roof):
                    pairs = tree.query(
                        geometry_points(roof[:, 0], roof[:, 1]),
                        predicate="intersects",
                    )
                    for index in np.unique(pairs[1]):
                        roof_arrays[int(index)].append(
                            roof[pairs[0][pairs[1] == index]]
                        )

    roofs = [
        np.unique(np.concatenate(parts), axis=0)
        if parts else np.empty((0, 3))
        for parts in roof_arrays
    ]
    ground = (
        np.unique(np.concatenate(ground_arrays), axis=0)
        if ground_arrays else np.empty((0, 3))
    )
    return roofs, ground


# ============================================================================
# ROOF FITTING
# ============================================================================

def error_score(residual):
    """Trim the largest 10 percent before calculating RMSE."""
    squared = np.sort(np.asarray(residual) ** 2)
    return float(np.sqrt(
        squared[:max(1, math.ceil(len(squared) * 0.9))].mean()
    ))


def gable_height(parameters, xy):
    """Evaluate an asymmetric gable with a slightly tilted ridge."""
    theta, offset, elevation, along, left, right = parameters
    direction = np.array([math.cos(theta), math.sin(theta)])
    across = np.array([-math.sin(theta), math.cos(theta)])
    s = xy @ direction
    t = xy @ across - offset
    return (
        elevation + along * s
        - left * np.maximum(-t, 0)
        - right * np.maximum(t, 0)
    )


def fit_roof(points, origin):
    """Fit flat, shed, and gable models, with conservative gable selection."""
    rng = np.random.default_rng(CONFIG["seed"])
    if len(points) > CONFIG["maximum_fit_points"]:
        points = points[rng.choice(
            len(points), CONFIG["maximum_fit_points"], replace=False
        )]

    xy = points[:, :2] - origin
    z = points[:, 2]
    design = np.column_stack((np.ones(len(xy)), xy))
    scale = CONFIG["robust_scale"]

    flat = float(np.median(z))
    flat_error = error_score(z - flat)
    initial = np.linalg.lstsq(design, z, rcond=None)[0]
    plane = least_squares(
        lambda p: design @ p - z, initial,
        loss="soft_l1", f_scale=scale,
    ).x
    plane_error = error_score(design @ plane - z)
    angle = math.degrees(math.atan(math.hypot(plane[1], plane[2])))

    if (
        angle <= CONFIG["flat_max_slope_degrees"]
        and flat_error <= plane_error + CONFIG["flat_error_allowance_metres"]
    ):
        selected = {"type": "flat", "parameters": [flat], "error": flat_error}
    else:
        selected = {
            "type": "shed", "parameters": plane.tolist(), "error": plane_error
        }

    minimum = math.tan(math.radians(CONFIG["gable_min_slope_degrees"]))
    maximum = math.tan(math.radians(CONFIG["gable_max_slope_degrees"]))
    best = None
    samples = CONFIG["ridge_angle_samples"]

    for theta in np.linspace(0, math.pi, samples, endpoint=False):
        projected = xy @ np.array([-math.sin(theta), math.cos(theta)])
        low, high = np.percentile(projected, [5, 95])
        span = high - low
        if span < 1:
            continue

        for position in CONFIG["ridge_start_positions"]:
            lower = [
                theta - math.pi / samples, low + span * .10,
                float(z.min()) - 3, -.20, minimum, minimum,
            ]
            upper = [
                theta + math.pi / samples, high - span * .10,
                float(z.max()) + span * maximum + 3,
                .20, maximum, maximum,
            ]
            initial = [
                theta,
                np.clip((low + high)/2 + position*span, lower[1], upper[1]),
                float(np.percentile(z, 85)), 0,
                np.clip(.35, minimum, maximum),
                np.clip(.35, minimum, maximum),
            ]

            result = least_squares(
                lambda p: gable_height(p, xy) - z,
                initial, bounds=(lower, upper),
                loss="soft_l1", f_scale=scale, max_nfev=120,
            )
            if not result.success:
                continue

            p = result.x
            t = xy @ np.array([-math.sin(p[0]), math.cos(p[0])]) - p[1]
            required = max(
                CONFIG["gable_min_points_per_side"],
                math.ceil(len(xy) * CONFIG["gable_min_fraction_per_side"]),
            )
            if min((t < -.25).sum(), (t > .25).sum()) < required:
                continue

            left_rise = np.percentile(-t[t < 0], 85) * p[4]
            right_rise = np.percentile(t[t > 0], 85) * p[5]
            if min(left_rise, right_rise) < CONFIG["gable_min_rise_metres"]:
                continue

            score = error_score(gable_height(p, xy) - z)
            if best is None or score < best["error"]:
                best = {
                    "type": "gable", "parameters": p.tolist(), "error": score
                }

    if best:
        improvement = selected["error"] - best["error"]
        if (
            improvement >= CONFIG["gable_min_absolute_improvement_metres"]
            and improvement / max(selected["error"], 1e-6)
                >= CONFIG["gable_min_relative_improvement"]
            and best["error"] <= CONFIG["gable_max_error_metres"]
        ):
            selected = best

    selected.update({
        "flat_error": flat_error,
        "plane_error": plane_error,
        "gable_error": best["error"] if best else None,
        "status": (
            "uncertain"
            if selected["error"] > CONFIG["uncertain_error_metres"]
            else "accepted"
        ),
    })
    return selected


def roof_height(fit, xy):
    if fit["type"] == "flat":
        return np.full(len(xy), fit["parameters"][0])
    if fit["type"] == "shed":
        c, gx, gy = fit["parameters"]
        return c + gx*xy[:, 0] + gy*xy[:, 1]
    return gable_height(fit["parameters"], xy)


# ============================================================================
# MATERIALS AND TEXTURED GEOMETRY
# ============================================================================

class Materials:
    """Shared definitions, numeric IDs, and cached GLB textures."""

    def __init__(self, path):
        self.path = path.resolve()
        self.root = self.path.parent
        self.config = json.loads(self.path.read_text(encoding="utf-8"))
        self.table = self.config["materials"]
        self.ids = {name: i for i, name in enumerate(sorted(self.table))}
        self.images = {}

        for surface in ("wall", "roof"):
            if self.config["default"][surface] not in self.table:
                raise ValueError(f"Unknown default {surface} material.")

        for name, item in self.table.items():
            for key in ("repeat_width_metres", "repeat_height_metres"):
                if float(item.get(key, 0)) <= 0:
                    raise ValueError(f"{name}: invalid {key}")
            for key in ("texture", "viewer_texture"):
                if key not in item or not (self.root / item[key]).is_file():
                    raise ValueError(f"{name}: missing {key} file.")

        for identifier, override in self.config.get("buildings", {}).items():
            for surface in ("wall", "roof"):
                if surface in override and override[surface] not in self.table:
                    raise ValueError(f"{identifier}: unknown {surface} material")

    def choose(self, identifier, surface):
        return self.config.get("buildings", {}).get(identifier, {}).get(
            surface, self.config["default"][surface]
        )

    def image(self, name):
        if name not in self.images:
            with Image.open(self.root / self.table[name]["texture"]) as image:
                self.images[name] = image.convert("RGBA").copy()
        return self.images[name]

    def export(self, output):
        """Copy TGA textures and write the simple runtime material index."""
        (output / "textures").mkdir()
        with (output / "building_materials.txt").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            handle.write(f"BUILDING_MATERIALS_V1\n{len(self.ids)}\n")
            for name, identifier in self.ids.items():
                relative = f"textures/material_{identifier:04d}.tga"
                shutil.copy2(
                    self.root / self.table[name]["viewer_texture"],
                    output / relative,
                )
                handle.write(f"{identifier}\n{name}\n{relative}\n")

        exported = json.loads(json.dumps(self.config))
        for name, item in exported["materials"].items():
            identifier = self.ids[name]
            png = f"textures/material_{identifier:04d}.png"
            shutil.copy2(self.root / item["texture"], output / png)
            item["texture"] = png
            item["viewer_texture"] = f"textures/material_{identifier:04d}.tga"

        (output / "building_materials.json").write_text(
            json.dumps(exported, indent=2), encoding="utf-8"
        )


def ridge_parts(polygon, fit, origin):
    """Split polygons on the ridge to preserve the crease during triangulation."""
    if fit["type"] != "gable":
        return [polygon]

    theta, offset = fit["parameters"][:2]
    direction = np.array([math.cos(theta), math.sin(theta)])
    across = np.array([-math.sin(theta), math.cos(theta)])
    center = origin + across * offset
    extent = math.hypot(
        polygon.bounds[2] - polygon.bounds[0],
        polygon.bounds[3] - polygon.bounds[1],
    ) * 4 + abs(offset) + 100
    return list(polygon_parts(split(polygon, LineString([
        center - direction*extent, center + direction*extent
    ]))))


def lifted_triangles(polygon, height_function, upward):
    xy, faces = trimesh.creation.triangulate_polygon(polygon, engine="earcut")
    triangles = np.column_stack((xy, height_function(xy)))[faces].copy()
    for triangle in triangles:
        nz = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])[2]
        if (nz > 0) != upward:
            triangle[[1, 2]] = triangle[[2, 1]]
    return triangles.reshape(-1, 3)


def roof_uv(xyz, fit, origin, material):
    """Map along-ridge U and down-slope V in physical metres."""
    xy = xyz[:, :2] - origin
    p = fit["parameters"]

    if fit["type"] == "gable":
        theta, offset, _, along, left, right = p
        s = xy @ np.array([math.cos(theta), math.sin(theta)])
        t = xy @ np.array([-math.sin(theta), math.cos(theta)]) - offset
        u = s * math.sqrt(1 + along*along)
        v = np.abs(t) * np.sqrt(
            1 + np.where(t < 0, left, right)**2
        )
    elif fit["type"] == "shed" and math.hypot(p[1], p[2]) > 1e-10:
        magnitude = math.hypot(p[1], p[2])
        uphill = np.array(p[1:3]) / magnitude
        u = xy @ np.array([-uphill[1], uphill[0]])
        v = -(xy @ uphill) * math.sqrt(1 + magnitude*magnitude)
    else:
        u, v = xy[:, 0], xy[:, 1]

    return np.column_stack((
        u / material["repeat_width_metres"],
        v / material["repeat_height_metres"],
    ))


def packed_vertices(xyz, uv, anchor):
    """Pack local position, flat normal, and UV into eight floats per vertex."""
    xyz = np.asarray(xyz).reshape(-1, 3)
    local = np.column_stack((
        xyz[:, 0]-anchor[0], xyz[:, 2]-anchor[2], anchor[1]-xyz[:, 1]
    ))
    triangles = local.reshape(-1, 3, 3)
    normals = np.cross(
        triangles[:, 1]-triangles[:, 0],
        triangles[:, 2]-triangles[:, 0],
    )
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-10

    return np.column_stack((
        triangles[valid].reshape(-1, 3),
        np.repeat(normals[valid] / lengths[valid, None], 3, axis=0),
        np.asarray(uv).reshape(-1, 3, 2)[valid].reshape(-1, 2),
    )).astype(np.float32)


def make_mesh(building, fit, ground, materials):
    """Generate walls, roof, and floor with material IDs and UVs."""
    origin = building["origin"]
    base = ground - CONFIG["wall_below_ground_metres"]
    anchor = [float(origin[0]), float(origin[1]), float(base)]
    wall_name = materials.choose(building["id"], "wall")
    roof_name = materials.choose(building["id"], "roof")
    wall_material = materials.table[wall_name]
    roof_material = materials.table[roof_name]
    walls, wall_uvs, roofs, roof_uvs, floors = [], [], [], [], []

    def heights(xy):
        return roof_height(fit, xy-origin)

    roof_geometry = building["footprint"]
    if fit["type"] == "gable" and CONFIG["gable_overhang_metres"] > 0:
        roof_geometry = make_valid(roof_geometry.buffer(
            CONFIG["gable_overhang_metres"], join_style=2
        ))

    for polygon in polygon_parts(roof_geometry):
        for part in ridge_parts(polygon, fit, origin):
            xyz = lifted_triangles(part, heights, True)
            roofs.append(xyz)
            roof_uvs.append(roof_uv(xyz, fit, origin, roof_material))

    for original in polygon_parts(building["footprint"]):
        polygon = orient(original, sign=1.0)
        floors.append(lifted_triangles(
            polygon, lambda xy: np.full(len(xy), base), False
        ))

        for ring in [polygon.exterior, *polygon.interiors]:
            coordinates = list(ring.coords)
            for index in range(len(coordinates)-1):
                a = np.asarray(coordinates[index][:2])
                b = np.asarray(coordinates[index+1][:2])
                if np.linalg.norm(b-a) < 1e-10:
                    continue
                segments = [a, b]

                if fit["type"] == "gable":
                    theta, offset = fit["parameters"][:2]
                    across = np.array([-math.sin(theta), math.cos(theta)])
                    ta, tb = (a-origin) @ across-offset, (b-origin) @ across-offset
                    if ta*tb < 0:
                        segments = [a, a+(b-a)*ta/(ta-tb), b]

                for first, second in zip(segments[:-1], segments[1:]):
                    top = heights(np.array([first, second]))
                    xyz = np.array([
                        [*first, base], [*second, base], [*second, top[1]],
                        [*first, base], [*second, top[1]], [*first, top[0]],
                    ])
                    ua = np.linalg.norm(first-a) / wall_material["repeat_width_metres"]
                    ub = np.linalg.norm(second-a) / wall_material["repeat_width_metres"]
                    vb = (base-ground) / wall_material["repeat_height_metres"]
                    va_top = (top[0]-ground) / wall_material["repeat_height_metres"]
                    vb_top = (top[1]-ground) / wall_material["repeat_height_metres"]

                    walls.append(xyz)
                    wall_uvs.append(np.array([
                        [ua,vb], [ub,vb], [ub,vb_top],
                        [ua,vb], [ub,vb_top], [ua,va_top],
                    ]))

    if not walls or not roofs or not floors:
        raise ValueError("Could not construct complete building geometry.")

    floor = np.concatenate(floors)
    groups = [
        {
            "category": WALL, "material_name": wall_name,
            "material_id": materials.ids[wall_name],
            "vertices": packed_vertices(
                np.concatenate(walls), np.concatenate(wall_uvs), anchor
            ),
        },
        {
            "category": ROOF, "material_name": roof_name,
            "material_id": materials.ids[roof_name],
            "vertices": packed_vertices(
                np.concatenate(roofs), np.concatenate(roof_uvs), anchor
            ),
        },
        {
            "category": GROUND, "material_name": None,
            "material_id": NO_MATERIAL,
            "vertices": packed_vertices(
                floor, np.zeros((len(floor), 2)), anchor
            ),
        },
    ]
    return anchor, groups


# ============================================================================
# OUTPUT
# ============================================================================

def write_bld(path, models):
    """Write BLD2: IDs, anchors, bounds, material groups, and textured vertices."""
    with path.open("wb") as handle:
        handle.write(b"BLD2")
        handle.write(struct.pack("<I", len(models)))
        for model in models:
            identifier = model["id"].encode("utf-8")
            if len(identifier) > 65535:
                raise ValueError("Building ID too long.")
            positions = np.concatenate([
                group["vertices"][:, :3] for group in model["groups"]
            ])
            handle.write(struct.pack("<H", len(identifier)))
            handle.write(identifier)
            handle.write(struct.pack("<3d", *model["anchor"]))
            handle.write(struct.pack(
                "<6f", *positions.min(axis=0), *positions.max(axis=0)
            ))
            handle.write(struct.pack("<I", len(model["groups"])))
            for group in model["groups"]:
                vertices = group["vertices"]
                handle.write(struct.pack(
                    "<3I", group["category"], group["material_id"], len(vertices)
                ))
                handle.write(vertices.astype("<f4").tobytes())


def write_glb(path, tile, models, materials):
    """Export embedded textures and UVs with a local tile origin."""
    scene = trimesh.Scene()
    xmin, _, _, ymax = tile["bounds"]
    for model in models:
        ax, ay, az = model["anchor"]
        translation = np.array([ax-xmin, az, ymax-ay])
        for number, group in enumerate(model["groups"]):
            data = group["vertices"]
            mesh = trimesh.Trimesh(
                vertices=data[:, :3].astype(float)+translation,
                faces=np.arange(len(data)).reshape(-1, 3),
                vertex_normals=data[:, 3:6].astype(float),
                process=False,
            )
            name = group["material_name"]
            if name:
                mesh.visual = TextureVisuals(
                    uv=data[:, 6:8],
                    material=PBRMaterial(
                        name=name,
                        baseColorTexture=materials.image(name),
                        baseColorFactor=[1.,1.,1.,1.],
                        metallicFactor=0.,
                        roughnessFactor=.9,
                        doubleSided=True,
                    ),
                )
            else:
                mesh.visual.face_colors = [115,110,100,255]

            name = f"{model['id']}_{number}"
            scene.add_geometry(mesh, node_name=name, geom_name=name)
    path.write_bytes(scene.export(file_type="glb"))


def write_format(output):
    (output / "BUILDING_FORMAT.txt").write_text(
        """BLD2, little endian

Header:
  char[4] BLD2
  uint32 building count

Per building:
  uint16 UTF-8 ID length, followed by ID bytes
  float64 anchor easting, northing, elevation
  float32 local minimum X,Y,Z
  float32 local maximum X,Y,Z
  uint32 group count

Per group:
  uint32 category: 1 wall, 2 roof, 3 floor
  uint32 material ID; 0xFFFFFFFF means untextured
  uint32 vertex count, divisible by three

Per vertex:
  float32 position X,Y,Z
  float32 normal X,Y,Z
  float32 U,V

Local axes: X east, Y up, Z south.
Textures repeat; material IDs appear in building_materials.txt.
Buildings may extend beyond owner-tile boundaries.
""", encoding="utf-8"
    )

def estimate_ground(building, ground, ground_tree):
    """
    Estimate ground from nearby class-2 observations.

    If none are available, use the owner terrain tile's minimum elevation.
    That last-resort estimate can be inaccurate on hilly tiles and is
    explicitly recorded in the report.
    """
    if ground_tree is not None and len(ground):
        distances, indexes = ground_tree.query(
            building["origin"],
            k=min(CONFIG["ground_neighbors"], len(ground)),
            distance_upper_bound=CONFIG["ground_context_metres"],
        )

        distances = np.atleast_1d(distances)
        indexes = np.atleast_1d(indexes)
        valid = np.isfinite(distances) & (indexes < len(ground))

        if np.any(valid):
            elevation = float(np.median(ground[indexes[valid], 2]))
            return elevation, "nearby_class_2"

    return float(building["tile"]["zmin"]), "terrain_tile_minimum"


def fallback_roof(ground_height, reason):
    """
    Create a one-floor flat-roof model when measured reconstruction cannot
    be trusted. Roof elevation is measured above estimated ground, not
    above the intentionally buried wall base.
    """
    height = (
        CONFIG["fallback_floor_height_metres"]
        * CONFIG["fallback_floors"]
    )

    return {
        "type": "flat",
        "parameters": [float(ground_height + height)],
        "error": None,
        "flat_error": None,
        "plane_error": None,
        "gable_error": None,
        "status": reason,
    }

def update_selection_attributes(gpkg_path, reports):
    """
    Add roof-fitting results to selected_buildings.gpkg.

    Matches reports to footprints using building_id, not feature order.
    Missing numeric results remain NULL, including fallback fit errors.
    The separate CSV output is unchanged.
    """
    dataset = ogr.Open(str(gpkg_path), 1)  # 1 = update mode
    if dataset is None:
        raise ValueError(
            f"Cannot open selection GeoPackage for updating: {gpkg_path}"
        )

    layer = dataset.GetLayerByName("buildings")
    if layer is None:
        raise ValueError("Selection GeoPackage has no 'buildings' layer.")

    # Report attributes and their appropriate GeoPackage field types.
    field_types = {
        "roof_points": ogr.OFTInteger64,
        "status": ogr.OFTString,
        "roof_type": ogr.OFTString,
        "fit_error_m": ogr.OFTReal,
        "flat_error_m": ogr.OFTReal,
        "single_plane_error_m": ogr.OFTReal,
        "gable_error_m": ogr.OFTReal,
        "ground_elevation_m": ogr.OFTReal,
        "ground_source": ogr.OFTString,
        "parameters": ogr.OFTString,
        "message": ogr.OFTString,
    }

    # Create missing fields. Existing fields are retained.
    for name, field_type in field_types.items():
        if layer.GetLayerDefn().GetFieldIndex(name) < 0:
            field = ogr.FieldDefn(name, field_type)

            if layer.CreateField(field) != 0:
                raise RuntimeError(
                    f"Could not create GeoPackage field: {name}"
                )

    # Validate report IDs so duplicate entries do not overwrite silently.
    reports_by_id = {}

    for report in reports:
        identifier = str(report["building_id"])

        if identifier in reports_by_id:
            raise ValueError(
                f"Duplicate roof-fit report for {identifier}"
            )

        reports_by_id[identifier] = report

    updated_count = 0
    unmatched_count = 0

    dataset.StartTransaction()

    try:
        layer.ResetReading()

        for feature in layer:
            identifier = str(feature.GetField("building_id"))
            report = reports_by_id.get(identifier)

            if report is None:
                unmatched_count += 1
                continue

            for name, field_type in field_types.items():
                field_index = feature.GetFieldIndex(name)
                value = report.get(name)

                if value is None:
                    # For example, fallback roofs have no measured fit error.
                    feature.SetFieldNull(field_index)

                elif field_type == ogr.OFTInteger64:
                    feature.SetField(field_index, int(value))

                elif field_type == ogr.OFTReal:
                    numeric = float(value)

                    if math.isfinite(numeric):
                        feature.SetField(field_index, numeric)
                    else:
                        feature.SetFieldNull(field_index)

                else:
                    feature.SetField(field_index, str(value))

            if layer.SetFeature(feature) != 0:
                raise RuntimeError(
                    f"Could not update attributes for {identifier}"
                )

            updated_count += 1

        dataset.CommitTransaction()

    except Exception:
        dataset.RollbackTransaction()
        raise

    finally:
        layer = None
        dataset = None

    print(
        f"Added roof-fit attributes to {updated_count} "
        "selected building footprints."
    )

    if unmatched_count:
        print(
            f"WARNING: {unmatched_count} selected buildings "
            "had no matching report."
        )


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate textured LiDAR buildings.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--buildings")
    selection.add_argument("--selection")
    parser.add_argument("--layer", default="rakennus")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--downloads", required=True)
    parser.add_argument("--index")
    parser.add_argument("--terrain", required=True)
    parser.add_argument("--materials", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--glb", action="store_true")
    parser.add_argument("--assume-epsg-3067", action="store_true")
    args = parser.parse_args()

    if args.count < 0:
        parser.error("--count must be zero or positive.")

    ogr.UseExceptions()
    osr.UseExceptions()
    output = Path(args.output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("Output folder must be empty or not yet exist.")

    reuse = bool(args.selection)
    buildings = read_buildings(
        Path(args.selection or args.buildings),
        "buildings" if reuse else args.layer,
        args.count, reuse,
    )
    materials = Materials(Path(args.materials))
    terrain_origin, tiles = read_terrain(Path(args.terrain))
    tile_tree = STRtree([box(*tile["bounds"]) for tile in tiles])

    # Assign ownership before processing; buildings outside viewer coverage
    # are recorded rather than assigned to an unrelated tile.
    reports, batches = [], defaultdict(list)
    for building in buildings:
        tile = owner_tile(building["origin"], tiles, tile_tree)
        if tile is None:
            reports.append({
                "building_id": building["id"],
                "status": "skipped_outside_terrain",
            })
            continue
        building["tile"] = tile
        cell = CONFIG["processing_cell_metres"]
        key = tuple(np.floor(building["origin"]/cell).astype(int))
        batches[key].append(building)

    if not batches:
        raise ValueError("No selected buildings fall inside terrain coverage.")

    overall_region = union_all([
        b["footprint"] for batch in batches.values() for b in batch
    ]).buffer(CONFIG["ground_context_metres"])

    downloads_folder = Path(args.downloads).resolve()
    index_path = (
        Path(args.index).resolve() if args.index
        else downloads_folder / "_index" / "2008_latest.shp"
    )
    downloads = downloaded_files(downloads_folder)
    print("Reading acquisition footprints...")
    coverages = read_coverages(index_path, overall_region)

    output.mkdir(parents=True)
    (output / "tiles").mkdir()
    if args.glb:
        (output / "glb").mkdir()
    save_selection(output / "selected_buildings.gpkg", buildings)
    materials.export(output)

    # Groups are written one batch at a time into tile fragments and merged
    # later, keeping the full city's mesh data out of RAM.
    import tempfile
    fragments = defaultdict(list)
    manifest_buildings = defaultdict(list)

    with tempfile.TemporaryDirectory(prefix="_building_work_", dir=output) as tmp:
        temporary = Path(tmp)

        for batch_number, (_, batch) in enumerate(sorted(batches.items()), 1):
            print(f"Batch {batch_number}/{len(batches)}: {len(batch)} buildings")
            region = union_all([b["footprint"] for b in batch]).buffer(
                CONFIG["ground_context_metres"]
            )
            sources, coverage = owned_sources(coverages, region, downloads)
            uncovered = region.difference(coverage).area
            if uncovered > .01:
                print(f"  WARNING: {uncovered:.1f} m² lacks indexed coverage.")

            roofs, ground = read_batch_points(
                batch, sources, region, args.assume_epsg_3067
            )
            ground_tree = cKDTree(ground[:, :2]) if len(ground) else None
            models_by_tile = defaultdict(list)

            for building, points in zip(batch, roofs):
                report = {
                    "building_id": building["id"],
                    "roof_points": len(points),
                }

                try:
                    # Estimate a base even when roof observations are sparse.
                    ground_height, ground_source = estimate_ground(
                        building, ground, ground_tree
                    )

                    report["ground_elevation_m"] = ground_height
                    report["ground_source"] = ground_source

                    if len(points) < CONFIG["minimum_roof_points"]:
                        # Do not remove the building: give it an explicitly
                        # approximate one-floor, flat-roof representation.
                        fit = fallback_roof(
                            ground_height,
                            "fallback_insufficient_roof_points",
                        )

                    else:
                        try:
                            fit = fit_roof(points, building["origin"])

                            predicted = roof_height(
                                fit,
                                points[:, :2] - building["origin"],
                            )

                            if (
                                not np.isfinite(predicted).all()
                                or np.percentile(predicted, 10)
                                   < ground_height + 0.5
                            ):
                                fit = fallback_roof(
                                    ground_height,
                                    "fallback_roof_ground_inconsistent",
                                )

                        except Exception as error:
                            # A roof-fitting failure should not erase the
                            # footprint either.
                            report["message"] = (
                                f"Roof fitting failed: {error}"
                            )
                            fit = fallback_roof(
                                ground_height,
                                "fallback_roof_fit_failed",
                            )

                    anchor, groups = make_mesh(
                        building,
                        fit,
                        ground_height,
                        materials,
                    )

                    model = {
                        "id": building["id"],
                        "anchor": anchor,
                        "groups": groups,
                        "roof_type": fit["type"],
                        "status": fit["status"],
                    }

                    tile = building["tile"]
                    models_by_tile[tile["name"]].append(model)

                    manifest_buildings[tile["name"]].append({
                        "id": model["id"],
                        "anchor_easting_northing_elevation": anchor,
                        "roof_type": fit["type"],
                        "status": fit["status"],
                        "ground_source": ground_source,
                        "wall_material": materials.choose(
                            model["id"], "wall"
                        ),
                        "roof_material": materials.choose(
                            model["id"], "roof"
                        ),
                    })

                    report.update({
                        "status": fit["status"],
                        "roof_type": fit["type"],
                        "fit_error_m": fit["error"],
                        "flat_error_m": fit["flat_error"],
                        "single_plane_error_m": fit["plane_error"],
                        "gable_error_m": fit["gable_error"],
                        "parameters": json.dumps(fit["parameters"]),
                    })

                    error_text = (
                        f"{fit['error']:.3f} m"
                        if fit["error"] is not None
                        else "not measured"
                    )

                    print(
                        f"  {building['id']}: {fit['type']}, "
                        f"error {error_text}, {fit['status']}"
                    )

                except Exception as error:
                    # Geometry generation/export failures are still reported
                    # as failures. They are not roof-point insufficiency.
                    report["status"] = "failed_geometry"
                    report["message"] = str(error)

                    print(
                        f"  ERROR {building['id']}: {error}"
                    )

                reports.append(report)


            # Temporary mesh fragments are local trusted files generated here.
            import pickle
            for tile_name, models in models_by_tile.items():
                fragment = temporary / f"{batch_number}_{tile_name}.pickle"
                with fragment.open("wb") as handle:
                    pickle.dump(models, handle, protocol=pickle.HIGHEST_PROTOCOL)
                fragments[tile_name].append(fragment)

        tile_lookup = {tile["name"]: tile for tile in tiles}
        manifest_tiles = []

        for tile_name in sorted(fragments):
            import pickle
            models = []
            for fragment in fragments[tile_name]:
                with fragment.open("rb") as handle:
                    models.extend(pickle.load(handle))

            tile = tile_lookup[tile_name]
            filename = f"tiles/{tile_name}.bld"
            write_bld(output / filename, models)
            glb_file = None
            if args.glb:
                glb_file = f"glb/{tile_name}.glb"
                write_glb(output / glb_file, tile, models, materials)

            manifest_tiles.append({
                "terrain_file": tile["file"],
                "building_file": filename,
                "glb_file": glb_file,
                "glb_origin_easting_northing_elevation": [
                    tile["bounds"][0], tile["bounds"][3], 0.
                ],
                "building_count": len(models),
                "buildings": manifest_buildings[tile_name],
            })

    with (output / "building_index.txt").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(f"BUILDING_INDEX_V1\n{len(manifest_tiles)}\n")
        for tile in manifest_tiles:
            handle.write(Path(tile["terrain_file"]).name+"\n")
            handle.write(tile["building_file"]+"\n")

    fields = [
        "building_id",
        "roof_points",
        "status",
        "roof_type",
        "fit_error_m",
        "flat_error_m",
        "single_plane_error_m",
        "gable_error_m",
        "ground_elevation_m",
        "ground_source",
        "parameters",
        "message",
    ]

    with (output / "roof_fit_report.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(reports)
        
    # Also attach the report to the saved footprint attribute table.
    # The CSV remains available as a separate output.
    update_selection_attributes(
        output / "selected_buildings.gpkg",
        reports,
    )


    manifest = {
        "format": "building_manifest_v1",
        "geometry_format": "BLD2",
        "crs": "EPSG:3067",
        "terrain_origin": terrain_origin,
        "selection_file": "selected_buildings.gpkg",
        "source_materials": str(materials.path),
        "config": CONFIG,
        "tiles": manifest_tiles,
        "notes": [
            "Roofer is not used.",
            "Complex buildings use one best-fit simple roof; check uncertain cases.",
            "Ground elevation is estimated from nearby class-2 points.",
            "Window layout is repeating texture mapping, not architectural inference.",
            "Gable overhangs have no soffit or fascia thickness.",
            "Same-year overlapping point files may contribute together.",
            "No automatic building LOD hierarchy is generated yet.",
            "C++ viewer requires a BLD2 loader and material renderer.",
        ],
    }
    (output / "building_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    write_format(output)

    exported = sum(tile["building_count"] for tile in manifest_tiles)
    print(f"\nFinished: {exported}/{len(buildings)} buildings exported.")
    print(f"Output: {output}")
    print("Inspect roof_fit_report.csv and textured GLBs before a full run.")


if __name__ == "__main__":
    main()
