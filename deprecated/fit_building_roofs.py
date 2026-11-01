#!/usr/bin/env python3
"""
Fit simple roofs directly to prepared LiDAR building points.

Usage:
    python fit_building_roofs.py ^
        --buildings roofer_test/buildings.gpkg ^
        --points roofer_test/points.laz ^
        --terrain Historyviewer/2008_terrain ^
        -o fitted_buildings --glb

Inputs:
    buildings.gpkg: polygon layer "buildings", attribute "building_id".
    points.las/laz: class 6 roof candidates, class 2 ground.
    terrain folder: terrain_index.txt.

Outputs:
    roof_fit_report.csv
    building_manifest.json
    building_index.txt
    building_materials.json
    tiles/*.bld
    glb/*.glb, with --glb

Does not read or modify Roofer output.
Uses the BLD1 format from prepare_buildings.py.
Requires GDAL, NumPy, SciPy, laspy+lazrs, Shapely 2, trimesh, mapbox-earcut.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import laspy
import numpy as np
import trimesh
from osgeo import ogr, osr
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from shapely import from_wkb, make_valid
from shapely.geometry import LineString, Polygon
from shapely.ops import split


# ============================================================================
# EDITABLE SETTINGS
# ============================================================================

CONFIG = {
    "roof_class": 6,
    "ground_class": 2,
    "minimum_roof_points": 20,
    "maximum_fit_points": 4000,

    # Robust fitting tolerance in metres.
    "robust_scale": 0.35,

    # Single-plane slope below this angle may be considered flat.
    "flat_max_slope_degrees": 5.0,
    "flat_error_allowance_metres": 0.08,

    # Gable acceptance criteria.
    "gable_min_slope_degrees": 8.0,
    "gable_max_slope_degrees": 65.0,
    "gable_min_points_per_side": 8,
    "gable_min_fraction_per_side": 0.15,
    "gable_min_rise_metres": 0.60,
    "gable_min_relative_improvement": 0.20,
    "gable_min_absolute_improvement_metres": 0.12,
    "gable_max_error_metres": 0.90,

    # Flag any chosen fit above this trimmed RMSE.
    "uncertain_error_metres": 0.90,

    # Ground estimation uses nearby class-2 points; no roof buffer is used.
    "ground_neighbors": 24,
    "ground_max_distance_metres": 60.0,

    # Extend walls slightly below estimated ground.
    "wall_below_ground_metres": 0.50,

    # Applied after fitting. Set to 0 to disable.
    "gable_overhang_metres": 0.30,

    # Broad search followed by local optimization.
    "ridge_angle_samples": 16,
    "ridge_start_positions": [-0.20, 0.0, 0.20],

    "seed": 12345,
}

WALL = 1
ROOF = 2
GROUND = 3


# ============================================================================
# INPUT
# ============================================================================

def polygon_parts(geometry):
    """Extract polygons from repaired multipart geometries."""
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    result = []
    if hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            result.extend(polygon_parts(part))
    return result


def read_buildings(path, layer_name):
    """Read project IDs and footprints; require EPSG:3067."""
    dataset = ogr.Open(str(path))
    if dataset is None:
        raise ValueError(f"Cannot open {path}")

    layer = dataset.GetLayerByName(layer_name)
    if layer is None:
        raise ValueError(f"Layer missing: {layer_name}")

    expected = osr.SpatialReference()
    expected.ImportFromEPSG(3067)
    expected.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    source = layer.GetSpatialRef()
    if source is None:
        raise ValueError("Building layer has no CRS.")

    source = source.Clone()
    source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    if not source.IsSame(expected):
        raise ValueError("Building footprints must use EPSG:3067.")

    if layer.GetLayerDefn().GetFieldIndex("building_id") < 0:
        raise ValueError("Building layer needs a building_id attribute.")

    buildings = []
    seen = set()

    for feature in layer:
        identifier = str(feature.GetField("building_id"))
        if identifier in seen:
            raise ValueError(f"Duplicate building ID: {identifier}")
        seen.add(identifier)

        geometry = feature.GetGeometryRef()
        if geometry is None:
            continue

        parts = polygon_parts(make_valid(
            from_wkb(bytes(geometry.ExportToWkb()))
        ))
        if not parts:
            continue

        from shapely import union_all
        footprint = union_all(parts)
        anchor = footprint.representative_point()

        buildings.append({
            "id": identifier,
            "footprint": footprint,
            "xy_origin": np.array([anchor.x, anchor.y]),
        })

    dataset = None
    return buildings


def read_points(paths, buildings):
    """
    Read points in chunks and assign roof points to exact footprints.

    Roof arrays are retained in memory for this test dataset. This is intended
    for the first 100 buildings, not an entire nationwide point-cloud set.
    """
    from shapely import points as shapely_points
    from shapely.strtree import STRtree

    tree = STRtree([b["footprint"] for b in buildings])
    roof_arrays = [[] for _ in buildings]
    ground_arrays = []

    xmin = min(b["footprint"].bounds[0] for b in buildings)
    ymin = min(b["footprint"].bounds[1] for b in buildings)
    xmax = max(b["footprint"].bounds[2] for b in buildings)
    ymax = max(b["footprint"].bounds[3] for b in buildings)
    margin = CONFIG["ground_max_distance_metres"]

    for path in paths:
        print(f"Reading {path}")

        with laspy.open(path) as reader:
            crs = reader.header.parse_crs()
            if crs is not None:
                from pyproj import CRS
                if not CRS(crs).equals(
                    CRS.from_epsg(3067), ignore_axis_order=True
                ):
                    raise ValueError(f"Point-cloud CRS mismatch: {path}")
            else:
                print("WARNING: No LAS CRS metadata; assuming EPSG:3067.")

            for chunk in reader.chunk_iterator(1_000_000):
                xyz = np.column_stack((
                    np.asarray(chunk.x),
                    np.asarray(chunk.y),
                    np.asarray(chunk.z),
                ))
                classes = np.asarray(chunk.classification)

                valid = (
                    np.isfinite(xyz).all(axis=1)
                    & ~np.asarray(chunk.withheld, dtype=bool)
                    & (xyz[:, 0] >= xmin - margin)
                    & (xyz[:, 0] <= xmax + margin)
                    & (xyz[:, 1] >= ymin - margin)
                    & (xyz[:, 1] <= ymax + margin)
                )

                ground = xyz[
                    valid & (classes == CONFIG["ground_class"])
                ]
                if len(ground):
                    ground_arrays.append(ground)

                roof = xyz[
                    valid & (classes == CONFIG["roof_class"])
                ]
                if not len(roof):
                    continue

                pairs = tree.query(
                    shapely_points(roof[:, 0], roof[:, 1]),
                    predicate="intersects",
                )
                for building_index in np.unique(pairs[1]):
                    selected = pairs[0][pairs[1] == building_index]
                    roof_arrays[int(building_index)].append(roof[selected])

    roofs = [
        np.unique(np.concatenate(arrays), axis=0)
        if arrays else np.empty((0, 3))
        for arrays in roof_arrays
    ]
    ground = (
        np.unique(np.concatenate(ground_arrays), axis=0)
        if ground_arrays else np.empty((0, 3))
    )

    return roofs, ground


def read_terrain(folder):
    """Read terrain origins and actual tile bounds."""
    with (folder / "terrain_index.txt").open() as handle:
        if handle.readline().strip() != "TERRAIN_INDEX_V1":
            raise ValueError("Invalid terrain_index.txt")

        origin = list(map(float, handle.readline().split()))
        count = int(handle.readline())
        tiles = []

        for _ in range(count):
            record = handle.readline().split()
            tiles.append({
                "file": record[0],
                "name": Path(record[0]).stem,
                "bounds": list(map(float, record[1:5])),
            })

    return origin, tiles


# ============================================================================
# ROOF FITTING
# ============================================================================

def fit_error(residuals):
    """Trim the largest 10% of residuals to reduce isolated-object influence."""
    squared = np.sort(np.asarray(residuals) ** 2)
    count = max(1, math.ceil(len(squared) * 0.90))
    return float(np.sqrt(squared[:count].mean()))


def gable_values(parameters, xy):
    """
    Asymmetric gable with a potentially slightly tilted ridge.

    theta: ridge direction
    offset: ridge location across its direction
    c: ridge elevation at along-ridge coordinate zero
    along: elevation gradient along the ridge
    left/right: descending cross-ridge slopes
    """
    theta, offset, c, along, left, right = parameters
    direction = np.array([math.cos(theta), math.sin(theta)])
    across = np.array([-math.sin(theta), math.cos(theta)])

    s = xy @ direction
    t = xy @ across - offset

    return (
        c + along * s
        - left * np.maximum(-t, 0)
        - right * np.maximum(t, 0)
    )


def fit_roof(points, building):
    """Fit three roof families and conservatively choose a gable."""
    rng = np.random.default_rng(CONFIG["seed"])
    fit_points = points
    if len(points) > CONFIG["maximum_fit_points"]:
        fit_points = points[rng.choice(
            len(points), CONFIG["maximum_fit_points"], replace=False
        )]

    xy = fit_points[:, :2] - building["xy_origin"]
    z = fit_points[:, 2]

    scale = CONFIG["robust_scale"]

    # Flat roof.
    flat = float(np.median(z))
    flat_error = fit_error(z - flat)

    # General single-plane roof.
    design = np.column_stack((np.ones(len(xy)), xy))
    initial = np.linalg.lstsq(design, z, rcond=None)[0]

    plane = least_squares(
        lambda p: design @ p - z,
        initial,
        loss="soft_l1",
        f_scale=scale,
    ).x

    plane_error = fit_error(design @ plane - z)
    plane_slope = math.degrees(math.atan(
        math.hypot(plane[1], plane[2])
    ))

    if (
        plane_slope <= CONFIG["flat_max_slope_degrees"]
        and flat_error <= plane_error + CONFIG["flat_error_allowance_metres"]
    ):
        selected = {"type": "flat", "parameters": [flat], "error": flat_error}
    else:
        selected = {
            "type": "shed",
            "parameters": plane.tolist(),
            "error": plane_error,
        }

    minimum_slope = math.tan(math.radians(
        CONFIG["gable_min_slope_degrees"]
    ))
    maximum_slope = math.tan(math.radians(
        CONFIG["gable_max_slope_degrees"]
    ))

    best_gable = None

    for theta in np.linspace(
        0, math.pi, CONFIG["ridge_angle_samples"], endpoint=False
    ):
        across = np.array([-math.sin(theta), math.cos(theta)])
        projected = xy @ across
        low, high = np.percentile(projected, [5, 95])
        span = high - low
        if span < 1:
            continue

        center = (low + high) / 2
        for relative_start in CONFIG["ridge_start_positions"]:
            start_offset = center + relative_start * span

            lower = [
                theta - math.pi / CONFIG["ridge_angle_samples"],
                low + span * 0.10,
                float(z.min()) - 3,
                -0.20,
                minimum_slope,
                minimum_slope,
            ]
            upper = [
                theta + math.pi / CONFIG["ridge_angle_samples"],
                high - span * 0.10,
                float(z.max()) + span * maximum_slope + 3,
                0.20,
                maximum_slope,
                maximum_slope,
            ]
            initial_gable = [
                theta,
                np.clip(start_offset, lower[1], upper[1]),
                float(np.percentile(z, 85)),
                0.0,
                0.35,
                0.35,
            ]

            result = least_squares(
                lambda p: gable_values(p, xy) - z,
                initial_gable,
                bounds=(lower, upper),
                loss="soft_l1",
                f_scale=scale,
                max_nfev=120,
            )
            if not result.success:
                continue

            p = result.x
            across_fit = np.array([-math.sin(p[0]), math.cos(p[0])])
            t = xy @ across_fit - p[1]

            # Ignore points almost exactly on the ridge for side support.
            left_count = int((t < -0.25).sum())
            right_count = int((t > 0.25).sum())
            required = max(
                CONFIG["gable_min_points_per_side"],
                math.ceil(
                    len(xy) * CONFIG["gable_min_fraction_per_side"]
                ),
            )
            if min(left_count, right_count) < required:
                continue

            # Both sides must have substantial measured ridge-to-eave rise.
            left_rise = float(np.percentile(-t[t < 0], 85) * p[4])
            right_rise = float(np.percentile(t[t > 0], 85) * p[5])
            if min(left_rise, right_rise) < CONFIG["gable_min_rise_metres"]:
                continue

            error = fit_error(gable_values(p, xy) - z)
            if best_gable is None or error < best_gable["error"]:
                best_gable = {
                    "type": "gable",
                    "parameters": p.tolist(),
                    "error": error,
                }

    if best_gable is not None:
        improvement = selected["error"] - best_gable["error"]
        relative = improvement / max(selected["error"], 1e-6)

        if (
            improvement >= CONFIG["gable_min_absolute_improvement_metres"]
            and relative >= CONFIG["gable_min_relative_improvement"]
            and best_gable["error"] <= CONFIG["gable_max_error_metres"]
        ):
            selected = best_gable

    selected["flat_error"] = flat_error
    selected["plane_error"] = plane_error
    selected["gable_error"] = (
        best_gable["error"] if best_gable else None
    )
    selected["status"] = (
        "uncertain"
        if selected["error"] > CONFIG["uncertain_error_metres"]
        else "accepted"
    )
    return selected


def roof_elevation(fit, xy):
    """Evaluate the selected roof surface at local XY coordinates."""
    p = fit["parameters"]
    if fit["type"] == "flat":
        return np.full(len(xy), p[0])
    if fit["type"] == "shed":
        return p[0] + p[1] * xy[:, 0] + p[2] * xy[:, 1]
    return gable_values(p, xy)


# ============================================================================
# MESH GENERATION
# ============================================================================

def split_at_ridge(polygon, fit, origin):
    """Split roof polygons along the ridge so triangles never bridge its crease."""
    if fit["type"] != "gable":
        return [polygon]

    theta, offset = fit["parameters"][:2]
    direction = np.array([math.cos(theta), math.sin(theta)])
    across = np.array([-math.sin(theta), math.cos(theta)])
    center = origin + across * offset

    diagonal = math.hypot(
        polygon.bounds[2] - polygon.bounds[0],
        polygon.bounds[3] - polygon.bounds[1],
    )
    extent = diagonal * 4 + abs(offset) + 100

    line = LineString([
        center - direction * extent,
        center + direction * extent,
    ])
    return polygon_parts(split(polygon, line))


def triangulated_surface(polygon, height_function, upward):
    """Triangulate an XY polygon, including holes, and lift it into 3D."""
    xy, faces = trimesh.creation.triangulate_polygon(
        polygon, engine="earcut"
    )
    z = height_function(xy)
    vertices = np.column_stack((xy, z))

    triangles = vertices[faces]
    for triangle in triangles:
        normal_z = np.cross(
            triangle[1] - triangle[0],
            triangle[2] - triangle[0],
        )[2]
        if (normal_z > 0) != upward:
            triangle[[1, 2]] = triangle[[2, 1]]

    return triangles.reshape(-1, 3)


def make_mesh(building, fit, ground):
    """
    Create roof, walls, and floor.

    Walls follow the original footprint. Optional gable overhangs affect only
    the roof surface. Thin roof-edge fascia/soffits are not modeled here.
    """
    footprint = building["footprint"]
    origin = building["xy_origin"]
    base = ground - CONFIG["wall_below_ground_metres"]

    roof_geometry = footprint
    if fit["type"] == "gable" and CONFIG["gable_overhang_metres"] > 0:
        roof_geometry = make_valid(footprint.buffer(
            CONFIG["gable_overhang_metres"], join_style=2
        ))

    roof_triangles = []
    floor_triangles = []
    wall_triangles = []

    def heights(xy):
        return roof_elevation(fit, xy - origin)

    for polygon in polygon_parts(roof_geometry):
        for part in split_at_ridge(polygon, fit, origin):
            roof_triangles.append(
                triangulated_surface(part, heights, upward=True)
            )

    for polygon in polygon_parts(footprint):
        floor_triangles.append(triangulated_surface(
            polygon,
            lambda xy: np.full(len(xy), base),
            upward=False,
        ))

        for ring in [polygon.exterior, *polygon.interiors]:
            coordinates = list(ring.coords)

            # Insert ridge crossings on footprint edges so gable wall tops
            # contain their apex instead of connecting eaves with a straight edge.
            for index in range(len(coordinates) - 1):
                a = np.asarray(coordinates[index][:2], dtype=float)
                b = np.asarray(coordinates[index + 1][:2], dtype=float)
                points = [a, b]

                if fit["type"] == "gable":
                    theta, offset = fit["parameters"][:2]
                    across = np.array([-math.sin(theta), math.cos(theta)])
                    ta = (a - origin) @ across - offset
                    tb = (b - origin) @ across - offset
                    if ta * tb < 0:
                        crossing = a + (b - a) * (ta / (ta - tb))
                        points = [a, crossing, b]

                for first, second in zip(points[:-1], points[1:]):
                    top_heights = heights(np.array([first, second]))
                    lower_a = [first[0], first[1], base]
                    lower_b = [second[0], second[1], base]
                    upper_a = [first[0], first[1], top_heights[0]]
                    upper_b = [second[0], second[1], top_heights[1]]

                    wall_triangles.extend([
                        [lower_a, lower_b, upper_b],
                        [lower_a, upper_b, upper_a],
                    ])

    meshes_absolute = {
        WALL: np.asarray(wall_triangles).reshape(-1, 3),
        ROOF: np.concatenate(roof_triangles),
        GROUND: np.concatenate(floor_triangles),
    }

    anchor = [float(origin[0]), float(origin[1]), float(base)]

    # Convert easting/northing/elevation to local east/up/south.
    meshes_local = {}
    for category, xyz in meshes_absolute.items():
        meshes_local[category] = np.column_stack((
            xyz[:, 0] - anchor[0],
            xyz[:, 2] - anchor[2],
            anchor[1] - xyz[:, 1],
        )).astype(np.float32)

    return anchor, meshes_local


# ============================================================================
# OUTPUT
# ============================================================================

def owner_tile(anchor, tiles):
    """Assign a complete building to one terrain tile by its anchor."""
    x, y = anchor[:2]
    for tile in tiles:
        xmin, ymin, xmax, ymax = tile["bounds"]
        if xmin <= x < xmax and ymin <= y < ymax:
            return tile
    return None


def write_bld(path, buildings):
    """Write the same BLD1 triangle-group format as prepare_buildings.py."""
    with path.open("wb") as handle:
        handle.write(b"BLD1")
        handle.write(__import__("struct").pack("<I", len(buildings)))

        for building in buildings:
            import struct
            identifier = building["id"].encode("utf-8")
            if len(identifier) > 65535:
                raise ValueError("Building ID is too long.")

            handle.write(struct.pack("<H", len(identifier)))
            handle.write(identifier)
            handle.write(struct.pack("<3d", *building["anchor"]))
            handle.write(struct.pack("<I", len(building["meshes"])))

            for category, vertices in sorted(building["meshes"].items()):
                handle.write(struct.pack("<II", category, len(vertices)))
                handle.write(vertices.astype("<f4").tobytes())


def write_glb(path, tile, buildings):
    """Export tile-local inspection geometry with neutral surface colors."""
    scene = trimesh.Scene()
    xmin, _, _, ymax = tile["bounds"]
    colors = {
        WALL: [205, 195, 180, 255],
        ROOF: [115, 105, 100, 255],
        GROUND: [115, 110, 100, 255],
    }

    for building in buildings:
        ax, ay, az = building["anchor"]
        translation = np.array([ax - xmin, az, ymax - ay])

        for category, vertices in building["meshes"].items():
            mesh = trimesh.Trimesh(
                vertices=vertices.astype(float) + translation,
                faces=np.arange(len(vertices)).reshape(-1, 3),
                process=False,
            )
            mesh.visual.face_colors = colors[category]
            scene.add_geometry(
                mesh,
                node_name=f"{building['id']}_{category}",
                geom_name=f"{building['id']}_{category}",
            )

    path.write_bytes(scene.export(file_type="glb"))


def main():
    parser = argparse.ArgumentParser(
        description="Fit flat, shed, and gable roofs from prepared LiDAR."
    )
    parser.add_argument("--buildings", required=True)
    parser.add_argument("--layer", default="buildings")
    parser.add_argument("--points", required=True, nargs="+")
    parser.add_argument("--terrain", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--glb", action="store_true")
    args = parser.parse_args()

    ogr.UseExceptions()
    output = Path(args.output).resolve()
    if output.exists() and (
        not output.is_dir() or any(output.iterdir())
    ):
        parser.error("Output folder must be empty or not yet exist.")

    buildings = read_buildings(Path(args.buildings), args.layer)
    terrain_origin, tiles = read_terrain(Path(args.terrain))
    roofs, ground = read_points(
        [Path(p) for p in args.points], buildings
    )

    if not len(ground):
        raise ValueError("No class-2 ground points available.")

    ground_tree = cKDTree(ground[:, :2])
    output.mkdir(parents=True, exist_ok=True)
    (output / "tiles").mkdir()
    if args.glb:
        (output / "glb").mkdir()

    grouped = {}
    reports = []

    for number, (building, points) in enumerate(zip(buildings, roofs), 1):
        print(
            f"[{number}/{len(buildings)}] {building['id']}: "
            f"{len(points)} roof points"
        )
        report = {
            "building_id": building["id"],
            "roof_points": len(points),
            "status": "",
        }

        if len(points) < CONFIG["minimum_roof_points"]:
            report["status"] = "skipped_insufficient_roof_points"
            reports.append(report)
            continue

        distances, indexes = ground_tree.query(
            building["xy_origin"],
            k=min(CONFIG["ground_neighbors"], len(ground)),
            distance_upper_bound=CONFIG["ground_max_distance_metres"],
        )
        indexes = np.atleast_1d(indexes)[np.isfinite(np.atleast_1d(distances))]
        if not len(indexes):
            report["status"] = "skipped_no_nearby_ground"
            reports.append(report)
            continue

        ground_height = float(np.median(ground[indexes, 2]))
        fit = fit_roof(points, building)

        # Reject obviously inconsistent roofs below/at estimated ground.
        test_z = roof_elevation(
            fit, points[:, :2] - building["xy_origin"]
        )
        if np.percentile(test_z, 10) < ground_height + 0.5:
            report["status"] = "skipped_roof_ground_inconsistent"
            reports.append(report)
            continue

        anchor, meshes = make_mesh(building, fit, ground_height)
        tile = owner_tile(anchor, tiles)
        if tile is None:
            report["status"] = "skipped_outside_terrain"
            reports.append(report)
            continue

        model = {
            "id": building["id"],
            "anchor": anchor,
            "meshes": meshes,
            "roof_type": fit["type"],
            "status": fit["status"],
        }
        grouped.setdefault(tile["name"], {
            "tile": tile, "buildings": []
        })["buildings"].append(model)

        report.update({
            "status": fit["status"],
            "roof_type": fit["type"],
            "fit_error_m": fit["error"],
            "flat_error_m": fit["flat_error"],
            "single_plane_error_m": fit["plane_error"],
            "gable_error_m": fit["gable_error"],
            "ground_elevation_m": ground_height,
            "parameters": json.dumps(fit["parameters"]),
        })
        reports.append(report)
        print(f"  {fit['type']}, error {fit['error']:.3f} m, {fit['status']}")

    manifest_tiles = []
    for tile_name, entry in sorted(grouped.items()):
        tile = entry["tile"]
        models = entry["buildings"]
        filename = f"{tile_name}.bld"
        write_bld(output / "tiles" / filename, models)

        if args.glb:
            write_glb(output / "glb" / f"{tile_name}.glb", tile, models)

        manifest_tiles.append({
            "terrain_file": tile["file"],
            "building_file": f"tiles/{filename}",
            "glb_file": f"glb/{tile_name}.glb" if args.glb else None,
            "glb_origin_easting_northing_elevation": [
                tile["bounds"][0], tile["bounds"][3], 0
            ],
            "buildings": [
                {
                    "id": model["id"],
                    "anchor_easting_northing_elevation": model["anchor"],
                    "roof_type": model["roof_type"],
                    "status": model["status"],
                }
                for model in models
            ],
        })

    with (output / "building_index.txt").open("w") as handle:
        handle.write("BUILDING_INDEX_V1\n")
        handle.write(f"{len(manifest_tiles)}\n")
        for tile in manifest_tiles:
            handle.write(Path(tile["terrain_file"]).name + "\n")
            handle.write(tile["building_file"] + "\n")

    fields = [
        "building_id", "roof_points", "status", "roof_type",
        "fit_error_m", "flat_error_m", "single_plane_error_m",
        "gable_error_m", "ground_elevation_m", "parameters",
    ]
    with (output / "roof_fit_report.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(reports)

    manifest = {
        "format": "building_manifest_v1",
        "crs": "EPSG:3067",
        "terrain_origin": terrain_origin,
        "config": CONFIG,
        "tiles": manifest_tiles,
        "notes": [
            "Fits original prepared points; Roofer geometry is not used.",
            "Uncertain buildings use their best simple approximation.",
            "Complex roofs, roof steps, dormers and hips are not reconstructed.",
            "Ground bases use nearby ground observations at the anchor.",
            "GLBs have placeholder colors; UV textures are not generated.",
            "Gable overhangs have no modeled soffit or fascia thickness.",
        ],
    }
    (output / "building_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    (output / "building_materials.json").write_text(
        json.dumps({
            "default": {
                "wall": "wall_default",
                "roof": "roof_default",
            },
            "materials": {
                "wall_default": {"texture": "textures/wall_default.png"},
                "roof_default": {"texture": "textures/roof_default.png"},
            },
            "buildings": {},
        }, indent=2),
        encoding="utf-8",
    )

    print(f"\nFinished: {output}")
    print("Inspect roof_fit_report.csv and the GLBs before viewer integration.")


if __name__ == "__main__":
    main()
