#!/usr/bin/env python3
"""
Convert Roofer CityJSONSeq buildings into tiled viewer data and optional GLBs.

Usage:
python prepare_buildings_from_roofer.py ^
    -i roofer_test\roofer_output ^
    --terrain Historyviewer\2008_terrain ^
    -o Historyviewer\2008_buildings_test ^
    --glb


Arguments:
    -i, --input      CityJSONSeq file or folder containing *.jsonl files.
    --terrain        Prepared terrain folder containing terrain_index.txt.
    -o, --output     New or empty output folder.
    --lod            Roofer geometry LoD to use; default: 2.2.
    --glb            Also export one inspection GLB per populated tile. (optional!)

Output:
    building_index.txt
    building_manifest.json
    building_materials.json
    BUILDING_FORMAT.txt
    tiles/*.bld
    glb/*.glb                 Only with --glb.

Coordinates:
    Input CityJSON scale/translation is applied.
    Binary vertices use building-local coordinates:
        X = east, Y = up, Z = south.
    Each building stores an absolute EPSG:3067 anchor in the tile file.

Important:
    - Does not perform roof cleanup.
    - A building is stored once, in the terrain tile containing its anchor.
    - Buildings without a containing terrain tile are reported and skipped.
    - Unknown surface semantics receive the default material.
    - The existing C++ viewer needs a BLD1 loader and renderer.
    - The converter expects CityJSONSeq with a header and CityJSONFeature
      records containing their own vertices.
    - GLBs are inspection models and need the manifest for georeferencing.

Requires:
    numpy, trimesh, shapely, mapbox-earcut.
"""

import argparse
import json
import math
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Point, Polygon


# Surface category values used in BLD1 files.
UNKNOWN = 0
WALL = 1
ROOF = 2
GROUND = 3

CATEGORY_NAMES = {
    UNKNOWN: "unknown",
    WALL: "wall",
    ROOF: "roof",
    GROUND: "ground",
}

# Neutral placeholder colors for GLB inspection.
COLORS = {
    UNKNOWN: [165, 165, 165, 255],
    WALL: [205, 195, 180, 255],
    ROOF: [115, 105, 100, 255],
    GROUND: [120, 115, 105, 255],
}


def read_terrain_index(folder):
    """
    Read terrain tile bounds from the existing viewer terrain index.

    The terrain index can omit water-only tiles, so matching uses the
    actual tile filenames and bounds, not array position.
    """
    filename = folder / "terrain_index.txt"

    with filename.open("r", encoding="utf-8") as handle:
        magic = handle.readline().strip()

        if magic != "TERRAIN_INDEX_V1":
            raise ValueError(f"Invalid terrain index: {filename}")

        origin_parts = handle.readline().split()
        if len(origin_parts) != 3:
            raise ValueError("Invalid terrain origin.")

        origin = [float(value) for value in origin_parts]
        count = int(handle.readline().strip())

        tiles = []

        for _ in range(count):
            parts = handle.readline().split()

            if len(parts) != 7:
                raise ValueError("Invalid terrain tile record.")

            terrain_file = parts[0]
            xmin, ymin, xmax, ymax, zmin, zmax = map(
                float, parts[1:]
            )

            tiles.append({
                "terrain_file": terrain_file,
                "name": Path(terrain_file).stem,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "zmin": zmin,
                "zmax": zmax,
            })

    return origin, tiles


def input_files(path):
    """Find CityJSONSeq files in a folder, or use one specified file."""
    if path.is_file():
        return [path]

    if not path.is_dir():
        raise ValueError(f"Input does not exist: {path}")

    files = sorted(path.rglob("*.jsonl"))

    if not files:
        raise ValueError(
            f"No .jsonl CityJSONSeq files found in: {path}"
        )

    return files


def category_from_semantics(semantics, index):
    """Map CityJSON surface semantics to the viewer's material categories."""
    if not semantics or index is None:
        return UNKNOWN

    surfaces = semantics.get("surfaces", [])

    if not isinstance(index, int) or not 0 <= index < len(surfaces):
        return UNKNOWN

    surface_type = surfaces[index].get("type", "")

    if surface_type == "RoofSurface":
        return ROOF
    if surface_type == "WallSurface":
        return WALL
    if surface_type == "GroundSurface":
        return GROUND

    return UNKNOWN


def iter_surfaces(geometry):
    """
    Yield (polygon rings, semantic category) from supported CityJSON geometry.

    Surface nesting:
      MultiSurface/CompositeSurface: surfaces
      Solid: shells -> surfaces
      MultiSolid/CompositeSolid: solids -> shells -> surfaces
    """
    geometry_type = geometry.get("type")
    boundaries = geometry.get("boundaries", [])
    semantics = geometry.get("semantics") or {}
    values = semantics.get("values")

    if geometry_type in ("MultiSurface", "CompositeSurface"):
        for surface_index, surface in enumerate(boundaries):
            semantic = (
                values[surface_index]
                if isinstance(values, list)
                and surface_index < len(values)
                else None
            )
            yield surface, category_from_semantics(semantics, semantic)

    elif geometry_type == "Solid":
        for shell_index, shell in enumerate(boundaries):
            shell_values = (
                values[shell_index]
                if isinstance(values, list)
                and shell_index < len(values)
                else None
            )

            for surface_index, surface in enumerate(shell):
                semantic = (
                    shell_values[surface_index]
                    if isinstance(shell_values, list)
                    and surface_index < len(shell_values)
                    else None
                )
                yield surface, category_from_semantics(
                    semantics, semantic
                )

    elif geometry_type in ("MultiSolid", "CompositeSolid"):
        for solid_index, solid in enumerate(boundaries):
            solid_values = (
                values[solid_index]
                if isinstance(values, list)
                and solid_index < len(values)
                else None
            )

            for shell_index, shell in enumerate(solid):
                shell_values = (
                    solid_values[shell_index]
                    if isinstance(solid_values, list)
                    and shell_index < len(solid_values)
                    else None
                )

                for surface_index, surface in enumerate(shell):
                    semantic = (
                        shell_values[surface_index]
                        if isinstance(shell_values, list)
                        and surface_index < len(shell_values)
                        else None
                    )
                    yield surface, category_from_semantics(
                        semantics, semantic
                    )

    else:
        raise ValueError(
            f"Unsupported CityJSON geometry type: {geometry_type}"
        )


def newell_normal(points):
    """Calculate a polygon's 3D normal using its exterior ring."""
    normal = np.zeros(3, dtype=np.float64)

    for index in range(len(points)):
        current = points[index]
        following = points[(index + 1) % len(points)]

        normal[0] += (
            (current[1] - following[1])
            * (current[2] + following[2])
        )
        normal[1] += (
            (current[2] - following[2])
            * (current[0] + following[0])
        )
        normal[2] += (
            (current[0] - following[0])
            * (current[1] + following[1])
        )

    length = np.linalg.norm(normal)

    if length < 1e-10:
        raise ValueError("Degenerate polygon: cannot determine its plane.")

    return normal / length


def triangulate_surface(rings, vertices):
    """
    Triangulate a 3D polygon by projecting it into its plane.

    The result retains the original 3D vertex positions and polygon
    orientation. Exterior rings and holes are supported.
    """
    if not rings:
        return []

    cleaned = []

    for ring in rings:
        indices = list(ring)

        # CityJSON rings need not repeat their first vertex.
        if len(indices) > 1 and indices[0] == indices[-1]:
            indices.pop()

        if len(indices) < 3:
            raise ValueError("Polygon ring has fewer than three vertices.")

        if min(indices) < 0 or max(indices) >= len(vertices):
            raise ValueError("Polygon references an invalid vertex index.")

        cleaned.append(indices)

    exterior_3d = vertices[cleaned[0]]
    normal = newell_normal(exterior_3d)
    plane_origin = exterior_3d[0]

    # Choose a stable in-plane basis.
    reference = np.array([1.0, 0.0, 0.0])

    if abs(np.dot(reference, normal)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])

    basis_u = np.cross(reference, normal)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(normal, basis_u)

    flat_indices = []
    projected_rings = []

    for ring in cleaned:
        ring_points = vertices[ring] - plane_origin
        projected = np.column_stack((
            ring_points @ basis_u,
            ring_points @ basis_v,
        ))

        projected_rings.append(projected)
        flat_indices.extend(ring)

    polygon = Polygon(
        projected_rings[0],
        holes=projected_rings[1:],
    )

    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
        raise ValueError(
            "Invalid projected polygon; inspect the Roofer output."
        )

    # Earcut triangulates concave polygons and holes.
    # Returned coordinates are mapped back to original vertices.
    points_2d, triangles = trimesh.creation.triangulate_polygon(
        polygon,
        engine="earcut",
    )

    original_2d = np.vstack(projected_rings)

    # Earcut ordinarily preserves input ring vertices. Map by nearest
    # coordinate and fail if the triangulator introduced unexpected points.
    mapped = []

    for point in points_2d:
        distances = np.linalg.norm(original_2d - point, axis=1)
        nearest = int(np.argmin(distances))

        if distances[nearest] > 1e-7:
            raise ValueError(
                "Triangulator returned an unexpected vertex."
            )

        mapped.append(flat_indices[nearest])

    output = []

    for triangle in triangles:
        indices = [mapped[int(i)] for i in triangle]
        a, b, c = vertices[indices]

        # Keep the original surface orientation.
        if np.dot(np.cross(b - a, c - a), normal) < 0:
            indices[1], indices[2] = indices[2], indices[1]

        if np.linalg.norm(np.cross(
            vertices[indices[1]] - vertices[indices[0]],
            vertices[indices[2]] - vertices[indices[0]],
        )) > 1e-10:
            output.append(indices)

    return output


def select_geometry(city_object, lod):
    """Select geometries matching the requested LoD."""
    result = []

    for geometry in city_object.get("geometry", []):
        if str(geometry.get("lod")) == str(lod):
            result.append(geometry)

    return result


def read_cityjsonseq(filename, lod):
    """
    Yield building records from one CityJSONSeq file.

    Roofer normally writes a CityJSON header followed by CityJSONFeature
    records. Each feature's vertices are local to that feature.
    """
    transform = None
    saw_header = False

    with filename.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue

            record = json.loads(line)
            record_type = record.get("type")

            if record_type == "CityJSON":
                saw_header = True
                transform = record.get("transform")
                continue

            if record_type != "CityJSONFeature":
                raise ValueError(
                    f"{filename}:{line_number}: unsupported record "
                    f"type {record_type!r}"
                )

            if not saw_header:
                raise ValueError(
                    f"{filename}:{line_number}: CityJSONFeature appears "
                    "before its CityJSON header."
                )

            raw_vertices = np.asarray(
                record.get("vertices", []), dtype=np.float64
            )

            if raw_vertices.ndim != 2 or raw_vertices.shape[1] != 3:
                raise ValueError(
                    f"{filename}:{line_number}: invalid vertices."
                )

            if transform:
                scale = np.asarray(
                    transform["scale"], dtype=np.float64
                )
                translate = np.asarray(
                    transform["translate"], dtype=np.float64
                )
                vertices = raw_vertices * scale + translate
            else:
                vertices = raw_vertices

            city_objects = record.get("CityObjects", {})

            for object_id, city_object in city_objects.items():
                if city_object.get("type") not in (
                    "Building", "BuildingPart"
                ):
                    continue

                geometries = select_geometry(city_object, lod)

                if not geometries:
                    continue

                triangles_by_category = defaultdict(list)

                for geometry in geometries:
                    for rings, category in iter_surfaces(geometry):
                        triangles = triangulate_surface(rings, vertices)
                        triangles_by_category[category].extend(triangles)

                if not any(triangles_by_category.values()):
                    continue

                yield {
                    "id": str(object_id),
                    "vertices": vertices,
                    "triangles": dict(triangles_by_category),
                    "source_file": str(filename),
                }


def choose_anchor(building):
    """
    Choose an anchor within a ground surface when possible.

    If no ground surface semantics are present, use the center of the
    building's XY bounding box. Elevation is the minimum referenced Z.
    """
    vertices = building["vertices"]
    used_indices = sorted({
        index
        for triangles in building["triangles"].values()
        for triangle in triangles
        for index in triangle
    })
    used = vertices[used_indices]

    ground_triangles = building["triangles"].get(GROUND, [])
    ground_polygons = []

    for triangle in ground_triangles:
        xy = vertices[triangle][:, :2]
        polygon = Polygon(xy)
        if polygon.is_valid and polygon.area > 0:
            ground_polygons.append(polygon)

    if ground_polygons:
        from shapely.ops import unary_union

        point = unary_union(ground_polygons).representative_point()
        anchor_x, anchor_y = point.x, point.y
    else:
        anchor_x = (float(used[:, 0].min()) + float(used[:, 0].max())) / 2
        anchor_y = (float(used[:, 1].min()) + float(used[:, 1].max())) / 2

    anchor_z = float(used[:, 2].min())

    return anchor_x, anchor_y, anchor_z


def find_owner_tile(anchor, tiles):
    """
    Find the terrain tile containing a building anchor.

    Half-open boundaries prevent assigning a building to two tiles.
    """
    x, y, _ = anchor

    for tile in tiles:
        if (
            tile["xmin"] <= x < tile["xmax"]
            and tile["ymin"] <= y < tile["ymax"]
        ):
            return tile

    # Include anchors exactly on the outer maximum edge.
    for tile in tiles:
        if (
            tile["xmin"] <= x <= tile["xmax"]
            and tile["ymin"] <= y <= tile["ymax"]
        ):
            return tile

    return None


def local_meshes(building, anchor):
    """
    Convert triangle positions to building-local viewer coordinates.

    Input: easting, northing, elevation.
    Output: east, up, south.
    """
    ax, ay, az = anchor
    source = building["vertices"]
    result = {}

    for category, triangles in building["triangles"].items():
        if not triangles:
            continue

        points = source[np.asarray(triangles, dtype=int)].reshape(-1, 3)

        local = np.column_stack((
            points[:, 0] - ax,
            points[:, 2] - az,
            ay - points[:, 1],
        )).astype(np.float32)

        result[category] = local

    return result


def write_bld(filename, buildings):
    """
    Write one BLD1 tile.

    Vertices are duplicated per triangle to keep the XP loader simple.
    Face category is stored per mesh group.
    """
    with filename.open("wb") as handle:
        handle.write(b"BLD1")
        handle.write(struct.pack("<I", len(buildings)))

        for building in buildings:
            identifier = building["id"].encode("utf-8")

            if len(identifier) > 65535:
                raise ValueError("Building ID is too long.")

            handle.write(struct.pack("<H", len(identifier)))
            handle.write(identifier)
            handle.write(struct.pack("<3d", *building["anchor"]))

            meshes = building["meshes"]
            handle.write(struct.pack("<I", len(meshes)))

            for category in sorted(meshes):
                vertices = meshes[category]

                handle.write(struct.pack(
                    "<II", category, len(vertices)
                ))
                handle.write(vertices.astype("<f4").tobytes())


def write_glb(filename, tile, buildings):
    """
    Write an inspection GLB with shared placeholder material colors.

    Coordinates are local to the terrain tile's northwest corner and
    elevation zero. The manifest records that origin.
    """
    scene = trimesh.Scene()

    tile_origin_x = tile["xmin"]
    tile_origin_y = tile["ymax"]

    grouped = defaultdict(list)

    for building in buildings:
        ax, ay, az = building["anchor"]

        # Translate building-local coordinates into tile-local coordinates.
        translation = np.array([
            ax - tile_origin_x,
            az,
            tile_origin_y - ay,
        ], dtype=np.float64)

        for category, vertices in building["meshes"].items():
            grouped[category].append(
                vertices.astype(np.float64) + translation
            )

    for category, arrays in grouped.items():
        vertices = np.concatenate(arrays)
        faces = np.arange(len(vertices)).reshape(-1, 3)

        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=False,
        )
        mesh.visual.face_colors = COLORS[category]
        scene.add_geometry(
            mesh,
            node_name=CATEGORY_NAMES[category],
            geom_name=CATEGORY_NAMES[category],
        )

    filename.write_bytes(scene.export(file_type="glb"))


def write_material_template(output):
    """Create defaults; later entries can override individual buildings."""
    material_config = {
        "default": {
            "wall": "wall_default",
            "roof": "roof_default",
        },
        "materials": {
            "wall_default": {
                "texture": "textures/wall_default.png"
            },
            "roof_default": {
                "texture": "textures/roof_default.png"
            },
        },
        "buildings": {},
    }

    (output / "building_materials.json").write_text(
        json.dumps(material_config, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Roofer buildings for the legacy viewer."
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="CityJSONSeq file or folder containing .jsonl files.",
    )
    parser.add_argument(
        "--terrain", required=True,
        help="Prepared terrain folder containing terrain_index.txt.",
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="New or empty output folder.",
    )
    parser.add_argument(
        "--lod", default="2.2",
        help="CityJSON geometry LoD; default: 2.2.",
    )
    parser.add_argument(
        "--glb", action="store_true",
        help="Also create inspection GLBs.",
    )
    args = parser.parse_args()

    source = Path(args.input).resolve()
    terrain = Path(args.terrain).resolve()
    output = Path(args.output).resolve()

    if output.exists() and (
        not output.is_dir() or any(output.iterdir())
    ):
        parser.error("Output folder must be empty or not yet exist.")

    files = input_files(source)
    terrain_origin, tiles = read_terrain_index(terrain)

    output.mkdir(parents=True, exist_ok=True)
    tile_dir = output / "tiles"
    tile_dir.mkdir()

    glb_dir = output / "glb"
    if args.glb:
        glb_dir.mkdir()

    grouped = defaultdict(list)
    skipped = []
    errors = []
    seen_ids = set()

    for filename in files:
        print(f"Reading: {filename}")

        # A malformed feature stops that input file rather than silently
        # producing incomplete geometry.
        try:
            for building in read_cityjsonseq(filename, args.lod):
                building_id = building["id"]

                if building_id in seen_ids:
                    raise ValueError(
                        f"Duplicate CityObject ID: {building_id}"
                    )
                seen_ids.add(building_id)

                anchor = choose_anchor(building)
                tile = find_owner_tile(anchor, tiles)

                if tile is None:
                    skipped.append({
                        "building_id": building_id,
                        "reason": "No terrain tile contains its anchor.",
                        "anchor": anchor,
                    })
                    continue

                meshes = local_meshes(building, anchor)

                if not meshes:
                    skipped.append({
                        "building_id": building_id,
                        "reason": "No usable triangles.",
                    })
                    continue

                # Bounds include the complete building, not just its anchor.
                all_vertices = np.concatenate(list(meshes.values()))
                bounds_min = all_vertices.min(axis=0).tolist()
                bounds_max = all_vertices.max(axis=0).tolist()

                grouped[tile["name"]].append({
                    "id": building_id,
                    "anchor": anchor,
                    "meshes": meshes,
                    "local_bounds_min": bounds_min,
                    "local_bounds_max": bounds_max,
                    "source_file": building["source_file"],
                })

        except Exception as error:
            errors.append({
                "file": str(filename),
                "error": str(error),
            })
            print(f"ERROR in {filename}: {error}", file=sys.stderr)

    if errors:
        raise RuntimeError(
            "One or more CityJSONSeq files failed. "
            "Output is incomplete; inspect the errors above."
        )

    tile_lookup = {tile["name"]: tile for tile in tiles}
    manifest_tiles = []

    for tile_name in sorted(grouped):
        tile = tile_lookup[tile_name]
        buildings = grouped[tile_name]

        bld_name = f"{tile_name}.bld"
        write_bld(tile_dir / bld_name, buildings)

        glb_relative = None
        if args.glb:
            glb_name = f"{tile_name}.glb"
            write_glb(glb_dir / glb_name, tile, buildings)
            glb_relative = f"glb/{glb_name}"

        manifest_tiles.append({
            "terrain_file": tile["terrain_file"],
            "building_file": f"tiles/{bld_name}",
            "glb_file": glb_relative,
            "glb_origin_easting_northing_elevation": [
                tile["xmin"], tile["ymax"], 0.0
            ],
            "building_count": len(buildings),
            "buildings": [
                {
                    "id": building["id"],
                    "anchor_easting_northing_elevation": building["anchor"],
                    "local_bounds_min": building["local_bounds_min"],
                    "local_bounds_max": building["local_bounds_max"],
                    "surface_groups": [
                        CATEGORY_NAMES[category]
                        for category in sorted(building["meshes"])
                    ],
                    "source_file": building["source_file"],
                }
                for building in buildings
            ],
        })

    with (output / "building_index.txt").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write("BUILDING_INDEX_V1\n")
        handle.write(f"{len(manifest_tiles)}\n")

        for tile in manifest_tiles:
            handle.write(Path(tile["terrain_file"]).name + "\n")
            handle.write(tile["building_file"] + "\n")

    manifest = {
        "format": "building_manifest_v1",
        "source": str(source),
        "terrain_index": str(terrain / "terrain_index.txt"),
        "terrain_origin": terrain_origin,
        "assumed_crs": "EPSG:3067",
        "lod": args.lod,
        "surface_categories": CATEGORY_NAMES,
        "tiles": manifest_tiles,
        "skipped_buildings": skipped,
        "notes": [
            "Raw Roofer geometry; roof cleanup has not been applied.",
            "GLB axes: X east, Y up, Z south.",
            "GLBs use placeholder colors, not image textures.",
            "Material assignments are a template; they are not baked "
            "or interpreted by the current viewer.",
        ],
    }

    (output / "building_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    write_material_template(output)

    (output / "BUILDING_FORMAT.txt").write_text(
        """BLD1 binary format, little endian

File:
  char[4]  magic: BLD1
  uint32   building count

For each building:
  uint16   UTF-8 ID byte length
  byte[]   UTF-8 building ID
  float64  anchor easting
  float64  anchor northing
  float64  anchor elevation
  uint32   mesh group count

For each mesh group:
  uint32   surface category:
             0 unknown, 1 wall, 2 roof, 3 ground
  uint32   vertex count, divisible by 3
  float32  XYZ for each vertex; three vertices per triangle

Local coordinates:
  X = east from anchor
  Y = up from anchor
  Z = south from anchor

Absolute placement in the existing terrain viewer:
  viewer X = anchor easting - originX + local X
  viewer Y = anchor elevation - originZ + local Y
  viewer Z = originY - anchor northing + local Z

Building meshes can extend beyond their owner tile. Culling bounds must
include the complete geometry, not just the anchor or tile rectangle.
""",
        encoding="utf-8",
    )

    count = sum(tile["building_count"] for tile in manifest_tiles)

    print(f"\nPrepared {count} buildings in {len(manifest_tiles)} tiles.")
    print(f"Skipped outside terrain: {len(skipped)}")
    print(f"Output: {output}")

    if not count:
        print(
            "WARNING: No buildings were exported. Check requested LoD, "
            "CityObject types, and terrain coverage."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
