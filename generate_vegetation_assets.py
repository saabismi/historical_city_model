#!/usr/bin/env python3
"""
Generate stylized Finnish vegetation billboard assets automatically.

Usage:
    python generate_vegetation.py -o vegetation_assets
    python generate_vegetation.py -o vegetation_assets \
        --variants 3 --size 1024 --overwrite

Arguments:
    -o, --output-dir  Output asset folder.
    --variants        Variants per species; default 3.
    --size            Near texture size; default 512, power of two.
    --seed            Repeatable random seed; default 12345.
    --overwrite       Allow replacement of generated files.

Each asset contains:
    near_front.png, near_side.png
    medium_front.png, medium_side.png
    far.png
    preview.png       Shows the intended world-space proportions.
    asset.json       Dimensions, pivot, and LOD definitions.

vegetation.json indexes all generated assets.

These are procedural visual approximations, not botanical reconstructions.
Textures use alpha-tested transparency. Viewer integration is separate.

Requires: Pillow, NumPy.
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# Tree heights and widths are measured in metres.
# Width is derived from height * width_ratio.
#
# All structure coordinates below use actual metre units. This keeps branch
# lengths, crown widths, and trunk thickness consistent between projections.
SPECIES = {
    "birch": {
        "height": (23, 30),
        "width_ratio": (0.28, 0.36),
        "crown_bottom": (0.32, 0.40),
        "form": "birch",
        "foliage": (88, 132, 57),
        "bark": (205, 207, 192),
        "trunk_diameter": (0.30, 0.55),
        "branches": 27,
    },
    "aspen": {
        "height": (24, 32),
        "width_ratio": (0.23, 0.31),
        "crown_bottom": (0.40, 0.48),
        "form": "aspen",
        "foliage": (90, 137, 65),
        "bark": (155, 164, 145),
        "trunk_diameter": (0.35, 0.60),
        "branches": 27,
    },
    "alder": {
        "height": (17, 24),
        "width_ratio": (0.32, 0.42),
        "crown_bottom": (0.29, 0.38),
        "form": "alder",
        "foliage": (61, 109, 49),
        "bark": (97, 91, 76),
        "trunk_diameter": (0.28, 0.48),
        "branches": 28,
    },
    "pine": {
        "height": (24, 32),
        "width_ratio": (0.22, 0.30),
        "crown_bottom": (0.60, 0.69),
        "form": "pine",
        "foliage": (55, 95, 66),
        "bark": (139, 93, 59),
        "trunk_diameter": (0.40, 0.70),
        "branches": 23,
    },
    "spruce": {
        "height": (25, 34),
        "width_ratio": (0.17, 0.23),
        "crown_bottom": (0.17, 0.25),
        "form": "spruce",
        "foliage": (40, 82, 55),
        "bark": (102, 88, 72),
        "trunk_diameter": (0.35, 0.60),
        "branches": 25,
    },
    "willow_shrub": {
        "height": (2.2, 4.8),
        "width_ratio": (0.95, 1.30),
        "crown_bottom": (0.08, 0.15),
        "form": "shrub",
        "foliage": (101, 134, 76),
        "bark": (111, 99, 76),
        "trunk_diameter": (0.03, 0.07),
        "branches": 22,
    },
    "leafy_shrub": {
        "height": (1.2, 2.7),
        "width_ratio": (1.10, 1.45),
        "crown_bottom": (0.06, 0.12),
        "form": "shrub",
        "foliage": (69, 116, 52),
        "bark": (102, 86, 64),
        "trunk_diameter": (0.02, 0.05),
        "branches": 20,
    },
}


def interpolate(a, b, fraction):
    """Interpolate between two 3D points."""
    return tuple(
        a[i] + (b[i] - a[i]) * fraction
        for i in range(3)
    )


def tint(base, multiplier):
    """Scale an RGB color without exceeding the byte range."""
    return tuple(
        max(0, min(255, round(channel * multiplier)))
        for channel in base
    )


def add_branch(branches, points, thickness):
    """Store a tapered polyline representing one branch."""
    branches.append({
        "points": points,
        "thickness": thickness,
    })


def add_foliage(clusters, position, rx, ry, rz, rng):
    """Store one ellipsoidal foliage clump shared by both rendered views."""
    clusters.append({
        "position": position,
        "radii": (rx, ry, rz),
        "seed": rng.randrange(2**31),
    })


def crown_radius(form, fraction, radius):
    """Return crown radius at a relative height within the crown."""
    if form == "spruce":
        # A slender pointed crown. Maximum width occurs above its base,
        # rather than producing a broad triangular skirt at ground level.
        return radius * (
            max(0.025, 1.0 - fraction) ** 0.85
        ) * (0.78 + 0.22 * min(1.0, fraction / 0.15))

    if form == "pine":
        # Rounded, compact upper crown on a tall clear trunk.
        return radius * math.sqrt(
            max(0.03, 1 - ((fraction - 0.45) / 0.63)**2)
        )

    if form == "aspen":
        return radius * math.sqrt(
            max(0.025, 1 - ((fraction - 0.48) / 0.56)**2)
        )

    return radius * math.sqrt(
        max(0.03, 1 - ((fraction - 0.48) / 0.62)**2)
    )


def make_structure(spec, seed):
    """Generate a trunk, branches, and foliage clumps in metre coordinates."""
    rng = random.Random(seed)

    height = rng.uniform(*spec["height"])
    width = height * rng.uniform(*spec["width_ratio"])
    radius = width / 2
    crown_base = height * rng.uniform(*spec["crown_bottom"])
    trunk_diameter = rng.uniform(*spec["trunk_diameter"])
    form = spec["form"]

    branches = []
    clusters = []

    # A small trunk lean avoids perfectly straight, identical silhouettes.
    lean_x = rng.uniform(-0.015, 0.015) * height
    lean_y = rng.uniform(-0.015, 0.015) * height

    def stem(z):
        fraction = z / height
        return (
            lean_x * fraction**1.5,
            lean_y * fraction**1.5,
            z,
        )

    # The main stem tapers upward.
    trunk_points = [
        stem(height * fraction)
        for fraction in (0, 0.25, 0.50, 0.75, 1.0)
    ]
    add_branch(branches, trunk_points, trunk_diameter)

    if form == "spruce":
        # Branch whorls give spruce a recognizable layered silhouette.
        for layer in range(spec["branches"]):
            fraction = layer / (spec["branches"] - 1)
            z = crown_base + fraction * (height - crown_base)
            reach = crown_radius(form, fraction, radius)

            rotation = rng.uniform(0, math.tau)
            arms = rng.randint(4, 6)

            for arm in range(arms):
                angle = rotation + arm * math.tau / arms
                length = reach * rng.uniform(0.82, 1.08)
                start = stem(z)

                middle = (
                    start[0] + math.cos(angle) * length * 0.58,
                    start[1] + math.sin(angle) * length * 0.58,
                    z - height * 0.015,
                )
                end = (
                    start[0] + math.cos(angle) * length,
                    start[1] + math.sin(angle) * length,
                    z - height * rng.uniform(0.015, 0.035),
                )
                add_branch(
                    branches, [start, middle, end],
                    trunk_diameter * 0.13,
                )

                for step in range(1, 5):
                    position = interpolate(start, end, step / 4)
                    size = max(height * 0.008, length * 0.15)
                    add_foliage(
                        clusters, position,
                        size, size, height * 0.014, rng,
                    )

        # Narrow leader at the crown tip.
        add_foliage(
            clusters, stem(height * 0.98),
            height * 0.009, height * 0.009,
            height * 0.025, rng,
        )

    else:
        for number in range(spec["branches"]):
            fraction = (number + rng.uniform(0.1, 0.9)) / spec["branches"]
            z = crown_base + fraction * (height - crown_base)
            reach = crown_radius(form, fraction, radius)

            # Golden-angle spacing distributes branches without rigid symmetry.
            angle = number * 2.39996 + rng.uniform(-0.35, 0.35)
            start = stem(z)

            if form == "shrub":
                start = (
                    rng.uniform(-0.15, 0.15) * width,
                    rng.uniform(-0.15, 0.15) * width,
                    height * 0.03,
                )

            length = reach * rng.uniform(0.72, 1.05)
            end_z = min(height * 0.98, z + height * 0.035)

            end = (
                stem(z)[0] + math.cos(angle) * length,
                stem(z)[1] + math.sin(angle) * length,
                end_z,
            )
            middle = interpolate(start, end, 0.55)
            add_branch(
                branches, [start, middle, end],
                trunk_diameter * (0.22 if form == "pine" else 0.13),
            )

            # Pine foliage is concentrated near branch ends.
            steps = (0.65, 0.85, 1.0) if form == "pine" else (0.45, 0.72, 1.0)

            for step in steps:
                position = interpolate(start, end, step)
                size = width * rng.uniform(0.055, 0.095)

                add_foliage(
                    clusters, position,
                    size,
                    size,
                    size * (0.45 if form == "pine" else 0.90),
                    rng,
                )

            if form == "birch":
                # Slender hanging sprays on outer birch branches.
                hanging_length = height * rng.uniform(0.055, 0.12)
                hanging_end = (
                    end[0] * 1.025,
                    end[1] * 1.025,
                    max(height * 0.16, end[2] - hanging_length),
                )
                add_branch(
                    branches, [end, hanging_end],
                    trunk_diameter * 0.035,
                )

                for step in (0.20, 0.45, 0.70, 0.95):
                    size = width * rng.uniform(0.025, 0.042)
                    add_foliage(
                        clusters,
                        interpolate(end, hanging_end, step),
                        size, size, size * 1.3, rng,
                    )

        # Fill the central crown without making it a completely solid blob.
        for _ in range(spec["branches"]):
            fraction = rng.uniform(0.12, 0.96)
            z = crown_base + fraction * (height - crown_base)
            reach = crown_radius(form, fraction, radius)
            angle = rng.uniform(0, math.tau)
            distance = reach * rng.uniform(0.10, 0.55)
            size = width * rng.uniform(0.045, 0.075)

            add_foliage(
                clusters,
                (
                    stem(z)[0] + math.cos(angle) * distance,
                    stem(z)[1] + math.sin(angle) * distance,
                    z,
                ),
                size, size,
                size * (0.55 if form == "pine" else 1.0),
                rng,
            )

    # Card dimensions encompass all branch/foliage extents with padding.
    # Use one square horizontal envelope for consistent front/side scale.
    horizontal_extent = width / 2
    top_extent = height

    for cluster in clusters:
        x, y, z = cluster["position"]
        rx, ry, rz = cluster["radii"]
        horizontal_extent = max(
            horizontal_extent, abs(x) + rx, abs(y) + ry
        )
        top_extent = max(top_extent, z + rz)

    card_width = horizontal_extent * 2 / 0.92
    card_height = top_extent / 0.94

    return {
        "branches": branches,
        "clusters": clusters,
        "height": height,
        "card_width": card_width,
        "card_height": card_height,
        "trunk_diameter": trunk_diameter,
    }


def render_view(species, spec, structure, size, angle):
    """Render an orthographic RGBA view with consistent world-space scale."""
    factor = 2
    pixels = size * factor

    image = Image.new("RGBA", (pixels, pixels), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    card_width = structure["card_width"]
    card_height = structure["card_height"]
    cosine, sine = math.cos(angle), math.sin(angle)

    def project(position):
        x, y, z = position
        horizontal = x * cosine + y * sine
        return (
            (0.5 + horizontal / card_width) * pixels,
            (0.98 - z / card_height) * pixels,
        )

    # Draw tapered trunk and branch segments before foliage.
    for branch in structure["branches"]:
        points = branch["points"]

        for index in range(len(points) - 1):
            fraction = index / max(1, len(points) - 1)
            thickness = branch["thickness"] * (1 - fraction * 0.80)
            pixel_width = max(1, round(thickness / card_width * pixels))

            color = spec["bark"]
            if species == "pine":
                # Scots pine's upper trunk is often warmer/orange.
                relative_height = points[index][2] / structure["height"]
                color = (
                    (163, 111, 70) if relative_height > 0.45
                    else (105, 86, 66)
                )

            draw.line(
                [project(points[index]), project(points[index + 1])],
                fill=color + (255,),
                width=pixel_width,
            )

    if species == "birch":
        bark_rng = random.Random(919)
        for _ in range(38):
            z = bark_rng.uniform(0.03, 0.80) * structure["height"]
            x, y = project((0, 0, z))
            half = structure["trunk_diameter"] / card_width * pixels * 0.35
            draw.line(
                [(x - half, y), (x + half, y + 1)],
                fill=(72, 75, 67, 255),
                width=max(1, round(pixels * 0.0015)),
            )

    # Sort clumps by depth so the front/side render uses the same plant.
    ordered = sorted(
        structure["clusters"],
        key=lambda cluster: (
            -cluster["position"][0] * sine
            + cluster["position"][1] * cosine
        ),
    )

    for cluster in ordered:
        rng = random.Random(cluster["seed"])
        center_x, center_y = project(cluster["position"])
        rx, ry, rz = cluster["radii"]

        projected_radius = math.sqrt(
            (rx * cosine)**2 + (ry * sine)**2
        )
        horizontal_pixels = projected_radius / card_width * pixels
        vertical_pixels = rz / card_height * pixels

        # Dense irregular patches suggest foliage without drawing individual
        # botanically accurate leaves or needles.
        patch_count = 48 if spec["form"] == "spruce" else 60

        for _ in range(patch_count):
            theta = rng.uniform(0, math.tau)
            distance = math.sqrt(rng.random())

            px = center_x + math.cos(theta) * distance * horizontal_pixels
            py = center_y + math.sin(theta) * distance * vertical_pixels

            patch_width = horizontal_pixels * rng.uniform(0.10, 0.23)
            patch_height = vertical_pixels * rng.uniform(0.12, 0.30)

            # Restrained variation avoids a strongly baked sun direction.
            color = tint(spec["foliage"], rng.uniform(0.78, 1.18))

            draw.ellipse(
                (
                    px - max(1, patch_width),
                    py - max(1, patch_height),
                    px + max(1, patch_width),
                    py + max(1, patch_height),
                ),
                fill=color + (255,),
            )

    return image.resize((size, size), Image.Resampling.LANCZOS)


def bleed_edges(image, iterations=8):
    """Extend foliage RGB into transparent pixels without changing alpha."""
    array = np.array(image, dtype=np.uint8)
    rgb = array[:, :, :3].astype(np.float32)
    alpha = array[:, :, 3].copy()
    known = alpha > 0
    height, width = known.shape

    for _ in range(iterations):
        total = np.zeros_like(rgb)
        count = np.zeros((height, width), dtype=np.float32)

        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            sy = slice(max(0, -dy), min(height, height - dy))
            sx = slice(max(0, -dx), min(width, width - dx))
            ty = slice(max(0, dy), min(height, height + dy))
            tx = slice(max(0, dx), min(width, width + dx))

            valid = known[sy, sx]
            total[ty, tx] += rgb[sy, sx] * valid[:, :, None]
            count[ty, tx] += valid

        added = ~known & (count > 0)
        rgb[added] = total[added] / count[added][:, None]
        known |= added

    array[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    array[:, :, 3] = alpha
    return Image.fromarray(array)


def reduced_texture(image, size):
    """Reduce resolution and approximately preserve alpha-tested coverage."""
    resized = image.resize((size, size), Image.Resampling.LANCZOS)
    array = np.array(resized)
    alpha = array[:, :, 3].astype(np.float32)

    target_coverage = float(
        (np.array(image)[:, :, 3] >= 128).mean()
    )

    low, high = 0.1, 8.0
    for _ in range(20):
        multiplier = (low + high) / 2
        coverage = float(
            (np.minimum(alpha * multiplier, 255) >= 128).mean()
        )
        if coverage < target_coverage:
            low = multiplier
        else:
            high = multiplier

    array[:, :, 3] = np.clip(
        alpha * ((low + high) / 2), 0, 255
    ).astype(np.uint8)

    return bleed_edges(Image.fromarray(array))


def save_preview(front, side, structure, destination):
    """Show front/side cards at their intended physical aspect ratio."""
    display_height = 650
    display_width = max(
        50,
        round(
            display_height
            * structure["card_width"]
            / structure["card_height"]
        ),
    )

    preview = Image.new(
        "RGB", (display_width * 2 + 60, display_height + 40),
        (185, 197, 208),
    )

    for index, image in enumerate((front, side)):
        scaled = image.resize(
            (display_width, display_height),
            Image.Resampling.LANCZOS,
        )
        preview.paste(scaled, (20 + index * (display_width + 20), 20), scaled)

    preview.save(destination)


def generate_asset(species, spec, variant, seed, size, output):
    """Generate one asset's images and metadata."""
    asset_id = f"{species}_{variant:02d}"
    folder = output / asset_id
    folder.mkdir(parents=True, exist_ok=True)

    structure = make_structure(spec, seed)
    front = render_view(species, spec, structure, size, 0)
    side = render_view(species, spec, structure, size, math.pi / 2)

    images = {
        "near_front.png": bleed_edges(front),
        "near_side.png": bleed_edges(side),
        "medium_front.png": reduced_texture(front, size // 2),
        "medium_side.png": reduced_texture(side, size // 2),
        "far.png": reduced_texture(front, size // 4),
    }

    for name, image in images.items():
        image.save(folder / name)

    save_preview(front, side, structure, folder / "preview.png")

    asset = {
        "id": asset_id,
        "species": species,
        "plant_height_metres": round(structure["height"], 3),
        "card_height_metres": round(structure["card_height"], 3),
        "card_width_metres": round(structure["card_width"], 3),
        "pivot_uv": [0.5, 0.02],
        "alpha_cutoff": 0.5,
        "double_sided": True,
        "lods": [
            {
                "name": "near",
                "representation": "crossed_cards",
                "front": "near_front.png",
                "side": "near_side.png",
                "suggested_max_distance_metres": 60,
            },
            {
                "name": "medium",
                "representation": "crossed_cards",
                "front": "medium_front.png",
                "side": "medium_side.png",
                "suggested_max_distance_metres": 150,
            },
            {
                "name": "far",
                "representation": "axial_billboard",
                "front": "far.png",
                "suggested_max_distance_metres": 300,
            },
        ],
    }

    (folder / "asset.json").write_text(
        json.dumps(asset, indent=2), encoding="utf-8"
    )

    return {
        "id": asset_id,
        "definition": f"{asset_id}/asset.json",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate stylized mature Finnish vegetation billboards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.variants <= 100:
        parser.error("--variants must be between 1 and 100.")
    if not 128 <= args.size <= 2048 or args.size & (args.size - 1):
        parser.error("--size must be a power of two from 128 to 2048.")

    output = Path(args.output_dir).resolve()
    if output.exists() and not output.is_dir():
        parser.error("Output path is not a folder.")
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        parser.error("Output folder is not empty; use --overwrite.")

    output.mkdir(parents=True, exist_ok=True)
    assets = []

    for species_index, (species, spec) in enumerate(SPECIES.items()):
        for variant in range(1, args.variants + 1):
            seed = args.seed + species_index * 10000 + variant * 101
            print(f"Generating {species}_{variant:02d}")

            assets.append(generate_asset(
                species, spec, variant, seed, args.size, output
            ))

    library = {
        "format": "procedural_vegetation_library_v1",
        "season": "summer",
        "seed": args.seed,
        "assets": assets,
        "notes": [
            "Stylized Finnish vegetation approximations.",
            "Use card dimensions and pivot from asset.json.",
            "preview.png displays the intended card proportions.",
            "Separate texture LODs are not OpenGL mipmap chains.",
            "Overhead canopy representation is not included.",
            "World placement and viewer integration are separate.",
        ],
    }

    (output / "vegetation.json").write_text(
        json.dumps(library, indent=2), encoding="utf-8"
    )

    print(f"\nGenerated {len(assets)} assets in {output}")


if __name__ == "__main__":
    main()
