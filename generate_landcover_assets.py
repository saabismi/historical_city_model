#!/usr/bin/env python3
"""Generate lightweight procedural Nordic RGBA billboard assets.

Requirements:
    Python 3.9+ and Pillow:
        python -m pip install Pillow

Example:
    python generate_landcover_assets.py \
        --output-dir ./billboard_assets \
        --size 256 \
        --seed 12345

Outputs:
    vegetation.json: Library manifest.
    <asset_id>/asset.json: Dimensions, pivot, category, and LOD paths.
    <asset_id>/lod<level>_<view>.png: Four views at three LODs.

Rendering:
    Assets are stylized procedural billboards, not 3D projections.
    Scots pine uses spreading, predominantly horizontal crown masses.
    Birch uses small-leaved crown masses and hanging terminal foliage.
    Broadleaf trees use overlapping, irregular crown lobes.
    Norway spruce uses narrow, drooping branch tiers.
    Tree trunks continue into crowns and are covered by opaque foliage.
    Rocks have opaque angular bodies, facets, grain, cracks, and moss.
    Grass, reed, wheat, and rock rendering retain the previous algorithms.
    Sources use 2x supersampling; lower LODs reuse each source.

Compatibility:
    Existing asset IDs, JSON fields, and texture naming are preserved.
    Placement code must select tree_spruce to place spruce.
    Texture dimensions vary across LODs.
    pivot_uv assumes V=0 is the bottom of the card.

Failure behavior:
    The output directory must be empty or nonexistent.
    Failed runs may leave partial output.

Reproducibility:
    Results repeat under the same implementation and environment.
    Reordering ASSETS changes the seeds assigned to assets.
"""

import argparse
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw


# Configuration: (height range in metres, width/height range, base RGB).
# Midpoints define canonical card dimensions; placement may scale instances.
ASSETS = {
    "tree_pine": ((22, 32), (0.32, 0.42), (44, 86, 54)),
    "tree_deciduous": ((20, 30), (0.40, 0.55), (82, 132, 58)),
    "tree_mixed": ((18, 30), (0.38, 0.52), (66, 116, 56)),
    "bush_leafy": ((1.2, 2.8), (1.0, 1.5), (68, 116, 52)),
    "grass_meadow": ((0.35, 0.65), (0.8, 1.4), (118, 148, 57)),
    "grass_lawn": ((0.25, 0.45), (0.9, 1.5), (88, 142, 52)),
    "grass_exposed": ((0.2, 0.4), (0.9, 1.4), (96, 145, 56)),
    "reed_wetland": ((0.7, 1.4), (0.35, 0.7), (76, 123, 61)),
    "wheat": ((0.5, 0.9), (0.35, 0.65), (190, 166, 68)),
    "rock_moss": ((0.6, 1.8), (0.8, 1.5), (112, 115, 93)),
    "tree_birch": ((18, 26), (0.35, 0.50), (105, 151, 65)),
    "tree_spruce": ((22, 32), (0.28, 0.40), (40, 78, 51)),
}

# Resolution factors do not change physical card dimensions.
LOD_FACTORS = (1.0, 0.55, 0.28)
GROUND_V = 0.02


def asset_category(key: str) -> str:
    """Return the manifest category for an asset ID."""
    if key.startswith("rock_"):
        return "rock"
    return key.split("_", 1)[0]


def shaded_colour(colour: tuple, amount: float) -> tuple:
    """Return opaque RGBA with an additive, clamped brightness adjustment."""
    return tuple(
        max(0, min(255, round(channel + amount)))
        for channel in colour
    ) + (255,)


def draw_asset(key: str, size: int, colour: tuple, seed: int) -> Image.Image:
    """Render one Nordic asset view on a transparent RGBA card.

    Local drawing coordinates are normalized: X spans the card width;
    Y spans the artwork top to the declared ground baseline. Increasing
    Y therefore means downward, including drooping branch tips.

    Trees use continuous trunks beneath opaque, overlapping crown masses.
    Branches are drawn before foliage. Flattened pine foliage and rounded
    broadleaf foliage avoid upward-pointing triangular crown shapes.

    Args:
        key: Asset ID defined in ASSETS.
        size: Output width and height, at least 64 pixels.
        colour: Base RGB colour.
        seed: Seed for this view's private random generator.

    Returns:
        An RGBA Pillow image owned by the caller.

    Raises:
        KeyError: If key is unknown.
        ValueError: If size is below 64.
    """
    if key not in ASSETS:
        raise KeyError(f"Unknown asset ID: {key}")
    if size < 64:
        raise ValueError("Image size must be at least 64.")

    rng = random.Random(seed)
    scale = 2
    canvas_size = size * scale
    image = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    top = round(size * 0.02) * scale
    base = round((size - 1) * (1.0 - GROUND_V)) * scale
    span = base - top

    def point(x: float, y: float) -> tuple:
        """Convert normalized artwork coordinates into raster pixels."""
        return (
            round(x * (canvas_size - 1)),
            round(top + y * span),
        )

    def polygon(vertices: list, fill: tuple) -> None:
        """Draw a filled polygon using normalized vertices."""
        draw.polygon([point(x, y) for x, y in vertices], fill=fill)

    def stroke(vertices: list, fill: tuple, width: float) -> None:
        """Draw a normalized polyline with card-relative thickness."""
        draw.line(
            [point(x, y) for x, y in vertices],
            fill=fill,
            width=max(1, round(width * canvas_size)),
            joint="curve",
        )

    def patch(
        x: float,
        y: float,
        rx: float,
        ry: float,
        fill: tuple,
        needle: bool = False,
    ) -> None:
        """Draw the original irregular foliage or moss patch.

        This helper remains unchanged for rock moss and plant seed heads.
        """
        count = 12 if needle else 9
        vertices = []
        for index in range(count):
            angle = index * math.tau / count
            radius = rng.uniform(0.60 if needle else 0.78, 1.0)
            vertices.append((
                x + math.cos(angle) * rx * radius,
                y + math.sin(angle) * ry * radius,
            ))
        polygon(vertices, fill)

    def foliage(
        x: float,
        y: float,
        rx: float,
        ry: float,
        count: int,
        needle: bool = False,
    ) -> None:
        """Scatter original small foliage patches with upper-left lighting.

        Retained for the established narrow spruce branch treatment.
        """
        for _ in range(count):
            angle = rng.uniform(0, math.tau)
            radius = math.sqrt(rng.random())
            px = x + math.cos(angle) * rx * radius
            py = y + math.sin(angle) * ry * radius
            brightness = (
                rng.uniform(-20, 15)
                - (px - x) / max(rx, 0.001) * 10
                - (py - y) / max(ry, 0.001) * 8
            )
            patch(
                px,
                py,
                rng.uniform(0.012, 0.027) if needle
                else rng.uniform(0.013, 0.030),
                rng.uniform(0.009, 0.021) if needle
                else rng.uniform(0.012, 0.025),
                shaded_colour(colour, brightness),
                needle,
            )

    def trunk(birch: bool = False, pine: bool = False) -> tuple:
        """Draw a connected, slightly crooked tapered trunk and bark.

        Upper trunk details are covered later by crown foliage. Returns
        the bark colour for matching supporting branches.
        """
        bark = (
            (224, 225, 211, 255) if birch
            else (137, 91, 60, 255) if pine
            else (104, 82, 60, 255)
        )
        lean = rng.uniform(-0.008, 0.008)
        polygon(
            [
                (0.480, 1.0),
                (0.489, 0.66),
                (0.496 + lean, 0.07),
                (0.504 + lean, 0.07),
                (0.510, 0.66),
                (0.523, 1.0),
            ],
            bark,
        )
        stroke(
            [(0.486, 0.99), (0.494, 0.66), (0.498 + lean, 0.08)],
            (158, 165, 155, 255) if birch else (77, 64, 50, 255),
            0.004,
        )

        if birch:
            # Bark marks are placed on the connected trunk before leaves.
            for index in range(32):
                y = 0.10 + index * 0.028
                x = 0.50 + rng.uniform(-0.006, 0.004)
                stroke(
                    [(x - rng.uniform(0.002, 0.009), y),
                     (x + rng.uniform(0.002, 0.007), y + 0.001)],
                    (49, 52, 47, 255),
                    rng.uniform(0.002, 0.004),
                )
        elif pine:
            # Darker lower bark and fine irregular vertical fissures.
            for _ in range(16):
                x = rng.uniform(0.490, 0.509)
                y = rng.uniform(0.62, 0.94)
                stroke(
                    [(x, y), (x + rng.uniform(-0.003, 0.003),
                              min(0.99, y + rng.uniform(0.02, 0.05)))],
                    (91, 69, 51, 255),
                    0.002,
                )
        return bark

    def tree_lobe(
        x: float,
        y: float,
        rx: float,
        ry: float,
        kind: str,
        brightness: float = 0,
    ) -> None:
        """Draw an opaque scalloped crown mass with fine leaf texture.

        The boundary is rounded and irregular, never triangular. Pine
        texture is horizontally flattened; birch texture is finer.
        Surface marks are clipped to the lobe, preserving opacity.
        """
        vertices = []
        count = 32
        for index in range(count):
            angle = math.tau * index / count
            radius = rng.uniform(0.88, 1.02)
            vertices.append((
                x + math.cos(angle) * rx * radius,
                y + math.sin(angle) * ry * radius,
            ))

        polygon(vertices, shaded_colour(colour, brightness - 8))

        # Clip texture to an opaque body instead of carving random holes.
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).polygon(
            [point(px, py) for px, py in vertices], fill=255
        )
        detail = Image.new("RGBA", image.size, (0, 0, 0, 0))
        detail_draw = ImageDraw.Draw(detail)

        for _ in range(110):
            angle = rng.uniform(0, math.tau)
            radius = math.sqrt(rng.random())
            dx = math.cos(angle) * radius
            dy = math.sin(angle) * radius
            px, py = point(x + dx * rx, y + dy * ry)
            light = brightness + rng.uniform(-17, 16) - dx * 7 - dy * 9

            if kind == "pine":
                # Short horizontal needle groups, not upward spikes.
                rw = rng.uniform(0.008, 0.018) * canvas_size
                rh = rng.uniform(0.003, 0.007) * span
            elif kind == "birch":
                rw = rng.uniform(0.003, 0.007) * canvas_size
                rh = rng.uniform(0.003, 0.008) * span
            else:
                rw = rng.uniform(0.005, 0.011) * canvas_size
                rh = rng.uniform(0.004, 0.010) * span

            detail_draw.ellipse(
                (px - rw, py - rh, px + rw, py + rh),
                fill=shaded_colour(colour, light),
            )

        alpha = detail.getchannel("A")
        clipped = ImageChops.multiply(alpha, mask)
        try:
            image.paste(detail, (0, 0), clipped)
        finally:
            clipped.close()
            alpha.close()
            detail.close()
            mask.close()

    def spruce_center() -> None:
        """Cover spruce's central trunk with its established foliage core.

        The lower foliage overlaps the continuous trunk, preventing the
        disconnected crown seen when the trunk was shortened.
        """
        start_y, end_y, half_width = 0.025, 0.875, 0.035
        steps = max(8, round((end_y - start_y) / 0.025))
        left_edge = []
        right_edge = []

        for index in range(steps + 1):
            fraction = index / steps
            y = start_y + fraction * (end_y - start_y)
            taper = 0.45 + 0.55 * math.sin(math.pi * fraction) ** 0.5
            width = half_width * taper
            center_x = 0.50 + rng.uniform(-0.003, 0.003)
            left_edge.append((
                center_x - width * rng.uniform(0.85, 1.15), y
            ))
            right_edge.append((
                center_x + width * rng.uniform(0.85, 1.15), y
            ))

        polygon(
            left_edge + list(reversed(right_edge)),
            shaded_colour(colour, -12),
        )

        for index in range(steps + 1):
            fraction = index / steps
            y = start_y + fraction * (end_y - start_y)
            taper = 0.45 + 0.55 * math.sin(math.pi * fraction) ** 0.5
            width = half_width * taper
            for side in (-1, 0, 1):
                patch(
                    0.50 + side * width * 0.75,
                    y,
                    width * rng.uniform(0.65, 0.95),
                    rng.uniform(0.018, 0.030),
                    shaded_colour(colour, rng.uniform(-18, 12)),
                    True,
                )

    category = asset_category(key)

    if category == "rock":
        # Rock generation is unchanged.
        outline = [
            (0.12, 0.98),
            (0.07, rng.uniform(0.64, 0.75)),
            (rng.uniform(0.15, 0.23), 0.29),
            (0.37, rng.uniform(0.01, 0.06)),
            (0.64, rng.uniform(0.04, 0.12)),
            (0.84, 0.31),
            (0.94, 0.73),
            (0.86, 0.98),
            (0.51, 1.0),
        ]
        polygon(outline, shaded_colour(colour, -10))
        hub = (0.53, 0.52)

        polygon(
            [outline[1], outline[2], outline[3], hub],
            shaded_colour(colour, 18),
        )
        polygon(
            [outline[3], outline[4], outline[5], hub],
            shaded_colour(colour, 30),
        )
        polygon(
            [hub, outline[5], outline[6], outline[7]],
            shaded_colour(colour, -22),
        )
        polygon(
            [outline[0], outline[1], hub, outline[7], outline[8]],
            shaded_colour(colour, -2),
        )

        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).polygon(
            [point(x, y) for x, y in outline], fill=255
        )
        detail = Image.new("RGBA", image.size, (0, 0, 0, 0))
        detail_draw = ImageDraw.Draw(detail)
        for _ in range(240):
            px, py = point(
                rng.uniform(0.09, 0.92),
                rng.uniform(0.03, 0.98),
            )
            radius = rng.choice((1, 1, 2)) * scale
            detail_draw.ellipse(
                (px, py, px + radius, py + radius),
                fill=shaded_colour(colour, rng.uniform(-28, 26)),
            )

        # Only actual grain pixels replace the opaque stone surface.
        detail_alpha = detail.getchannel("A")
        grain_mask = ImageChops.multiply(detail_alpha, mask)
        try:
            image.paste(detail, (0, 0), grain_mask)
        finally:
            grain_mask.close()
            detail_alpha.close()
            detail.close()
            mask.close()

        stroke(
            [(0.35, 0.24), (0.43, 0.39), (0.41, 0.53)],
            shaded_colour(colour, -35),
            0.003,
        )
        stroke(
            [(0.65, 0.63), (0.72, 0.73), (0.70, 0.87)],
            shaded_colour(colour, -32),
            0.003,
        )

        for _ in range(45):
            patch(
                rng.uniform(0.26, 0.60),
                rng.uniform(0.68, 0.93),
                rng.uniform(0.008, 0.023),
                rng.uniform(0.006, 0.015),
                shaded_colour((82, 101, 55), rng.uniform(-15, 18)),
            )

    elif category in ("grass", "reed", "wheat"):
        # Grass, reed, wheat, and their ground attachment are unchanged.
        is_reed = category == "reed"
        is_wheat = category == "wheat"
        count = 20 if is_reed or is_wheat else 48

        for index in range(count):
            root_x = rng.uniform(0.30, 0.70)
            tip_y = 0.015 if index == 0 else rng.uniform(0.08, 0.64)
            bend = rng.uniform(-0.21, 0.21)
            if is_reed or is_wheat:
                bend *= 0.45
            tip_x = root_x + bend
            blade_colour = shaded_colour(colour, rng.uniform(-23, 20))
            width = rng.uniform(0.003, 0.008)

            polygon(
                [
                    (root_x - width, 1.0),
                    (root_x + bend * 0.18 - width, 0.78),
                    (root_x + bend * 0.58, tip_y + 0.22),
                    (tip_x, tip_y),
                    (root_x + bend * 0.58 + width, tip_y + 0.22),
                    (root_x + bend * 0.18 + width, 0.78),
                    (root_x + width, 1.0),
                ],
                blade_colour,
            )

            if is_reed or is_wheat:
                stroke(
                    [(root_x, 1.0), (tip_x, tip_y)],
                    blade_colour,
                    0.003,
                )
                for side in (-1, 1):
                    leaf_y = rng.uniform(0.50, 0.78)
                    stem_x = root_x + bend * (1.0 - leaf_y)
                    polygon(
                        [
                            (stem_x, leaf_y + 0.10),
                            (stem_x + side * 0.10, leaf_y - 0.12),
                            (stem_x + side * 0.04, leaf_y + 0.01),
                        ],
                        blade_colour,
                    )

            if is_wheat:
                for kernel in range(6):
                    y = tip_y + kernel * 0.015
                    for side in (-1, 1):
                        patch(
                            tip_x + side * 0.008,
                            y,
                            0.008,
                            0.013,
                            shaded_colour(
                                (191, 164, 82), rng.uniform(-12, 18)
                            ),
                        )
                        stroke(
                            [
                                (tip_x + side * 0.008, y),
                                (tip_x + side * 0.023, y - 0.030),
                            ],
                            (183, 157, 79, 255),
                            0.0015,
                        )
            elif is_reed or (
                key == "grass_meadow" and index % 7 == 0
            ):
                for level in range(5):
                    y = tip_y + level * 0.022
                    spread = 0.008 + level * 0.005
                    for side in (-1, 1):
                        end = (tip_x + side * spread, y + 0.012)
                        stroke(
                            [(tip_x, y), end],
                            (137, 127, 83, 255),
                            0.002,
                        )
                        patch(
                            *end, 0.004, 0.009, (153, 139, 91, 255)
                        )

    elif key == "tree_spruce":
        # Retain the successful narrow spruce silhouette and drooping tiers.
        bark = (102, 79, 58, 255)
        polygon(
            [
                (0.481, 1.0),
                (0.495, 0.04),
                (0.503, 0.04),
                (0.523, 1.0),
            ],
            bark,
        )
        stroke(
            [(0.487, 0.98), (0.498, 0.06)],
            (76, 65, 51, 255),
            0.006,
        )

        for tier in range(12):
            y = 0.08 + tier * 0.062
            fraction = tier / 11
            reach = 0.025 + 0.235 * fraction
            if tier >= 9:
                reach *= 1.0 - 0.07 * (tier - 8)

            for side in (-1, 1):
                end_x = 0.50 + side * reach * rng.uniform(0.82, 1.06)
                end_y = y + rng.uniform(0.055, 0.105)
                stroke([(0.50, y), (end_x, end_y)], bark, 0.005)
                for step in range(1, 6):
                    t = step / 5
                    x = 0.50 + (end_x - 0.50) * t
                    yy = y + (end_y - y) * t
                    foliage(x, yy, 0.030, 0.033, 12, True)

        foliage(0.50, 0.04, 0.017, 0.025, 18, True)
        spruce_center()

    elif key == "tree_pine":
        bark = trunk(pine=True)

        # Mature Scots pine: a long exposed trunk beneath a spreading crown.
        # Supporting limbs travel sideways, with tips level or slightly down.
        for y, side in ((0.74, -1), (0.66, 1)):
            stroke(
                [(0.50, y),
                 (0.50 + side * 0.08, y - 0.012),
                 (0.50 + side * 0.13, y + 0.012)],
                (92, 73, 57, 255),
                0.004,
            )

        # Each row is a broad horizontal branch mass. Central lobes overlap
        # vertically and hide the trunk without a narrow foliage strip.
        lobes = []
        rows = [
            (0.085, 0.105, 0.065),
            (0.175, 0.180, 0.075),
            (0.285, 0.255, 0.082),
            (0.405, 0.235, 0.087),
            (0.505, 0.130, 0.070),
        ]

        for index, (y, spread, depth) in enumerate(rows):
            offset = rng.uniform(-0.012, 0.012)
            center_x = 0.50 + offset
            lobes.append((center_x, y, 0.105, depth))

            if index > 0:
                for side in (-1, 1):
                    x = center_x + side * spread
                    tip_y = y + rng.uniform(0.005, 0.025)
                    stroke(
                        [(0.50, y + 0.025),
                         (0.50 + side * spread * 0.55, y),
                         (x, tip_y)],
                        bark,
                        0.008,
                    )
                    lobes.append((
                        x, tip_y,
                        rng.uniform(0.105, 0.135),
                        depth * rng.uniform(0.75, 0.95),
                    ))

        # All bark is drawn before all foliage, avoiding exposed branch
        # roots cutting across finished leaf masses.
        for x, y, rx, ry in lobes:
            tree_lobe(x, y, rx, ry, "pine", rng.uniform(-4, 5))

    else:
        birch = key == "tree_birch"
        bush = category == "bush"
        bark = trunk(birch=birch)
        kind = "birch" if birch else "broadleaf"

        if bush:
            rows = [
                (0.105, 0.080, 0.090),
                (0.250, 0.195, 0.105),
                (0.425, 0.270, 0.115),
                (0.610, 0.270, 0.115),
                (0.785, 0.205, 0.110),
                (0.925, 0.085, 0.065),
            ]
        elif birch:
            rows = [
                (0.085, 0.055, 0.075),
                (0.205, 0.145, 0.090),
                (0.355, 0.225, 0.105),
                (0.525, 0.235, 0.105),
                (0.665, 0.160, 0.090),
            ]
        else:
            # Finnish broadleaf-inspired crown, suitable for aspen/alder
            # placeholders without claiming a precise botanical species.
            rows = [
                (0.090, 0.080, 0.080),
                (0.225, 0.180, 0.105),
                (0.405, 0.265, 0.120),
                (0.595, 0.230, 0.115),
                (0.715, 0.100, 0.075),
            ]

        lobes = []
        hanging = []

        for index, (y, spread, depth) in enumerate(rows):
            center_x = 0.50 + rng.uniform(-0.009, 0.009)

            # Broad central masses overlap to conceal the entire crown trunk.
            center_radius = 0.085 if birch else 0.105
            lobes.append((center_x, y, center_radius, depth))

            if index == 0:
                continue

            for side in (-1, 1):
                x = center_x + side * spread * rng.uniform(0.94, 1.04)
                droop = rng.uniform(0.020, 0.045) if birch else (
                    rng.uniform(0.005, 0.025)
                )
                tip_y = y + droop

                stroke(
                    [(0.50, y),
                     (0.50 + side * spread * 0.55, y + droop * 0.35),
                     (x, tip_y)],
                    bark,
                    0.004 if birch else 0.006,
                )

                radius = (
                    rng.uniform(0.075, 0.095) if birch
                    else rng.uniform(0.105, 0.135)
                )
                lobes.append((x, tip_y, radius, depth * 0.85))

                # Intermediate masses cover limbs without filling the whole
                # crown into a single smooth circular silhouette.
                lobes.append((
                    center_x + side * spread * 0.52,
                    y + droop * 0.4,
                    radius * 0.90,
                    depth * 0.90,
                ))

                if birch:
                    for _ in range(2):
                        tx = x + rng.uniform(-0.025, 0.025)
                        bottom_y = tip_y + depth + rng.uniform(0.015, 0.045)
                        stroke(
                            [(tx, tip_y),
                             (tx + side * 0.010, bottom_y)],
                            (117, 115, 88, 255),
                            0.0015,
                        )
                        hanging.append((
                            tx + side * 0.010,
                            bottom_y - 0.010,
                            rng.uniform(0.015, 0.025),
                            rng.uniform(0.025, 0.040),
                        ))

        # Foliage is drawn only after the complete branch skeleton.
        for x, y, rx, ry in lobes:
            tree_lobe(x, y, rx, ry, kind, rng.uniform(-5, 5))

        # Hanging terminal leaf groups reinforce birch's downward habit.
        for x, y, rx, ry in hanging:
            tree_lobe(x, y, rx, ry, "birch", -3)

    # Smooth polygon edges and fine detail before writing the source LOD.
    try:
        return image.resize(
            (size, size),
            resample=Image.Resampling.LANCZOS,
        )
    finally:
        image.close()


def write_json(path: Path, document: dict) -> None:
    """Write indented UTF-8 JSON with a trailing newline."""
    path.write_text(
        json.dumps(document, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Validate arguments and generate textures and the library manifest.

    Each source view is rendered once and reused for all LODs. The library
    manifest is written only after every asset has been generated.
    """
    parser = argparse.ArgumentParser(
        description="Generate Nordic land-cover billboard assets."
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Destination directory; must be empty or nonexistent.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Source texture width and height, at least 64 (default: 256).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Base random seed (default: 12345).",
    )
    args = parser.parse_args()

    if args.size < 64:
        parser.error("--size must be at least 64.")

    try:
        output = Path(args.output_dir).expanduser().resolve()
        if output.exists():
            if not output.is_dir():
                parser.error(f"Output path is not a directory: {output}")
            if any(output.iterdir()):
                parser.error(f"Output directory must be empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(f"Cannot prepare output directory: {exc}")

    # Use the actual rounded raster baseline for the manifest pivot.
    baseline = round((args.size - 1) * (1.0 - GROUND_V))
    pivot_v = 1.0 - baseline / (args.size - 1)

    records = []
    print(
        f"Generating {len(ASSETS)} billboard assets in {output}...",
        flush=True,
    )

    try:
        for index, (key, configuration) in enumerate(ASSETS.items()):
            height_range, width_range, colour = configuration
            height = sum(height_range) / 2.0
            width = height * (sum(width_range) / 2.0)

            print(
                f"Asset {index + 1}/{len(ASSETS)}: {key}",
                flush=True,
            )
            asset_dir = output / key
            asset_dir.mkdir()

            # Views are procedural variations; each stays fixed across LODs.
            for view in range(4):
                source = draw_asset(
                    key,
                    args.size,
                    colour,
                    args.seed + index * 100 + view,
                )
                try:
                    for lod, factor in enumerate(LOD_FACTORS):
                        filename = asset_dir / f"lod{lod}_{view}.png"
                        resolution = max(1, round(args.size * factor))

                        if resolution == args.size:
                            source.save(filename, format="PNG")
                        else:
                            reduced = source.resize(
                                (resolution, resolution),
                                resample=Image.Resampling.LANCZOS,
                            )
                            try:
                                reduced.save(filename, format="PNG")
                            finally:
                                reduced.close()
                finally:
                    source.close()

            lods = [
                {
                    "front": f"lod{lod}_0.png",
                    "side": f"lod{lod}_1.png",
                    "third": f"lod{lod}_2.png",
                    "fourth": f"lod{lod}_3.png",
                }
                for lod in range(len(LOD_FACTORS))
            ]

            definition = {
                "id": key,
                "species": key,
                "category": asset_category(key),
                "plant_height_metres": height,
                "card_width_metres": width,
                "card_height_metres": height,
                "pivot_uv": [0.5, pivot_v],
                "lods": lods,
            }
            definition_path = asset_dir / "asset.json"
            write_json(definition_path, definition)
            records.append({
                "id": key,
                "definition": definition_path.relative_to(output).as_posix(),
            })

        write_json(
            output / "vegetation.json",
            {"format": "landcover_assets_v1", "assets": records},
        )
    except (OSError, MemoryError, OverflowError) as exc:
        parser.error(
            f"Generation failed: {exc}. "
            f"Partial output may remain in {output}; "
            "inspect or remove it before rerunning."
        )

    print(f"Generated {len(records)} assets in {output}", flush=True)


if __name__ == "__main__":
    main()
