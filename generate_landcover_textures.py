#!/usr/bin/env python3
"""Generate seamlessly tileable terrain textures and a JSON manifest.

Requirements:
    Python 3 and Pillow:
        python -m pip install Pillow

Example:
    python generate_terrain_textures.py \
        --output-dir ./terrain_textures \
        --size 512 \
        --seed 12345

Outputs:
    One RGB PNG per material in PALETTES.
    A terrain_materials.json manifest with relative texture filenames.

Tiling:
    The base pattern uses periodic value noise. Surface patches are drawn
    across opposite edges, making each texture repeat seamlessly in both
    directions. Opposite edge pixels need not be identical: they are
    neighboring samples on the repeating surface.

Reproducibility:
    Each material uses a private random generator seeded with the base seed
    plus its position in PALETTES. Reordering PALETTES changes those seeds.

Failure behavior:
    The destination must be empty or nonexistent. Files are written directly
    into it, so a failed run can leave partial output. Inspect or remove that
    output before rerunning.
"""

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw


# Material IDs map to (dark_rgb, light_rgb) colour endpoints.
# Downstream configuration must use these exact IDs.
# Entry order determines each material's seed offset.
PALETTES = {
    "grass": ((102, 145, 73), (137, 161, 85)),
    "meadow": ((119, 145, 67), (177, 166, 83)),
    "forest_floor": ((62, 91, 52), (106, 112, 62)),
    "wetland_dark": ((48, 77, 55), (89, 105, 69)),
    "exposed_rock": ((108, 108, 99), (156, 148, 120)),
    "gravel_yard": ((119, 113, 101), (164, 151, 126)),
    "agriculture": ((164, 142, 62), (208, 183, 80)),
}


def smoothstep(value: float) -> float:
    """Smooth an interpolation weight in [0, 1], with flat endpoint slopes."""
    return value * value * (3.0 - 2.0 * value)


def make_periodic_layer(
    size: int,
    cells: int,
    rng: random.Random,
) -> tuple:
    """Build a random lattice and precomputed coordinates for periodic noise.

    Args:
        size: Positive texture width and height.
        cells: Number of lattice cells across each texture axis.
        rng: Random generator used to create lattice values.

    Returns:
        A pair containing:
        - A square lattice of random values in [0, 1).
        - One (start_index, end_index, weight) tuple per pixel coordinate.

    The lattice wraps through modulo indexing. Coordinates use size, rather
    than size - 1, so the texture contains one full period without duplicating
    its endpoint sample.
    """
    lattice = [
        [rng.random() for _ in range(cells)]
        for _ in range(cells)
    ]

    coordinates = []
    for pixel in range(size):
        position = pixel * cells / size
        start = int(position)
        weight = smoothstep(position - start)
        coordinates.append(
            (start % cells, (start + 1) % cells, weight)
        )

    return lattice, coordinates


def sample_periodic_layer(layer: tuple, x: int, y: int) -> float:
    """Sample a periodic lattice using smooth bilinear interpolation."""
    lattice, coordinates = layer
    x0, x1, tx = coordinates[x]
    y0, y1, ty = coordinates[y]

    top = lattice[y0][x0] * (1.0 - tx) + lattice[y0][x1] * tx
    bottom = lattice[y1][x0] * (1.0 - tx) + lattice[y1][x1] * tx
    return top * (1.0 - ty) + bottom * ty


def draw_wrapped_patch(
    draw: ImageDraw.ImageDraw,
    size: int,
    x: int,
    y: int,
    radius: int,
    colour: tuple,
) -> None:
    """Draw an ellipse and its edge-wrapping copies on a square texture.

    Args:
        draw: Drawing interface for the destination image.
        size: Image width and height.
        x: Patch centre's horizontal coordinate, within the image.
        y: Patch centre's vertical coordinate, within the image.
        radius: Nonnegative patch radius, smaller than size.
        colour: RGB fill colour.

    Copies shifted by one texture width or height also cover corner wrapping.
    Copies entirely outside the image are skipped.
    """
    for offset_y in (-size, 0, size):
        for offset_x in (-size, 0, size):
            left = x + offset_x - radius
            top = y + offset_y - radius
            right = x + offset_x + radius
            bottom = y + offset_y + radius

            if right < 0 or bottom < 0 or left >= size or top >= size:
                continue

            draw.ellipse((left, top, right, bottom), fill=colour)


def make_texture(name: str, size: int, seed: int) -> Image.Image:
    """Create a deterministic, seamlessly repeating RGB material texture.

    Args:
        name: Material ID present in PALETTES.
        size: Texture width and height; must be at least 64 pixels.
        seed: Seed for the material's private random generator.

    Returns:
        A Pillow image. The caller owns it and should close it after use.

    Raises:
        KeyError: If the material ID is unknown.
        ValueError: If size is less than 64.
        MemoryError: If insufficient memory is available.

    Four periodic noise layers provide variation at different scales.
    Wrapped ellipses add surface detail without introducing edge seams.
    """
    if size < 64:
        raise ValueError("Texture size must be at least 64.")

    rng = random.Random(seed)
    dark, light = PALETTES[name]

    # Each frequency has an integer number of cells per tile, so every
    # layer repeats over exactly the texture's width and height.
    frequencies = (4, 8, 16, 32)
    weights = (0.50, 0.25, 0.15, 0.10)
    layers = [
        make_periodic_layer(size, cells, rng)
        for cells in frequencies
    ]

    image = Image.new("RGB", (size, size), dark)
    pixels = image.load()

    for y in range(size):
        for x in range(size):
            fraction = sum(
                weight * sample_periodic_layer(layer, x, y)
                for weight, layer in zip(weights, layers)
            )

            # Slightly increase contrast while keeping channels within
            # their palette endpoints.
            fraction = max(0.0, min(1.0, (fraction - 0.5) * 1.5 + 0.5))
            pixels[x, y] = tuple(
                round(dark[channel] * (1.0 - fraction)
                      + light[channel] * fraction)
                for channel in range(3)
            )

    draw = ImageDraw.Draw(image)
    for _ in range(size // 3):
        x = rng.randrange(size)
        y = rng.randrange(size)
        radius = rng.randint(1, max(2, size // 32))
        colour = light if rng.random() < 0.5 else dark
        draw_wrapped_patch(draw, size, x, y, radius, colour)

    return image


def main() -> None:
    """Validate command-line arguments and write textures plus their manifest.

    The manifest is written after every texture has been saved. Expected
    filesystem and resource failures are reported as command-line errors.
    Already-written files are retained for inspection.
    """
    parser = argparse.ArgumentParser(
        description="Generate seamlessly tileable land-cover textures."
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
        default=512,
        help="Width and height: power of two, at least 64 (default: 512).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Base random seed (default: 12345).",
    )
    args = parser.parse_args()

    # A positive power of two has exactly one set bit.
    if args.size < 64 or (args.size & (args.size - 1)) != 0:
        parser.error("--size must be a power of two of at least 64.")

    try:
        output = Path(args.output_dir).expanduser().resolve()

        if output.exists():
            if not output.is_dir():
                parser.error(f"Output path is not a directory: {output}")
            if any(output.iterdir()):
                parser.error(
                    f"Output directory must be empty or nonexistent: {output}"
                )

        output.mkdir(parents=True, exist_ok=True)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(f"Cannot prepare output directory: {exc}")

    records = []
    print(
        f"Generating {len(PALETTES)} terrain materials in {output}...",
        flush=True,
    )

    try:
        for index, name in enumerate(PALETTES):
            filename = f"{name}.png"
            print(
                f"Material {index + 1}/{len(PALETTES)}: {name}",
                flush=True,
            )

            image = make_texture(name, args.size, args.seed + index)
            try:
                image.save(output / filename, format="PNG")
            finally:
                image.close()

            # Texture paths are relative to the manifest's directory.
            records.append({
                "id": name,
                "texture": filename,
                "tileable": True,
            })

        manifest = {
            "format": "terrain_materials_v1",
            "materials": records,
        }
        (output / "terrain_materials.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, MemoryError, OverflowError) as exc:
        parser.error(
            f"Generation failed: {exc}. "
            f"Partial output may remain in {output}; "
            "inspect or remove it before rerunning."
        )

    print(
        f"Generated {len(records)} terrain materials in {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
