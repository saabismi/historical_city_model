#!/usr/bin/env python3
"""Cut a large georeferenced orthomosaic into overlapping GeoTIFF tiles.

Why GeoTIFF tiles: every tile keeps its own CRS and affine transform, so model
predictions made on a tile can later be vectorised straight into real
coordinates (e.g. EPSG:3067) with no extra bookkeeping.

The mosaic is read window by window, so it never has to fit in RAM.

Usage
-----
    python tile_ortho.py mosaic.tif tiles/ --size 512 --overlap 64
    python tile_ortho.py mosaic.tif tiles/ --size 512 --overlap 64 --gray --min-valid 0.2

Requires: rasterio, numpy   (pip install rasterio numpy)
"""
import argparse
import csv
import os
import sys

import numpy as np
import rasterio
from rasterio.windows import Window, transform as win_transform


def starts(total, size, step):
    """Tile start offsets along one axis; the last tile is shifted back so every
    tile is full size and the whole raster is covered."""
    if total <= size:
        return [0]
    s = list(range(0, total - size + 1, step))
    if s[-1] + size < total:
        s.append(total - size)
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src', help='input georeferenced raster (GeoTIFF or anything GDAL reads)')
    ap.add_argument('out_dir', help='output folder for the tiles')
    ap.add_argument('--size', type=int, default=512, help='tile size in pixels (default 512)')
    ap.add_argument('--overlap', type=int, default=64, help='overlap between neighbouring tiles in pixels (default 64)')
    ap.add_argument('--gray', action='store_true', help='convert multi-band input to a single grey band')
    ap.add_argument('--min-valid', type=float, default=0.05,
                    help='skip tiles where less than this fraction of pixels is valid data (default 0.05)')
    args = ap.parse_args()

    if args.overlap >= args.size:
        sys.exit('overlap must be smaller than size')
    step = args.size - args.overlap
    os.makedirs(args.out_dir, exist_ok=True)

    written = skipped = 0
    rows = []
    with rasterio.open(args.src) as src:
        if src.crs is None:
            print('warning: input has no CRS; tiles will be written without georeferencing', file=sys.stderr)
        xs = starts(src.width, args.size, step)
        ys = starts(src.height, args.size, step)
        total = len(xs) * len(ys)
        print(f'{src.width} x {src.height} px, {src.count} band(s), CRS: {src.crs}')
        print(f'{len(xs)} x {len(ys)} = {total} candidate tiles (size {args.size}, overlap {args.overlap})')

        for r, y in enumerate(ys):
            for c, x in enumerate(xs):
                w = min(args.size, src.width - x)
                h = min(args.size, src.height - y)
                win = Window(x, y, w, h)

                # validity: use the dataset mask (handles nodata and alpha bands)
                valid = src.dataset_mask(window=win) > 0
                if valid.mean() < args.min_valid:
                    skipped += 1
                    continue

                data = src.read(window=win)
                if args.gray and data.shape[0] > 1:
                    bands = data[:3].astype(np.float32) if data.shape[0] >= 3 else data.astype(np.float32)
                    data = bands.mean(axis=0, keepdims=True).round().astype(src.dtypes[0])

                profile = src.profile.copy()
                profile.update(driver='GTiff', width=w, height=h, count=data.shape[0],
                               transform=win_transform(win, src.transform),
                               compress='deflate', tiled=False)
                profile.pop('blockxsize', None)
                profile.pop('blockysize', None)

                name = f'tile_r{r:04d}_c{c:04d}.tif'
                with rasterio.open(os.path.join(args.out_dir, name), 'w', **profile) as dst:
                    dst.write(data)

                b = rasterio.windows.bounds(win, src.transform)
                rows.append({'name': name, 'col_off': x, 'row_off': y, 'width': w, 'height': h,
                             'minx': b[0], 'miny': b[1], 'maxx': b[2], 'maxy': b[3]})
                written += 1
            print(f'row {r + 1}/{len(ys)} done ({written} written, {skipped} skipped)', end='\r')

    index_path = os.path.join(args.out_dir, 'tiles_index.csv')
    with open(index_path, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=['name', 'col_off', 'row_off', 'width', 'height',
                                           'minx', 'miny', 'maxx', 'maxy'])
        wr.writeheader()
        wr.writerows(rows)
    print(f'\nwrote {written} tiles, skipped {skipped}; index: {index_path}')


if __name__ == '__main__':
    main()