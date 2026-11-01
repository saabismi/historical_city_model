#!/usr/bin/env python3
"""First-pass extraction of colour-separable classes from a georeferenced peruskartta crop.

Extracted now:  water, fields, buildings (black boxes), paved roads
                (red) as centrelines, contour lines (brown), ditches (blue lines).
Not yet:        gardens (dotted fields; the 'garden' test below does not work reliably yet),
                small white buildings, narrow paved/unpaved roads, paths, cliff polygons,
                swamp, forest symbols, text/OCR, contour heights, slope ticks (viettoviiva).

Usage:  python map_extract_v1.py map.tif out.gpkg [preview.png]
Needs:  numpy opencv-python rasterio geopandas shapely scikit-image skan
All thresholds are in the CONFIG block so they can be tuned per sheet.
"""
import sys
import numpy as np
import cv2
import rasterio
from rasterio import features
from rasterio.transform import xy
import geopandas as gpd
from shapely.geometry import shape, LineString, Polygon
from shapely import affinity
from skimage.morphology import skeletonize
from skan import Skeleton, summarize

CONFIG = dict(
    black_s_max=115, black_v_max=160,          # dark, low-saturation pixels = "black"
    red_h_max=8, red_h_min=172, red_s_min=100, red_v_min=140,
    field_h=(14, 25), field_s_min=85, field_v_min=232,
    contour_h=(3, 20), contour_s_min=70, contour_v_max=228,
    blue_margin=25, blue_v_min=130,            # blue if B > R + margin
    building_area=(10, 140), building_min_solidity=0.55, building_max_aspect=3.0,
    min_field_px=150, min_water_px=250, min_line_px=8,
    road_width_by_class_m={'paved road': 6.0},  # assumed real width per class (edit me)
)

def load(path):
    with rasterio.open(path) as d:
        a = d.read()
        tr = d.transform
        crs = d.crs
    rgb = np.moveaxis(a[:3], 0, -1).copy()
    return rgb, tr, crs

def ell(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

def make_masks(rgb, c):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, S, V = [hsv[..., i].astype(int) for i in range(3)]
    R, G, B = [rgb[..., i].astype(int) for i in range(3)]
    m = {}
    m['black'] = (S < c['black_s_max']) & (V < c['black_v_max'])
    m['red'] = ((H <= c['red_h_max']) | (H >= c['red_h_min'])) & (S >= c['red_s_min']) & (V >= c['red_v_min'])
    m['field'] = (H >= c['field_h'][0]) & (H <= c['field_h'][1]) & (S >= c['field_s_min']) & (V >= c['field_v_min'])
    m['contour'] = ((H >= c['contour_h'][0]) & (H <= c['contour_h'][1]) & (S >= c['contour_s_min'])
                    & (V <= c['contour_v_max']) & ~m['red'])
    m['blue'] = (B > R + c['blue_margin']) & (V >= c['blue_v_min'])
    return {k: v.astype(np.uint8) for k, v in m.items()}

def polys_from_mask(mask, tr, min_px, simplify_px=0.8):
    out = []
    for geom, val in features.shapes(mask, mask=mask > 0, transform=tr):
        g = shape(geom)
        if g.area >= min_px * abs(tr.a * tr.e):
            out.append(g.simplify(simplify_px * abs(tr.a)))
    return out

def lines_from_mask(mask, tr, min_px):
    """Skeletonise a thin-line mask and return LineStrings in map coordinates."""
    sk = skeletonize(mask > 0)
    if sk.sum() == 0:
        return [], sk
    skel = Skeleton(sk)
    lines = []
    for i in range(skel.n_paths):
        coords = skel.path_coordinates(i)           # (row, col) pairs
        if len(coords) < min_px:
            continue
        xs, ys = xy(tr, coords[:, 0], coords[:, 1])
        lines.append(LineString(list(zip(xs, ys))))
    return lines, sk

def make_gdf(attrs, geoms, crs):
    """GeoDataFrame from a list of geometries with the same attributes on every row."""
    rows = [dict(attrs, geometry=g) for g in geoms]
    if not rows:
        return gpd.GeoDataFrame(columns=list(attrs) + ['geometry'], geometry='geometry', crs=crs)
    return gpd.GeoDataFrame(rows, geometry='geometry', crs=crs)

def main(src, out_gpkg, preview=None):
    c = CONFIG
    rgb, tr, crs = load(src)
    px = abs(tr.a)
    m = make_masks(rgb, c)
    layers = {}

    # --- water (blue area): large blobs only; thin blue lines are ditches
    water = cv2.morphologyEx(m['blue'], cv2.MORPH_OPEN, ell(5))
    water = cv2.morphologyEx(water, cv2.MORPH_CLOSE, ell(7))
    water_polys = polys_from_mask(water, tr, c['min_water_px'])
    layers['water'] = make_gdf({'class': 'water', 'confidence': 'medium'}, water_polys, crs)

    # --- ditches: blue pixels that are not part of water, thinned to lines
    ditch_mask = (m['blue'] & ~cv2.dilate(water, ell(5))).astype(np.uint8)
    ditch_mask = cv2.morphologyEx(ditch_mask, cv2.MORPH_CLOSE, ell(3))
    d_lines, _ = lines_from_mask(ditch_mask, tr, c['min_line_px'])
    layers['ditches'] = make_gdf({'class': 'ditch', 'confidence': 'low'}, d_lines, crs)

    # --- fields (yellow) and gardens (yellow with black dots)
    field = cv2.morphologyEx(m['field'], cv2.MORPH_OPEN, ell(5))
    field = cv2.morphologyEx(field, cv2.MORPH_CLOSE, ell(9))     # closes outline lines and small text
    f_polys = polys_from_mask(field, tr, c['min_field_px'])
    f_rows = []
    for g in f_polys:
        mask = features.rasterize([g], out_shape=m['black'].shape, transform=tr).astype(bool)
        inner = cv2.erode(mask.astype(np.uint8), ell(5)).astype(bool)
        dots = m['black'][inner].mean() if inner.sum() else 0.0
        f_rows.append({'class': 'garden' if dots > 0.06 else 'field', 'black_fraction': round(float(dots), 3),
                       'confidence': 'medium', 'geometry': g})
    layers['fields'] = gpd.GeoDataFrame(f_rows, geometry='geometry', crs=crs) if f_rows else gpd.GeoDataFrame(columns=['geometry'], crs=crs)

    # --- buildings: compact black blobs (opening removes 1-px lines and thin text)
    near_blue = cv2.dilate(m['blue'], ell(7))            # dark blue outlines of water/ditches are not buildings
    blk = cv2.morphologyEx(m['black'] & (near_blue == 0), cv2.MORPH_OPEN, ell(3))
    n, lab, st, _ = cv2.connectedComponentsWithStats(blk, connectivity=8)
    b_rows = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if not (c['building_area'][0] <= area <= c['building_area'][1]):
            continue
        solidity = area / float(w * h)
        aspect = max(w, h) / max(1, min(w, h))
        if solidity < c['building_min_solidity'] or aspect > c['building_max_aspect']:
            continue
        cnts, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        rect = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
        box = cv2.boxPoints(rect)                          # pixel corners (col,row)
        xs, ys = xy(tr, box[:, 1] - 0.5 + 0.5, box[:, 0] - 0.5 + 0.5)
        b_rows.append({'class': 'building', 'size': 'normal', 'area_m2': round(area * px * px, 1),
                       'confidence': 'medium', 'geometry': Polygon(list(zip(xs, ys)))})
    layers['buildings'] = gpd.GeoDataFrame(b_rows, geometry='geometry', crs=crs) if b_rows else gpd.GeoDataFrame(columns=['geometry'], crs=crs)

    # --- paved roads (red): centreline + measured symbol width
    red = cv2.morphologyEx(m['red'], cv2.MORPH_CLOSE, ell(3))
    r_lines, sk = lines_from_mask(red, tr, c['min_line_px'])
    dist = cv2.distanceTransform(red, cv2.DIST_L2, 3)
    r_rows = []
    for ln in r_lines:
        # sample distance transform along the line to estimate the red fill width
        pts = [~tr * (x, y) for x, y in ln.coords]
        vals = [dist[int(round(r)), int(round(cc))] for cc, r in pts
                if 0 <= int(round(r)) < dist.shape[0] and 0 <= int(round(cc)) < dist.shape[1]]
        w_px = 2 * float(np.median(vals)) if vals else 0.0
        r_rows.append({'class': 'paved road', 'symbol_width_m': round(w_px * px, 1),
                       'width_m': c['road_width_by_class_m']['paved road'],
                       'confidence': 'medium', 'geometry': ln})
    layers['roads_centerlines'] = gpd.GeoDataFrame(r_rows, geometry='geometry', crs=crs) if r_rows else gpd.GeoDataFrame(columns=['geometry'], crs=crs)

    # --- contours (brown thin lines), not yet height-attributed
    cont = (m['contour'] & ~cv2.dilate(field, ell(5)).astype(bool)).astype(np.uint8)
    cont = cv2.morphologyEx(cont, cv2.MORPH_CLOSE, ell(3))
    c_lines, _ = lines_from_mask(cont, tr, c['min_line_px'])
    layers['contours'] = make_gdf({'class': 'contour', 'height_m': np.nan, 'confidence': 'low'}, c_lines, crs)

    for name, gdf in layers.items():
        gdf.to_file(out_gpkg, layer=name, driver='GPKG')
        print(f'{name}: {len(gdf)} features')

    if preview:
        pv = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
        pv = cv2.resize(pv, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        def draw(gdf, col, closed=True, th=1):
            for g in gdf.geometry:
                geoms = getattr(g, 'geoms', [g])
                for gg in geoms:
                    cs = list(gg.exterior.coords) if hasattr(gg, 'exterior') else list(gg.coords)
                    pts = np.array([[(~tr * (x, y))[0] * 2, (~tr * (x, y))[1] * 2] for x, y in cs], np.int32)
                    cv2.polylines(pv, [pts], hasattr(gg, 'exterior'), col, th, cv2.LINE_AA)
        draw(layers['water'], (255, 120, 0), th=2)
        draw(layers['fields'], (0, 160, 255), th=2)
        draw(layers['buildings'], (0, 0, 255), th=2)
        draw(layers['roads_centerlines'], (255, 0, 255), th=2)
        draw(layers['contours'], (0, 200, 0), th=1)
        draw(layers['ditches'], (255, 255, 0), th=1)
        cv2.imwrite(preview, pv)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
