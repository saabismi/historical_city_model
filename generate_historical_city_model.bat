WORK IN PROGRESS! DO NOT USE! won't even run :D

there has to exist a geopackage or tif file that will be used as the extent!

python download_oldest_lidar.py -i area.gpkg

python make_tiled_dem.py -i espoo1965_clouds -o espoo2008_terrain --reference-gpkg espoo2008_area.gpkg

python download_mtk.py --year 2009

python clip_mtk.py -i mtk_2009 --extent-gpkg espoo2008_area.gpkg

python prepare_terrain.py -i espoo2008_dem_tiled -o Historyviewer/espoo2008_terrain --water MTK_2009_clipped/mtk-vakavesi_2009_clipped.gpkg

python prepare_vegetation_parallel.py ^
    --terrain espoo1965_clouds/terrain_model ^
    --downloads espoo1965_clouds ^
    --mtk MTK_2009_clipped ^
    --assets Historyviewer/assets/veg_billboards ^
    -o Historyviewer/2008_vegetation_4workers ^
    --workers 4

python pack_viewer_vegetation.py -i Historyviewer/2008_vegetation_4workers -o Historyviewer/2008_vegetation_viewer