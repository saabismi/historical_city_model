
one worker
python prepare_vegetation.py --terrain espoo1965_clouds/terrain_model --downloads espoo1965_clouds --mtk MTK_2009_clipped --assets assets/veg -o Historyviewer/2008_vegetation_test

multiple workers
python prepare_vegetation_parallel.py ^
    --terrain espoo1965_clouds/terrain_model ^
    --downloads espoo1965_clouds ^
    --mtk MTK_2009_clipped ^
    --assets Historyviewer/assets/veg_billboards ^
    -o Historyviewer/2008_vegetation_4workers ^
    --workers 4

python pack_viewer_vegetation.py -i Historyviewer/2008_vegetation_4workers -o Historyviewer/2008_vegetation_viewer