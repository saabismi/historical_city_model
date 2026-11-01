# historical_city_model
**very much work in progress, not ready :D**

A set of scripts and other guides / utilities to generate a historical city model from various geodata (in Finland / from Finnish sources).
A historical city model differs from a "traditional" city model in that it is deliberately "outdated". It aims to portray an area as it was in a specific point in time or prior to / after specific events. The example model in this repository is Espoo in 2008 (because of data availability for that year) but my previously done historical city model depicts Tapiola and Otaniemi in "the early 2000s". That model includes some buildings that didn't exist at the same time but they all depict "a lost time".

### Some uses for a historical city model include but are not limited to: 
- Providing information to planners/citizens/etc of how an area used to be. A 3D model tells more than a 2D image even if it's not 1:1 realistic (in that it doesn't contain every little detail)
- Showcasing/studying a historical area or point in time in a museum
- Reminiscing old times and lost places.

You can use old versions of maastotietokanta (MTK) (national terrain database) if they're suitable for your area an time point but you can (try to) digitise (either by hand or computer-aidedly) old drawn/printed map sheets and generate terrain and other elements using that.

I was first trying to digitise 1960s aerial images and maps (peruskartta) for this project but then resorted to 2008 maastotietokanta for simplicity's sake, and for actually progressing in making this pipeline. The oldest aerial point cloud data provided by Maanmittauslaitos (National Land Survey of Finland) is from 2008. The oldest maastotietokanta data is from 2005) and even after that there are certain [year releases](https://ftp.csc.fi/pub/sci/geo/geodata/mml/maastotietokanta/ "year releases"). After the 2008 model can be generated with ease, I can concentrate more on making a 1960s city model.

### Historyviewer
Also contains a simple (legacy compatible (hasn't been tested on actual legacy hardware or software yet)) viewer for the historical city model, work-name is "Historyviewer"

### AI disclaimer:
The scripts are mainly made using GPT 6.1 Sol with some manual edits :) the idea of the project is completely human-made :D

### Accepted/confimed data and development environment
The scripts are intended to work on data that is using CRS EPSG:3067 (ETRS-TM35FIN)
Some scripts might give warnings if the CRS is wrong and might reproject it and output epsg:3067, but that can't be guaranteed for now.

It is recommended to run the scripts in osgeo4W shell to have all the dependencies

The development has been done on windows 11 and python scripts have been run inside the osgeo4W shell that comes with QGIS

## Try it out

### Install dependencies:
`pip install requests` (for downloading point clouds)
`pip install "laspy[lazrs]" numpy scipy rasterio trimesh pyproj`
`pip install fiona` (this might not be anymore needed, but included just in case)
`pip install numpy scipy rasterio "laspy[lazrs]" "shapely>=2.0"` (for the vegetation generator)
`pip install pillow numpy` (for vegetation generation)
~~roofer 1.0.0 (included in the directory)~~
~~add roofer to environment variable in windows cmd: set "PATH=E:\Github\historical_city_model\roofer-v1.0.0\bin;%PATH%"~~
`pip install numpy trimesh shapely mapbox-earcut` (for 3D buildings generation)

### Script order:

__optional steps:__
1. merge_orthos_gpt.py (merges orthophotos to one big file)
2. tile_orthos_gpt.py (turns a large ortho to sensibly sized tiles)

**actual steps**
1. make a orthophoto or geopackage layer that acts as the "area of interest" (the extent of that layer will be the area of interest)
2. `download_oldest_pointcloud_gpt.py` (downloads the oldest available point cloud for an area from csc/maanmittauslaitos)
3. `make_tiled_dem.py` (makes a tiled DEM from the downloaded point cloud using the specified extent (tif or gpkg), also optionally makes glb files (tiles))
4. `download_mtk.py` (download a specific year's maastotietokanta files for whole Finland)
5. `clip_mtk.py` (clips maastotietokanta layers to the extent of the orthophoto/geopackage)
6. `prepare_terrain.py` (makes a lightweight tiled terrain with LOD levels (5 by default) that can be used in "Historyviewer"
7. `generate_vegetation.py` (generate vegetation billboard photo files)
8. `prepare_vegetation.py` (prepare vegetation areas (tiles and cells) and locations for plants)
- alternative: `prepare_vegetation_parallel.py` (runs multiple workers at the same time, should be faster than the standard script on the line above)
9. `pack_viewer_vegetation.py` (pack the vegetation tiles and tree billboards to a viewer compatible format)
alternative in the future:
- ~~`generate_historical_city_model.py` (run once and output a complete city model, high possibility that it won't run all the way without problems but technically possible)~~ (this script doesn't really exist yet)
10. `generate_building_textures.py` (generates simple procedural building textures.) 
11. `generate_buildings.py` (generates buildings from maastotietokanta gpkg + MML point clouds and outputs 3D buildings with cleaned roof shapes and texture mapping for adding the textures generated by the previous script (or other textures). if not enough points are found for the roof, the building will use a fallback setting)

### Bonus utility script
- `check_coverage.py` (check coverage of already downloaded pointclouds against a tif or gpkg file, in case you want to adjust the area of generated city)

### Todo/issues
- make textures for the terrain using the segmentation/maastotietokanta objects
- generate roads as vector/polygon objects using the segmentation/MTK data
- include grass objects in vegetation
- low vegetation classified points might include power lines! fix this by adding a buffer around MTK power lines and excluding those points from vegetation
- buildings with semi realistic placeholder textures
- buildings with accurate and realistic textures

### Done
- generate a tiled terrain from 2008 point cloud data 
- generate trees / vegetation using spatial data
- generate buildings with robust placeholder textures using building footprints from MTK and pointcloud data

### Ideas
- in segmentation of old orthoimages or maps maybe currently standing old buildings should be used, only use segmentation results for demolished buildings
- integrate geo sam pipeline for making training data for a visual model
	- after having training data, go over the tiles (of orthophoto) and segment/classify the image content to things like building, forest, shrub, cliff, road, etc.

## Information about the viewer "Historyviewer"
Historyviewer is a C++ OpenGL viewer intended to run also on old hardware (xp era) and windows xp
before running or compiling the viewer, you should run prepare_terrain.py in /history_viewer to preprocess the terrain glb to a better and lighter format.

### How to build
- Set up code::blocks 10.05 (for maximum ensurement that no modern libraries etc are used accidentally)
- create win32 gui app (frame based)
- go to project-> build options -> debug
- go to compiler settings -> other options and add -std=c++98
- go to release -> compiler settings -> compiler flags and make sure "optimize even more (for speed) [-O2]" is checked
- select the project name in the project build options window
- go to linker settings
-	to "link libraries" add "gdi32" "user32" and "opengl32" if they are not present already

After that, you should be able to build and run the program! (given that you have all the correct files and city model data in the correct folder specified in `main.cpp`. After the 2008 model can be generated with ease, I can concentrate more on making a 1960s city model.