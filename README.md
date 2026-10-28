# historical_city_model (WIP)
** very much work in progress, not ready :D**
Set of scripts and other guide / utilities to generate a historical city model (or ingredients for it) from various geodata.

Also contains a simple legacy compatible viewer for the historical city model.

it is recommended to run the scripts in osgeo4W shell to have all the dependencies
the scripts are made using GPT 6.1 Sol (Aalto AI)

the scripts are intended to work on data that is using CRS EPSG:3067 (ETRS-TM35FIN)

script order:
merge_orthos_gpt.py
tile_orthos_gpt.py
download_oldest_pointcloud_gpt.py
make_tiled_dem_glb.py
prepare_terrain.py

install dependencies:
pip install requests
pip install "laspy[lazrs]" numpy scipy rasterio trimesh pyproj

## Todo
- integrate geo sam pipeline for making training data for a visual model
	-after having training data, go over the tiles (of orthophoto) and segment/classify the image content to things like building, forest, shrub, cliff, road, etc.
- make textures for the glb files using the segmentation
- generate trees / vegetation using the segmentation 
- generate roads as vector/polygon objects using the segmentation

## Ideas
- maybe currently standing old buildings should be used, only use segmentation results for demolished buildings

### information about the viewer "Historyviewer"
It is a C++ OpenGL viewer intended to run also on old hardware (xp era) and windows xp
before running or compiling the viewer, you should run prepare_terrain.py in /history_viewer to preprocess the terrain glb to a better and lighter format.

Set up code::blocks 10.05
create win32 gui app (frame based)
go to project-> build options -> debug
go to compiler settings -> other options and add -std=c++98
go to release -> compiler settings -> compiler flags and make sure "optimize even more (fore speedd) [-O2]" is checked
select the project name in the project build options window
go to linker settings
to "link libraries" add "gdi32" "user32" and "opengl32" if they are not present already

go to project -> set programs' arguments in e.g. debug target, enter the location where the terrain_index.txt and the lightweight optimized tiles subfolder are located inside. e.g. "E:\Github\historical_city_model\espoo1965_clouds\terrain_test"

after that, you should be able to build and run the program!