@echo off
cd /d "%~dp0"

REM Roofer 1.0.0 starting configuration.
REM Raw output and diagnostic crop output are preserved in roofer_output.
roofer ^
    --polygon-source-layer buildings ^
    --id-attribute building_id ^
    --bld-class 6 ^
    --grnd-class 2 ^
    --no-clip-terrain ^
    --lod12 ^
    --lod13 ^
    --crop-output ^
    --index ^
    -j 4 ^
    points.laz ^
    buildings.gpkg ^
    roofer_output

pause
