@echo off
cd /d "%~dp0"
title Sentinel-2 Waterbody Change Analysis - GEE + Offline
REM Prefer Rasterio's bundled PROJ database to avoid conflicts with PostgreSQL/PostGIS PROJ installations.
for /f "delims=" %%I in ('python -c "import rasterio,pathlib; print(pathlib.Path(rasterio.__file__).parent/'proj_data')" 2^>nul') do set "S2_PROJ_DATA=%%I"
if exist "%S2_PROJ_DATA%\proj.db" (
    set "PROJ_DATA=%S2_PROJ_DATA%"
    set "PROJ_LIB=%S2_PROJ_DATA%"
)
python gui.py
if errorlevel 1 (
    echo.
    echo The application exited with an error. Review the traceback above.
)
pause
