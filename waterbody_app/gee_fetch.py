"""Google Earth Engine Sentinel-2 SR Harmonized fetcher.

Downloads independent date-window products: median B3/B8 reflectance, valid
observation count, water-occurrence frequency, and maximum observed water mask.
The maximum water mask is the union of per-scene water detections after SCL
cloud/shadow masking; it is intended for seasonal urban pond/lake footprint
screening, not as proof of a legal or permanent shoreline.
"""
from __future__ import annotations

from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from typing import Callable, Optional
import json
import math
import time
import zipfile

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from rasterio.windows import Window
from shapely.geometry import box
from pyproj import CRS

Progress = Optional[Callable[[int, str], None]]
COLLECTION_ID = "COPERNICUS/S2_SR_HARMONIZED"
TILE_PIXELS = 512
PIXEL_SIZE_M = 10
NODATA = -9999


def _report(cb: Progress, pct: int, message: str) -> None:
    if cb:
        cb(int(max(0, min(100, pct))), message)


def _read_aoi(shapefile: str):
    path = Path(shapefile)
    if not path.is_file():
        raise FileNotFoundError(f"Study boundary not found: {shapefile}")
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError("Study boundary is empty.")
    if gdf.crs is None:
        raise ValueError("Study boundary has no CRS. Define its CRS before using GEE.")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        raise ValueError("Study boundary contains no valid geometry.")
    try:
        gdf["geometry"] = gdf.geometry.make_valid()
    except Exception:
        from shapely.validation import make_valid
        gdf["geometry"] = gdf.geometry.apply(make_valid)
    gdf = gdf.to_crs("EPSG:4326")
    geom = gdf.geometry.union_all()
    if geom.is_empty:
        raise ValueError("Study boundary geometry is empty after validation.")
    return gdf, geom


def _utm_crs(gdf: gpd.GeoDataFrame) -> str:
    """Select a local UTM CRS from the study boundary centroid."""
    centroid = gdf.to_crs("EPSG:4326").geometry.union_all().centroid
    lon, lat = float(centroid.x), float(centroid.y)
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    zone = min(60, max(1, zone))
    epsg = (32600 if lat >= 0 else 32700) + zone
    return f"EPSG:{epsg}"


def _initialize_ee(project_id: str, authenticate_if_needed: bool = True):
    project_id = (project_id or "").strip()
    if not project_id:
        raise ValueError(
            "Enter your Google Cloud Project ID. The project must be registered for Earth Engine "
            "and have the Earth Engine API enabled."
        )
    try:
        import ee
    except ImportError as exc:
        raise RuntimeError("earthengine-api is not installed. Run run.bat or install requirements.txt.") from exc

    try:
        ee.Initialize(project=project_id)
        # Force a lightweight server request to validate credentials and project access.
        ee.String("GEE connection OK").getInfo()
        return ee
    except Exception as first_error:
        if not authenticate_if_needed:
            raise RuntimeError(
                "Earth Engine initialization failed. Check your project ID, Earth Engine API setup, "
                "and login. Details: " + str(first_error)
            ) from first_error
        try:
            _report(None, 0, "")
            ee.Authenticate(auth_mode="localhost")
            ee.Initialize(project=project_id)
            ee.String("GEE connection OK").getInfo()
            return ee
        except Exception as second_error:
            raise RuntimeError(
                "Could not authenticate/initialize Earth Engine. Confirm that you have Earth Engine "
                "access, the Cloud Project is registered for Earth Engine, the Earth Engine API is enabled, "
                "and the Project ID is correct.\n\nInitial error: " + str(first_error) +
                "\n\nAfter authentication: " + str(second_error)
            ) from second_error


def test_connection(project_id: str) -> str:
    _initialize_ee(project_id, authenticate_if_needed=True)
    return f"Earth Engine connection successful. Project: {project_id}."


def _parse_dates(start_date: str, end_date: str, label: str):
    try:
        start = date.fromisoformat(start_date.strip())
        end = date.fromisoformat(end_date.strip())
    except Exception as exc:
        raise ValueError(f"{label}: enter dates as YYYY-MM-DD.") from exc
    if end <= start:
        raise ValueError(f"{label}: end date must be later than start date. End date is exclusive in GEE.")
    return start, end


def _build_composite(ee, region, start_date: str, end_date: str, max_scene_cloud: float, water_threshold: float):
    start, end = _parse_dates(start_date, end_date, "Date range")
    if not 0 <= float(max_scene_cloud) <= 100:
        raise ValueError("Maximum scene cloud percentage must be between 0 and 100.")

    def mask_s2(image):
        scl = image.select("SCL")
        # Exclude saturated/defective, cloud shadow, unclassified/low-probability
        # cloud, medium/high probability cloud, cirrus, and snow/ice. Keep SCL=6 water.
        clear = (
            scl.neq(0)
            .And(scl.neq(1))
            .And(scl.neq(3))
            .And(scl.neq(7))
            .And(scl.neq(8))
            .And(scl.neq(9))
            .And(scl.neq(10))
            .And(scl.neq(11))
        )
        reflectance = image.select(["B3", "B8"]).updateMask(clear)
        ndwi = reflectance.expression(
            "(green - nir) / (green + nir)",
            {"green": reflectance.select("B3"), "nir": reflectance.select("B8")},
        ).rename("water_observation").gt(float(water_threshold)).toInt16().updateMask(clear)
        return reflectance.addBands(ndwi).copyProperties(
            image, ["system:time_start", "system:index", "CLOUDY_PIXEL_PERCENTAGE"]
        )

    raw = (
        ee.ImageCollection(COLLECTION_ID)
        .filterBounds(region)
        .filterDate(start.isoformat(), end.isoformat())
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", float(max_scene_cloud)))
    )
    scene_count = int(raw.size().getInfo())
    if scene_count < 1:
        raise ValueError(
            f"No Sentinel-2 SR Harmonized scenes found from {start} to {end} "
            f"for the selected boundary with scene cloud <= {max_scene_cloud:.0f}%."
        )

    collection = raw.map(mask_s2)
    composite = collection.select(["B3", "B8"]).median()
    valid_count = collection.select("B3").count().rename("valid_obs")
    water_count = collection.select("water_observation").sum().rename("water_count")
    # Frequency is stored as 0..1000 (0.1% increments) to retain precision in Int16.
    water_frequency = (water_count.divide(valid_count.max(ee.Image.constant(1)))
                       .multiply(1000).round().rename("water_frequency")
                       .updateMask(valid_count.gt(0)))
    # Maximum observed extent: a pixel is water if any clear observation in this
    # window classified it as water. No-observation pixels remain NoData.
    max_water = water_count.gt(0).rename("max_water").updateMask(valid_count.gt(0))
    image = composite.addBands(valid_count).addBands(water_frequency).addBands(max_water)
    image = image.select(["B3", "B8", "valid_obs", "water_frequency", "max_water"])
    # Int16 keeps files compact; -9999 explicitly marks cloud/no-observation pixels.
    image = image.toInt16().clip(region).unmask(NODATA).toInt16()

    dates = raw.aggregate_array("system:time_start").getInfo()
    dates = sorted({datetime.fromtimestamp(ts / 1000).date().isoformat() for ts in dates if ts})
    return image, scene_count, dates, start, end


def _extract_geotiff(payload: bytes):
    """Open either a direct GeoTIFF response or a ZIP containing GeoTIFF."""
    if payload[:2] == b"PK":
        with zipfile.ZipFile(BytesIO(payload)) as zf:
            tifs = [n for n in zf.namelist() if n.lower().endswith((".tif", ".tiff"))]
            if not tifs:
                details = ", ".join(zf.namelist()[:8])
                raise RuntimeError(f"GEE returned a ZIP without a GeoTIFF. Contents: {details}")
            data = zf.read(tifs[0])
            mem = MemoryFile(data)
            return mem, mem.open()
    if payload[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
        mem = MemoryFile(payload)
        return mem, mem.open()
    preview = payload[:300].decode("utf-8", errors="replace")
    raise RuntimeError("Earth Engine download did not return a GeoTIFF. Response: " + preview)


def _download_tiled_image(ee, image, region, projected_bounds, projected_aoi, out_dir: Path, year: int, target_crs: str,
                          progress: Progress, progress_start: int, progress_end: int,
                          tile_pixels: int = TILE_PIXELS):
    import requests
    minx0, miny0, maxx0, maxy0 = [float(v) for v in projected_bounds]
    minx = math.floor(minx0 / PIXEL_SIZE_M) * PIXEL_SIZE_M
    maxx = math.ceil(maxx0 / PIXEL_SIZE_M) * PIXEL_SIZE_M
    miny = math.floor(miny0 / PIXEL_SIZE_M) * PIXEL_SIZE_M
    maxy = math.ceil(maxy0 / PIXEL_SIZE_M) * PIXEL_SIZE_M
    width = int(round((maxx - minx) / PIXEL_SIZE_M))
    height = int(round((maxy - miny) / PIXEL_SIZE_M))
    if width < 1 or height < 1:
        raise ValueError("Could not derive a valid projected grid from the study boundary.")
    total_pixels = width * height
    if total_pixels > 250_000_000:
        raise ValueError(
            f"Study area would require {total_pixels:,} pixels at 10 m. This is too large for this "
            "direct-download mode. Use a smaller study boundary or split it into subareas."
        )
    cols = math.ceil(width / tile_pixels)
    rows = math.ceil(height / tile_pixels)
    # Restrict downloads to tiles that intersect the user-provided study boundary.
    # The raster still uses the boundary bounding box as its grid, but skipped areas
    # remain NoData instead of requesting tiles for irrelevant parts of the rectangle.
    intersecting_tiles = []
    for row in range(rows):
        for col in range(cols):
            x0 = minx + col * tile_pixels * PIXEL_SIZE_M
            y1 = maxy - row * tile_pixels * PIXEL_SIZE_M
            x1 = min(x0 + tile_pixels * PIXEL_SIZE_M, maxx)
            y0 = max(y1 - tile_pixels * PIXEL_SIZE_M, miny)
            if projected_aoi.intersects(box(x0, y0, x1, y1)):
                intersecting_tiles.append((row, col))
    ntiles = len(intersecting_tiles)
    if ntiles < 1:
        raise ValueError("No download tiles intersect the selected study boundary. Check its CRS and geometry.")
    if ntiles > 2500:
        raise ValueError(
            f"The 10 m AOI needs {ntiles:,} download tiles. Please split the boundary into smaller areas "
            "or use a coarser resolution in a future version."
        )

    transform = Affine(PIXEL_SIZE_M, 0, minx, 0, -PIXEL_SIZE_M, maxy)
    out_dir.mkdir(parents=True, exist_ok=True)
    composite_path = out_dir / f"S2_{year}_B03_B08_ValidObs_composite.tif"
    b3_path = out_dir / f"S2_{year}_B03_composite.tif"
    b8_path = out_dir / f"S2_{year}_B08_composite.tif"
    valid_path = out_dir / f"S2_{year}_valid_observations.tif"
    frequency_path = out_dir / f"S2_{year}_water_frequency_per_mille.tif"
    max_water_path = out_dir / f"S2_{year}_maximum_observed_water_extent.tif"
    profile = {
        "driver": "GTiff", "width": width, "height": height, "count": 5,
        "dtype": "int16", "crs": target_crs, "transform": transform,
        "nodata": NODATA, "compress": "deflate", "predictor": 2, "tiled": True,
        "blockxsize": 256, "blockysize": 256,
    }
    start_time = time.time()
    with rasterio.open(composite_path, "w", **profile) as dst:
        # Initialize the output grid to NoData so skipped non-AOI tiles are not zero-valued.
        for row_start in range(0, height, 256):
            chunk_h = min(256, height - row_start)
            fill = np.full((5, chunk_h, width), NODATA, dtype=np.int16)
            dst.write(fill, window=Window(0, row_start, width, chunk_h))
        completed = 0
        for row, col in intersecting_tiles:
                x0 = minx + col * tile_pixels * PIXEL_SIZE_M
                y1 = maxy - row * tile_pixels * PIXEL_SIZE_M
                x1 = x0 + tile_pixels * PIXEL_SIZE_M
                y0 = y1 - tile_pixels * PIXEL_SIZE_M
                tile_region = ee.Geometry.Rectangle(
                    [x0, y0, x1, y1], proj=ee.Projection(target_crs), geodesic=False
                )
                url = image.getDownloadURL({
                    "name": f"S2_{year}_tile_{row}_{col}",
                    "bands": ["B3", "B8", "valid_obs", "water_frequency", "max_water"],
                    "region": tile_region,
                    "crs": target_crs,
                    "scale": PIXEL_SIZE_M,
                    "format": "GEO_TIFF",
                })
                # Earth Engine may temporarily return 429/5xx responses under load. Retry
                # the same tile with exponential backoff rather than aborting the entire run.
                response = None
                last_error = None
                for attempt in range(6):
                    try:
                        response = requests.get(url, timeout=(30, 180))
                        if response.status_code in (429, 500, 502, 503, 504):
                            last_error = RuntimeError(
                                f"HTTP {response.status_code} while downloading tile {completed + 1}/{ntiles}"
                            )
                            if attempt < 5:
                                wait_s = min(60, 2 ** attempt * 2)
                                _report(progress, progress_start + (progress_end - progress_start) * completed / ntiles,
                                        f"GEE temporarily unavailable (HTTP {response.status_code}); retrying tile "
                                        f"{completed + 1}/{ntiles} in {wait_s}s (attempt {attempt + 1}/6)...")
                                time.sleep(wait_s)
                                continue
                        response.raise_for_status()
                        break
                    except requests.RequestException as exc:
                        last_error = exc
                        if attempt >= 5:
                            raise RuntimeError(
                                f"Failed downloading tile {completed + 1}/{ntiles} after 6 attempts: {exc}"
                            ) from exc
                        wait_s = min(60, 2 ** attempt * 2)
                        _report(progress, progress_start + (progress_end - progress_start) * completed / ntiles,
                                f"Network/GEE error; retrying tile {completed + 1}/{ntiles} in {wait_s}s "
                                f"(attempt {attempt + 1}/6)...")
                        time.sleep(wait_s)
                if response is None:
                    raise RuntimeError(f"Could not download tile {completed + 1}/{ntiles}: {last_error}")
                mem, src = _extract_geotiff(response.content)
                try:
                    expected_w = min(tile_pixels, width - col * tile_pixels)
                    expected_h = min(tile_pixels, height - row * tile_pixels)
                    if src.count < 5:
                        raise RuntimeError(f"Downloaded tile has {src.count} bands; expected B3, B8, valid_obs, water_frequency and max_water.")
                    if abs(src.width - tile_pixels) > 1 or abs(src.height - tile_pixels) > 1:
                        raise RuntimeError(
                            f"Unexpected GEE tile dimensions {src.width}x{src.height}; expected about "
                            f"{tile_pixels}x{tile_pixels}. Check the selected CRS and download parameters."
                        )
                    # Copy only the intersecting area for the final edge tiles.
                    take_w = min(src.width, expected_w)
                    take_h = min(src.height, expected_h)
                    win = Window(col * tile_pixels, row * tile_pixels, take_w, take_h)
                    for band_idx in range(1, 6):
                        arr = src.read(band_idx, window=Window(0, 0, take_w, take_h))
                        dst.write(arr.astype(np.int16, copy=False), band_idx, window=win)
                finally:
                    src.close()
                    mem.close()
                completed += 1
                pct = progress_start + (progress_end - progress_start) * completed / ntiles
                _report(progress, pct, f"Downloading {year} composite tile {completed}/{ntiles}...")

    # Split the five-band stack into individual GeoTIFFs for analysis and diagnostics.
    with rasterio.open(composite_path) as src:
        single_profile = src.profile.copy()
        single_profile.update(count=1, dtype="int16", nodata=NODATA)
        for band_idx, out_path in [(1, b3_path), (2, b8_path), (3, valid_path),
                                   (4, frequency_path), (5, max_water_path)]:
            with rasterio.open(out_path, "w", **single_profile) as dst:
                for _, window in src.block_windows(1):
                    dst.write(src.read(band_idx, window=window), 1, window=window)

    return {
        "composite": str(composite_path), "b3": str(b3_path), "b8": str(b8_path),
        "valid_obs": str(valid_path), "water_frequency": str(frequency_path),
        "max_water": str(max_water_path), "crs": target_crs, "width": width, "height": height,
    }


def fetch_composites(project_id: str, boundary_path: str, output_dir: str,
                     earlier_start: str, earlier_end: str, later_start: str, later_end: str,
                     max_scene_cloud: float = 80, progress: Progress = None,
                     authenticate_if_needed: bool = True, water_threshold: float = 0.0) -> dict:
    """Fetch two local 10 m cloud-masked composites using the GEE Python API.

    End dates follow Earth Engine's exclusive end-date convention. Each date
    window is processed independently. Per-scene NDWI uses the supplied threshold
    to create water-frequency and maximum-observed-extent outputs.
    """
    if not -1.0 <= float(water_threshold) <= 1.0:
        raise ValueError("Water NDWI threshold must be between -1 and 1.")
    ee = _initialize_ee(project_id, authenticate_if_needed=authenticate_if_needed)
    gdf, geom = _read_aoi(boundary_path)
    target_crs = _utm_crs(gdf)
    region = ee.Geometry(geom.__geo_interface__)
    out_root = Path(output_dir).expanduser().resolve() / "00_GEE_Fetched"
    out_root.mkdir(parents=True, exist_ok=True)

    date_specs = [
        ("earlier", earlier_start, earlier_end, 2017),
        ("later", later_start, later_end, 2026),
    ]
    results = {}
    all_metadata = {
        "collection": COLLECTION_ID,
        "cloud_mask": "SCL excludes classes 0,1,3,7,8,9,10,11; SCL water class 6 retained",
        "composite": "per-pixel median of cloud-masked B3/B8; valid_obs is clear observation count",
        "water_extent_method": "maximum observed extent is the union of per-scene NDWI water detections after SCL cloud/shadow masking",
        "water_ndwi_threshold": float(water_threshold),
        "water_frequency_encoding": "water_frequency raster is per-mille (0..1000); divide by 10 for percent",
        "resolution_m": PIXEL_SIZE_M,
        "target_crs": target_crs,
        "scene_cloud_filter_percent": float(max_scene_cloud),
        "nodata": NODATA,
        "boundary": str(Path(boundary_path).resolve()),
        "years": {},
    }
    for i, (label, start, end, fallback_year) in enumerate(date_specs):
        start_d, end_d = _parse_dates(start, end, f"{label.title()} date range")
        year = start_d.year
        if label == "earlier":
            _report(progress, 2, f"Searching Sentinel-2 SR Harmonized for {start_d} to {end_d}...")
            p0, p1 = 3, 48
        else:
            _report(progress, 50, f"Searching Sentinel-2 SR Harmonized for {start_d} to {end_d}...")
            p0, p1 = 51, 96
        image, scene_count, dates, start_parsed, end_parsed = _build_composite(
            ee, region, start, end, float(max_scene_cloud), float(water_threshold)
        )
        _report(progress, p0 + 1, f"Found {scene_count} candidate scenes for {year}; building cloud-masked composite for the selected boundary...")
        year_dir = out_root / str(year)
        projected_gdf = gdf.to_crs(target_crs)
        projected_bounds = projected_gdf.total_bounds
        projected_aoi = projected_gdf.geometry.union_all()
        fetched = _download_tiled_image(
            ee, image, region, projected_bounds, projected_aoi, year_dir, year, target_crs,
            (lambda pct, msg, lo=p0, hi=p1: _report(progress, lo + (hi - lo) * pct / 100, msg)),
            0, 100,
        )
        fetched["year"] = year
        fetched["scene_count"] = scene_count
        fetched["dates"] = dates
        fetched["start_date"] = start_parsed.isoformat()
        fetched["end_date_exclusive"] = end_parsed.isoformat()
        results[label] = fetched
        all_metadata["years"][str(year)] = {
            "start_date": start_parsed.isoformat(), "end_date_exclusive": end_parsed.isoformat(),
            "candidate_scene_count": scene_count, "acquisition_dates": dates,
            "output": fetched,
        }
        _report(progress, p1, f"Finished {year} composite ({scene_count} scenes, {len(dates)} acquisition dates).")

    metadata_path = out_root / "GEE_fetch_metadata.json"
    metadata_path.write_text(json.dumps(all_metadata, indent=2), encoding="utf-8")
    return {"earlier": results["earlier"], "later": results["later"],
            "metadata": str(metadata_path), "output_dir": str(out_root), "target_crs": target_crs}
