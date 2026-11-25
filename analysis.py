"""Core processing for Sentinel-2 waterbody change detection.

Inputs are local B3 Green and B8 NIR rasters. The caller is responsible for
providing pre-masked imagery (masked pixels should be NoData).
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, shapes
from rasterio.warp import calculate_default_transform, reproject, transform_geom
from rasterio.transform import array_bounds
import geopandas as gpd
from shapely.geometry import shape, mapping, Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from scipy import ndimage

Progress = Optional[Callable[[int, str], None]]


def _report(cb: Progress, pct: int, message: str):
    if cb:
        cb(pct, message)


def _make_valid_compat(geom):
    """Repair an invalid Shapely geometry across old and new Shapely versions.

    Shapely 2 exposes geometry.make_valid(); older releases may not. Prefer
    shapely.validation.make_valid when available and fall back to buffer(0).
    GeometryCollection outputs are reduced to their polygonal components.
    """
    if geom is None or geom.is_empty or geom.is_valid:
        return geom
    repaired = None
    try:
        from shapely.validation import make_valid
        repaired = make_valid(geom)
    except (ImportError, AttributeError, NotImplementedError):
        repaired = None
    if repaired is None or repaired.is_empty:
        repaired = geom.buffer(0)
    if repaired.geom_type == "GeometryCollection":
        polygon_parts = []
        for part in repaired.geoms:
            if isinstance(part, (Polygon, MultiPolygon)) and not part.is_empty:
                polygon_parts.append(part)
        if not polygon_parts:
            return geom.buffer(0)
        repaired = unary_union(polygon_parts)
    # A final conservative fallback if a library returned an invalid result.
    if not repaired.is_valid:
        repaired = repaired.buffer(0)
    return repaired


def discover_band(folder: str, band: str) -> str:
    """Find a Sentinel-2 10 m B03/B08 band in a SAFE folder or nested directory."""
    root = Path(folder)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Not a folder: {folder}")
    patterns = [f"*_{band}_10m.jp2", f"*_{band}_10m.tif", f"*_{band}.tif", f"*_{band}.jp2"]
    matches = []
    for pattern in patterns:
        matches.extend(root.rglob(pattern))
    # Prefer native 10 m band files, and then the shortest path.
    matches = sorted(set(matches), key=lambda p: ("_10m" not in p.name.lower(), len(str(p))))
    if not matches:
        raise FileNotFoundError(f"Could not find band {band} in {folder}. Expected a 10 m B{band[-2:]} JP2/TIFF.")
    return str(matches[0])


def _read_boundary(shapefile: str) -> gpd.GeoDataFrame:
    if not shapefile or not Path(shapefile).exists():
        raise FileNotFoundError("Study boundary file was not found.")
    gdf = gpd.read_file(shapefile)
    if gdf.empty:
        raise ValueError("Study boundary contains no features.")
    if gdf.crs is None:
        raise ValueError("Study boundary has no CRS. Define its CRS before running the analysis.")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        raise ValueError("Study boundary contains no valid geometries.")
    # Repair invalid geometries compatibly with older Shapely/GeoPandas versions.
    gdf["geometry"] = gdf.geometry.apply(_make_valid_compat)
    return gdf


def _infer_sentinel_crs(path: str):
    """Infer CRS from SAFE metadata or a Sentinel-2 MGRS tile ID.

    This is a fallback only when the raster driver reports no CRS. It never
    overrides CRS metadata already present in the raster.
    """
    p = Path(path)

    # Prefer official SAFE XML metadata when it declares an EPSG code.
    search_dirs = []
    for parent in [p.parent, *p.parents]:
        search_dirs.append(parent)
        if parent.name.upper().endswith('.SAFE'):
            break
    xml_candidates = []
    for directory in search_dirs:
        if directory.exists() and directory.is_dir():
            for name in ('MTD_TL.xml', 'MTD_MSIL2A.xml', 'MTD_MSIL1C.xml'):
                candidate = directory / name
                if candidate.exists():
                    xml_candidates.append(candidate)
    import xml.etree.ElementTree as ET
    for xml_path in xml_candidates:
        try:
            root = ET.parse(xml_path).getroot()
            for node in root.iter():
                if node.tag.split('}')[-1].upper() == 'HORIZONTAL_CS_CODE' and node.text:
                    value = node.text.strip().upper()
                    match = re.search(r'EPSG\s*:\s*(\d+)', value)
                    if match:
                        from rasterio.crs import CRS
                        return CRS.from_epsg(int(match.group(1)))
        except (OSError, ValueError, ET.ParseError):
            # Continue to the MGRS filename fallback.
            pass

    # Sentinel-2 granule names commonly contain e.g. T43PGQ.
    # MGRS latitude bands C-M are in the southern hemisphere; N-X are north.
    candidates = [p.name, *(parent.name for parent in p.parents)]
    for candidate in candidates:
        match = re.search(r'(?<![A-Z0-9])T?(\d{2})([C-HJ-NP-X])([A-HJ-NP-Z]{2})(?![A-Z0-9])', candidate.upper())
        if match:
            zone = int(match.group(1))
            band = match.group(2)
            if 1 <= zone <= 60:
                epsg = (32600 if band >= 'N' else 32700) + zone
                from rasterio.crs import CRS
                return CRS.from_epsg(epsg)
    return None


def _source_crs(path: str, dataset):
    """Return raster CRS, falling back to Sentinel SAFE metadata/tile ID."""
    if dataset.crs is not None:
        return dataset.crs
    inferred = _infer_sentinel_crs(path)
    if inferred is not None:
        return inferred
    raise ValueError(
        f"Raster has no CRS and it could not be inferred from Sentinel-2 SAFE metadata or tile ID: {path}. "
        "Use a GeoTIFF with a defined CRS or select the correct SAFE product."
    )


def _read_raster_meta(path: str):
    if not path or not Path(path).exists():
        raise FileNotFoundError(f"Raster file not found: {path}")
    with rasterio.open(path) as ds:
        if ds.count < 1:
            raise ValueError(f"Raster has no bands: {path}")
        crs = _source_crs(path, ds)
        return crs, ds.bounds, ds.res, ds.width, ds.height, ds.nodata


def _projected_metric_crs(boundary: gpd.GeoDataFrame):
    crs = boundary.crs
    try:
        if crs.is_projected:
            axes = crs.axis_info
            if axes and all((a.unit_name or "").lower() in ("metre", "meter", "metres", "meters") for a in axes[:2]):
                return crs
    except Exception:
        pass
    estimated = boundary.estimate_utm_crs()
    if estimated is None:
        raise ValueError("Could not determine a suitable local projected CRS. Reproject the boundary to a CRS in metres.")
    return estimated


def _make_reference(b3_path: str, target_crs, resolution: float = 10.0):
    with rasterio.open(b3_path) as src:
        source_crs = _source_crs(b3_path, src)
        if src.transform.is_identity:
            raise ValueError(
                f"Raster has no usable georeferencing transform: {b3_path}. "
                "The CRS can be inferred from the SAFE tile, but pixel coordinates cannot."
            )
        # Reproject the source bounds to target CRS, then build a north-up 10 m grid.
        from rasterio.warp import transform_bounds
        bounds = transform_bounds(source_crs, target_crs, *src.bounds, densify_pts=21)
        left, bottom, right, top = bounds
        left = math.floor(left / resolution) * resolution
        right = math.ceil(right / resolution) * resolution
        bottom = math.floor(bottom / resolution) * resolution
        top = math.ceil(top / resolution) * resolution
        width = max(1, int(round((right - left) / resolution)))
        height = max(1, int(round((top - bottom) / resolution)))
        from affine import Affine
        transform = Affine(resolution, 0, left, 0, -resolution, top)
        return transform, width, height


def _warp_band(path: str, target_crs, transform, width: int, height: int, resampling=Resampling.bilinear):
    with rasterio.open(path) as src:
        source_crs = _source_crs(path, src)
        if src.transform.is_identity:
            raise ValueError(f"Raster has no usable georeferencing transform: {path}")
        src_data = src.read(1, masked=True).astype("float32").filled(np.nan)
        dest = np.full((height, width), np.nan, dtype="float32")
        reproject(
            source=src_data,
            destination=dest,
            src_transform=src.transform,
            src_crs=source_crs,
            src_nodata=np.nan,
            dst_transform=transform,
            dst_crs=target_crs,
            dst_nodata=np.nan,
            resampling=resampling,
            init_dest_nodata=True,
        )
        return dest


def _write_raster(path: Path, array: np.ndarray, crs, transform, nodata=-9999.0, dtype="float32"):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array)
    out_dtype = np.dtype(dtype)
    if np.issubdtype(out_dtype, np.integer):
        data = np.where(np.isfinite(arr), arr, 0).astype(out_dtype)
        profile_nodata = 0 if nodata is None else nodata
    else:
        data = np.where(np.isfinite(arr), arr, nodata).astype(out_dtype)
        profile_nodata = nodata
    profile = {
        "driver": "GTiff", "height": data.shape[0], "width": data.shape[1], "count": 1,
        "dtype": out_dtype.name, "crs": crs, "transform": transform,
        "compress": "deflate", "predictor": 2 if out_dtype.kind == "f" else 1,
        "tiled": True, "BIGTIFF": "IF_SAFER", "nodata": profile_nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def _polygonize(mask_arr: np.ndarray, transform, crs, min_pixels: int, year: str, output_path: Path):
    structure = np.ones((3, 3), dtype=np.uint8)  # 8-connectivity
    labels, count = ndimage.label(mask_arr.astype(bool), structure=structure)
    if count:
        sizes = np.bincount(labels.ravel())
        keep = sizes >= int(min_pixels)
        keep[0] = False
        filtered = keep[labels]
    else:
        filtered = np.zeros(mask_arr.shape, dtype=bool)
    records = []
    idx = 1
    for geom, value in shapes(filtered.astype("uint8"), mask=filtered, transform=transform, connectivity=8):
        if int(value) != 1:
            continue
        poly = _make_valid_compat(shape(geom))
        if poly is None or poly.is_empty:
            continue
        # Geometry repair can return a GeometryCollection containing only
        # non-polygon parts; skip it rather than writing invalid waterbody data.
        if poly.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        area = float(poly.area)
        records.append({"waterbody_id": f"{year}_{idx:05d}", "year": str(year), "area_m2": area,
                        "area_ha": area / 10000.0, "pixel_count": int(round(area / 100.0)), "geometry": poly})
        idx += 1
    if records:
        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
    else:
        gdf = gpd.GeoDataFrame({"waterbody_id": [], "year": [], "area_m2": [], "area_ha": [], "pixel_count": []}, geometry=[], crs=crs)
    # Writing is handled by the caller so both years can be stored in one GeoPackage.
    return filtered, gdf


def _match_waterbodies(old: gpd.GeoDataFrame, new: gpd.GeoDataFrame, output_csv: Path, output_gpkg: Path):
    """Overlap-based candidate matching. Split/merge candidates are explicitly flagged."""
    old = old.copy()
    new = new.copy()
    rows = []
    if old.empty and new.empty:
        pass
    else:
        # Calculate intersections; use a spatial index when available.
        overlaps = []
        for oi, og in old.iterrows():
            for ni, ng in new.iterrows():
                if not og.geometry.intersects(ng.geometry):
                    continue
                inter_area = float(og.geometry.intersection(ng.geometry).area)
                if inter_area > 0:
                    union_area = float(og.geometry.union(ng.geometry).area)
                    iou = inter_area / union_area if union_area else 0.0
                    overlaps.append((oi, ni, inter_area, iou))
        old_degree = {}
        new_degree = {}
        for oi, ni, _, _ in overlaps:
            old_degree[oi] = old_degree.get(oi, 0) + 1
            new_degree[ni] = new_degree.get(ni, 0) + 1
        best_for_old = {}
        best_for_new = {}
        for oi, ni, inter_area, iou in overlaps:
            if oi not in best_for_old or iou > best_for_old[oi][2]:
                best_for_old[oi] = (ni, inter_area, iou)
            if ni not in best_for_new or iou > best_for_new[ni][2]:
                best_for_new[ni] = (oi, inter_area, iou)
        matched_old, matched_new = set(), set()
        for oi, (ni, inter_area, iou) in best_for_old.items():
            # One-to-one primary matches; multi-overlap events are marked separately.
            if ni in best_for_new and best_for_new[ni][0] == oi:
                o = old.loc[oi]
                n = new.loc[ni]
                matched_old.add(oi); matched_new.add(ni)
                delta = float(n.area_m2 - o.area_m2)
                pct = (delta / float(o.area_m2) * 100.0) if float(o.area_m2) else None
                if old_degree.get(oi, 0) > 1 or new_degree.get(ni, 0) > 1:
                    status = "possible_split_or_merge"
                elif abs(delta) < 100.0:
                    status = "persistent"
                elif delta > 0:
                    status = "expanded"
                else:
                    status = "contracted"
                rows.append({"earlier_id": o.waterbody_id, "later_id": n.waterbody_id,
                             "earlier_area_m2": float(o.area_m2), "later_area_m2": float(n.area_m2),
                             "earlier_area_ha": float(o.area_ha), "later_area_ha": float(n.area_ha),
                             "change_m2": delta, "change_ha": delta / 10000.0,
                             "change_percent": pct, "overlap_iou": iou, "status": status})
        for oi, o in old.iterrows():
            if oi not in matched_old:
                status = "potentially_disappeared" if old_degree.get(oi, 0) == 0 else "unmatched_overlap"
                rows.append({"earlier_id": o.waterbody_id, "later_id": "", "earlier_area_m2": float(o.area_m2),
                             "later_area_m2": 0.0, "earlier_area_ha": float(o.area_ha), "later_area_ha": 0.0,
                             "change_m2": -float(o.area_m2), "change_ha": -float(o.area_ha),
                             "change_percent": -100.0, "overlap_iou": 0.0, "status": status})
        for ni, n in new.iterrows():
            if ni not in matched_new:
                status = "newly_detected" if new_degree.get(ni, 0) == 0 else "unmatched_overlap"
                rows.append({"earlier_id": "", "later_id": n.waterbody_id, "earlier_area_m2": 0.0,
                             "later_area_m2": float(n.area_m2), "earlier_area_ha": 0.0, "later_area_ha": float(n.area_ha),
                             "change_m2": float(n.area_m2), "change_ha": float(n.area_ha),
                             "change_percent": None, "overlap_iou": 0.0, "status": status})
    import pandas as pd
    df = pd.DataFrame(rows, columns=["earlier_id", "later_id", "earlier_area_m2", "later_area_m2", "earlier_area_ha",
                                   "later_area_ha", "change_m2", "change_ha", "change_percent", "overlap_iou", "status"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def run_analysis(config: dict, progress: Progress = None) -> dict:
    """Run the full workflow. config keys: earlier_year, later_year, earlier_b3, earlier_b8,
    later_b3, later_b8, boundary, threshold, min_pixels, output_dir. Optional
    earlier_water_mask/later_water_mask rasters override NDWI thresholding and
    allow GEE per-scene maximum-observed water extent to drive change detection.
    """
    import pandas as pd
    earlier_year = str(config["earlier_year"])
    later_year = str(config["later_year"])
    if earlier_year == later_year:
        raise ValueError("Earlier and later years must be different.")
    threshold = float(config["threshold"])
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("NDWI threshold must be between -1 and 1.")
    min_pixels = int(config.get("min_pixels", 25))
    if min_pixels < 1:
        raise ValueError("Minimum pixels must be at least 1.")
    out = Path(config["output_dir"]).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    report_dir = out / "01_Input_Reports"
    ndwi_dir = out / "02_NDWI"
    masks_dir = out / "03_Water_Masks"
    change_dir = out / "04_Change_Maps"
    vectors_dir = out / "05_Waterbody_Vectors"
    stats_dir = out / "06_Statistics"
    maps_dir = out / "07_Maps"
    logs_dir = out / "08_Logs"
    for d in [report_dir, ndwi_dir, masks_dir, change_dir, vectors_dir, stats_dir, maps_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    _report(progress, 2, "Validating boundary and input rasters...")
    boundary = _read_boundary(config["boundary"])
    for key in ["earlier_b3", "earlier_b8", "later_b3", "later_b8"]:
        _read_raster_meta(config[key])
    earlier_mask_path = config.get("earlier_water_mask")
    later_mask_path = config.get("later_water_mask")
    if bool(earlier_mask_path) != bool(later_mask_path):
        raise ValueError("Provide both earlier and later water-mask rasters, or neither.")
    use_extent_masks = bool(earlier_mask_path and later_mask_path)
    if use_extent_masks:
        _read_raster_meta(config["earlier_water_mask"])
        _read_raster_meta(config["later_water_mask"])
    target_crs = _projected_metric_crs(boundary)
    transform, width, height = _make_reference(config["earlier_b3"], target_crs, 10.0)
    # Crop the reference grid to the district boundary bounds to avoid processing the full source scene.
    boundary_proj = boundary.to_crs(target_crs)
    minx, miny, maxx, maxy = boundary_proj.total_bounds
    from affine import Affine
    left = math.floor(minx / 10.0) * 10.0
    right = math.ceil(maxx / 10.0) * 10.0
    bottom = math.floor(miny / 10.0) * 10.0
    top = math.ceil(maxy / 10.0) * 10.0
    width = max(1, int(round((right - left) / 10.0)))
    height = max(1, int(round((top - bottom) / 10.0)))
    transform = Affine(10.0, 0, left, 0, -10.0, top)
    if width * height > 250_000_000:
        raise ValueError("Study area grid is extremely large. Check the boundary CRS/extent before proceeding.")
    _report(progress, 10, f"Preparing common 10 m grid ({width:,} × {height:,} pixels)...")
    geoms = [mapping(g) for g in boundary_proj.geometry]
    inside = geometry_mask(geoms, out_shape=(height, width), transform=transform, invert=True, all_touched=False)

    arrays = {}
    names = [("earlier_b3", "earlier_green"), ("earlier_b8", "earlier_nir"),
             ("later_b3", "later_green"), ("later_b8", "later_nir")]
    for i, (key, label) in enumerate(names):
        _report(progress, 12 + i * 10, f"Aligning {label.replace('_', ' ')}...")
        arr = _warp_band(config[key], target_crs, transform, width, height)
        arr[~inside] = np.nan
        arrays[label] = arr

    # Inputs are assumed to be surface reflectance or consistently scaled values.
    # NDWI is scale-invariant if both bands use the same multiplicative scale.
    _report(progress, 55, "Calculating NDWI and water masks...")
    valid_old = np.isfinite(arrays["earlier_green"]) & np.isfinite(arrays["earlier_nir"])
    valid_new = np.isfinite(arrays["later_green"]) & np.isfinite(arrays["later_nir"])
    denom_old = arrays["earlier_green"] + arrays["earlier_nir"]
    denom_new = arrays["later_green"] + arrays["later_nir"]
    valid_old &= np.abs(denom_old) > 1e-12
    valid_new &= np.abs(denom_new) > 1e-12
    ndwi_old = np.full((height, width), np.nan, dtype=np.float32)
    ndwi_new = np.full((height, width), np.nan, dtype=np.float32)
    ndwi_old[valid_old] = (arrays["earlier_green"][valid_old] - arrays["earlier_nir"][valid_old]) / denom_old[valid_old]
    ndwi_new[valid_new] = (arrays["later_green"][valid_new] - arrays["later_nir"][valid_new]) / denom_new[valid_new]
    ndwi_old = np.clip(ndwi_old, -1, 1)
    ndwi_new = np.clip(ndwi_new, -1, 1)
    if use_extent_masks:
        _report(progress, 58, "Using per-scene maximum observed water extent masks from GEE...")
        extent_old = _warp_band(config["earlier_water_mask"], target_crs, transform, width, height, resampling=Resampling.nearest)
        extent_new = _warp_band(config["later_water_mask"], target_crs, transform, width, height, resampling=Resampling.nearest)
        valid_old = np.isfinite(extent_old) & inside
        valid_new = np.isfinite(extent_new) & inside
        raw_old = valid_old & (extent_old >= 1)
        raw_new = valid_new & (extent_new >= 1)
    else:
        raw_old = valid_old & (ndwi_old > threshold)
        raw_new = valid_new & (ndwi_new > threshold)
    # Connected-component filter uses 8-neighbour connectivity.
    struct = np.ones((3, 3), dtype=np.uint8)
    lab_old, _ = ndimage.label(raw_old, structure=struct)
    lab_new, _ = ndimage.label(raw_new, structure=struct)
    sizes_old = np.bincount(lab_old.ravel())
    sizes_new = np.bincount(lab_new.ravel())
    keep_old = sizes_old >= min_pixels; keep_old[0] = False
    keep_new = sizes_new >= min_pixels; keep_new[0] = False
    water_old = keep_old[lab_old] & raw_old
    water_new = keep_new[lab_new] & raw_new

    _report(progress, 65, "Writing NDWI and water-mask rasters...")
    _write_raster(ndwi_dir / f"NDWI_{earlier_year}.tif", ndwi_old, target_crs, transform)
    _write_raster(ndwi_dir / f"NDWI_{later_year}.tif", ndwi_new, target_crs, transform)
    _write_raster(masks_dir / f"WaterMask_{earlier_year}.tif", water_old.astype("uint8"), target_crs, transform, nodata=255, dtype="uint8")
    _write_raster(masks_dir / f"WaterMask_{later_year}.tif", water_new.astype("uint8"), target_crs, transform, nodata=255, dtype="uint8")

    # Pixel-level classes: 0=NoData/other, 1=persistent, 2=gain, 3=potential loss.
    _report(progress, 72, "Building pixel-level change map...")
    common_valid = valid_old & valid_new & inside
    change = np.zeros((height, width), dtype=np.uint8)
    change[common_valid & water_old & water_new] = 1
    change[common_valid & ~water_old & water_new] = 2
    change[common_valid & water_old & ~water_new] = 3
    # Nodata class for pixels not comparable across both dates.
    change[~common_valid] = 255
    _write_raster(change_dir / f"WaterChange_{earlier_year}_{later_year}.tif", change, target_crs, transform, nodata=255, dtype="uint8")

    _report(progress, 78, "Polygonizing filtered waterbodies...")
    gpkg = vectors_dir / "Waterbodies.gpkg"
    old_filtered, old_gdf = _polygonize(water_old, transform, target_crs, min_pixels, earlier_year, gpkg)
    new_filtered, new_gdf = _polygonize(water_new, transform, target_crs, min_pixels, later_year, gpkg)
    # Store non-empty year layers in one GeoPackage. Empty results remain represented in the summary.
    if gpkg.exists():
        gpkg.unlink()
    if not old_gdf.empty:
        old_gdf.to_file(gpkg, layer=f"water_{earlier_year}", driver="GPKG")
    if not new_gdf.empty:
        new_gdf.to_file(gpkg, layer=f"water_{later_year}", driver="GPKG", mode="a" if gpkg.exists() else "w")

    _report(progress, 85, "Matching polygons and calculating statistics...")
    comparison = _match_waterbodies(old_gdf, new_gdf, stats_dir / "Waterbody_Comparison.csv", gpkg)
    area_old = int(water_old.sum()) * 100.0
    area_new = int(water_new.sum()) * 100.0
    gain_area = int((common_valid & ~water_old & water_new).sum()) * 100.0
    loss_area = int((common_valid & water_old & ~water_new).sum()) * 100.0
    persistent_area = int((common_valid & water_old & water_new).sum()) * 100.0
    valid_area_old = int(valid_old.sum()) * 100.0
    valid_area_new = int(valid_new.sum()) * 100.0
    summary = {
        "earlier_year": earlier_year, "later_year": later_year, "ndwi_threshold": threshold,
        "minimum_pixels": min_pixels, "pixel_connectivity": 8, "pixel_size_m": 10,
        "target_crs": str(target_crs), "valid_pixels_earlier": int(valid_old.sum()),
        "valid_pixels_later": int(valid_new.sum()), "water_area_earlier_m2": area_old,
        "water_area_later_m2": area_new, "water_area_earlier_ha": area_old / 10000,
        "water_area_later_ha": area_new / 10000, "net_area_change_m2": area_new - area_old,
        "net_area_change_ha": (area_new - area_old) / 10000,
        "persistent_water_m2": persistent_area, "gain_m2": gain_area, "potential_loss_m2": loss_area,
        "earlier_waterbody_count": int(len(old_gdf)), "later_waterbody_count": int(len(new_gdf)),
        "detection_method": "GEE maximum observed per-scene water extent" if use_extent_masks else "NDWI threshold on supplied reflectance composites",
        "threshold_note": ("GEE per-scene NDWI threshold was applied before combining all valid observations into the maximum observed extent." if use_extent_masks else "Water is NDWI > threshold; both dates filtered independently by minimum connected pixels."),
        "cloud_mask_note": "Inputs are assumed pre-masked; verify masked pixels are represented as NoData.",
        "change_note": "Potential loss means water was not detected in the later image; it does not prove permanent disappearance."
    }
    with open(stats_dir / "Summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame([summary]).to_csv(stats_dir / "Summary.csv", index=False)
    try:
        pd.DataFrame([summary]).to_excel(stats_dir / "Summary.xlsx", index=False)
    except Exception:
        pass

    _report(progress, 92, "Writing map previews...")
    _write_maps(ndwi_old, ndwi_new, change, inside, maps_dir, earlier_year, later_year, threshold)
    metadata = {
        "earlier_year": earlier_year, "later_year": later_year,
        "earlier_b3": os.path.abspath(config["earlier_b3"]), "earlier_b8": os.path.abspath(config["earlier_b8"]),
        "later_b3": os.path.abspath(config["later_b3"]), "later_b8": os.path.abspath(config["later_b8"]),
        "earlier_water_mask": os.path.abspath(config["earlier_water_mask"]) if use_extent_masks else None,
        "later_water_mask": os.path.abspath(config["later_water_mask"]) if use_extent_masks else None,
        "detection_method": "GEE maximum observed per-scene water extent" if use_extent_masks else "NDWI threshold on reflectance composites",
        "boundary": os.path.abspath(config["boundary"]), "threshold": threshold, "min_pixels": min_pixels,
        "connectivity": 8, "resolution_m": 10, "target_crs": str(target_crs),
        "created": datetime.now().isoformat(timespec="seconds")
    }
    with open(report_dir / "Run_Metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    _report(progress, 100, "Analysis complete.")
    return {"output_dir": str(out), "summary": summary, "comparison_rows": int(len(comparison))}


def _write_maps(ndwi_old, ndwi_new, change, inside, outdir: Path, old_year: str, new_year: str, threshold: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    outdir.mkdir(parents=True, exist_ok=True)
    # NDWI previews
    for arr, year in [(ndwi_old, old_year), (ndwi_new, new_year)]:
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(np.ma.masked_invalid(arr), vmin=-1, vmax=1, cmap="BrBG")
        ax.set_title(f"NDWI — {year}")
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, shrink=0.75, label="NDWI")
        fig.tight_layout()
        fig.savefig(outdir / f"NDWI_{year}.png", dpi=300, bbox_inches="tight")
        fig.savefig(outdir / f"NDWI_{year}.pdf", bbox_inches="tight")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#d9d9d9", "#2563eb", "#22c55e", "#ef4444", "#000000"]
    cmap = ListedColormap(colors)
    view = change.astype(float)
    view[view == 255] = 4
    im = ax.imshow(view, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    ax.set_title(f"Waterbody Change: {old_year} to {new_year}")
    ax.set_axis_off()
    from matplotlib.patches import Patch
    legend = [Patch(color="#2563eb", label="Persistent water"), Patch(color="#22c55e", label="Water gain"),
              Patch(color="#ef4444", label="Potential water loss"), Patch(color="#000000", label="Not comparable / NoData")]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(outdir / f"WaterChange_{old_year}_{new_year}.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / f"WaterChange_{old_year}_{new_year}.pdf", bbox_inches="tight")
    plt.close(fig)
