"""Synthetic test for direct GEE maximum-observed-extent mask mode."""
from pathlib import Path
import tempfile
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.transform import from_origin
from shapely.geometry import box
from waterbody_app.analysis import run_analysis


def write_raster(path, array, crs, transform, nodata=-9999):
    with rasterio.open(
        path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1],
        count=1, dtype=array.dtype, crs=crs, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(array, 1)


def main():
    root = Path(tempfile.mkdtemp(prefix="max_extent_test_"))
    crs = "EPSG:32643"
    transform = from_origin(500000, 1000, 10, 10)
    # Reflectance inputs are present for NDWI diagnostics; direct masks drive area/change.
    for year in ("2018", "2026"):
        write_raster(root / f"{year}_B3.tif", np.full((100, 100), 0.1, dtype="float32"), crs, transform)
        write_raster(root / f"{year}_B8.tif", np.full((100, 100), 0.3, dtype="float32"), crs, transform)
    for year, size in (("2018", 20), ("2026", 15)):
        mask = np.zeros((100, 100), dtype="int16")
        mask[20:20 + size, 20:20 + size] = 1
        write_raster(root / f"{year}_mask.tif", mask, crs, transform)
    boundary = root / "boundary.shp"
    gpd.GeoDataFrame({"id": [1]}, geometry=[box(500000, 0, 501000, 1000)], crs=crs).to_file(boundary)
    result = run_analysis({
        "earlier_year": "2018", "later_year": "2026",
        "earlier_b3": str(root / "2018_B3.tif"), "earlier_b8": str(root / "2018_B8.tif"),
        "later_b3": str(root / "2026_B3.tif"), "later_b8": str(root / "2026_B8.tif"),
        "earlier_water_mask": str(root / "2018_mask.tif"),
        "later_water_mask": str(root / "2026_mask.tif"),
        "boundary": str(boundary), "threshold": 0.0, "min_pixels": 25,
        "output_dir": str(root / "output"),
    })
    summary = result["summary"]
    assert abs(summary["water_area_earlier_ha"] - 4.0) < 1e-6, summary
    assert abs(summary["water_area_later_ha"] - 2.25) < 1e-6, summary
    assert abs(summary["potential_loss_m2"] - 17500.0) < 1e-6, summary
    assert "maximum observed" in summary["detection_method"].lower(), summary
    print("MAX-EXTENT MASK TEST PASSED")
    print(f"Historical: {summary['water_area_earlier_ha']:.2f} ha")
    print(f"Present: {summary['water_area_later_ha']:.2f} ha")
    print(f"Potential loss: {summary['potential_loss_m2']:.0f} m²")


if __name__ == "__main__":
    main()
