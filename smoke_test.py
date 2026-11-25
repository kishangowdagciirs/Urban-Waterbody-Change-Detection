"""Synthetic end-to-end test. Run with: python smoke_test.py"""
from pathlib import Path
import tempfile
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.transform import from_origin
from shapely.geometry import box
from waterbody_app.analysis import run_analysis


def main():
    root = Path(tempfile.mkdtemp(prefix="sentinel_water_smoke_"))
    crs = "EPSG:32643"
    transform = from_origin(500000, 1000, 10, 10)
    for year, size in [("2017", 20), ("2024", 25)]:
        green = np.full((100, 100), 0.1, dtype="float32")
        nir = np.full((100, 100), 0.3, dtype="float32")
        green[20:20 + size, 20:20 + size] = 0.3
        nir[20:20 + size, 20:20 + size] = 0.1
        for band, arr in [("B3", green), ("B8", nir)]:
            path = root / f"{year}_{band}.tif"
            with rasterio.open(path, "w", driver="GTiff", height=100, width=100, count=1,
                               dtype="float32", crs=crs, transform=transform, nodata=-9999) as dst:
                dst.write(arr, 1)
    boundary = root / "boundary.shp"
    gpd.GeoDataFrame({"id": [1]}, geometry=[box(500000, 0, 501000, 1000)], crs=crs).to_file(boundary)
    result = run_analysis({
        "earlier_year": "2017", "later_year": "2024",
        "earlier_b3": str(root / "2017_B3.tif"), "earlier_b8": str(root / "2017_B8.tif"),
        "later_b3": str(root / "2024_B3.tif"), "later_b8": str(root / "2024_B8.tif"),
        "boundary": str(boundary), "threshold": 0.0, "min_pixels": 25,
        "output_dir": str(root / "output")
    }, lambda pct, msg: print(f"[{pct:3d}%] {msg}"))
    summary = result["summary"]
    assert abs(summary["water_area_earlier_ha"] - 4.0) < 1e-6, summary
    assert abs(summary["water_area_later_ha"] - 6.25) < 1e-6, summary
    assert abs(summary["net_area_change_ha"] - 2.25) < 1e-6, summary
    print("SMOKE TEST PASSED")
    print("Output:", result["output_dir"])


if __name__ == "__main__":
    main()
