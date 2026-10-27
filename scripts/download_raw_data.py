import os
import time
import zipfile
from datetime import datetime

import geopandas as gpd
import pandas as pd
import requests
import xarray as xr

RAW_DATA_DIR = "data/raw"
LOGS = "logs"

# GridMET yearly files: {GRIDMET_URL}/{var}_{year}.nc
GRIDMET_URL = "https://www.northwestknowledge.net/metdata/data"
# pr=precip, tmmx=max temp, rmin=min humidity, vs=wind speed,
# erc=energy release component, fm100=100-hr fuel moisture
GRIDMET_VARS = ["pr", "tmmx", "rmin", "vs", "erc", "fm100"]

REGIONS = {
    "west": [-125.0, 31.0, -102.0, 49.5],
    "east": [-102.0, 25.0, -66.9, 49.5],
}

# LANDFIRE Product Service: submit a job, poll status, download zipped GeoTIFF
# Layer names come from https://lfps.usgs.gov/api/products
LFPS_URL = "https://lfps.usgs.gov/api/job"
LANDFIRE_LAYERS = ["LF2020_Elev", "LF2020_SlpD", "LF2022_FBFM40"]
# Resolution in meters, must be a multiple of 30. ~4km matches GridMET
LANDFIRE_RESOLUTION = 4020

# All MTBS fire perimeters (1984-present) in one shapefile zip
MTBS_URL = (
    "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/MTBS_Fire/data/"
    "composite_data/burned_area_extent_shapefile/mtbs_perimeter_data.zip"
)


class FireData:
    """
    Class used to wrap all methods that fetch fire data

    """

    def __init__(self):
        """
        Currently nothing in constructor
        """
        pass

    def _download_file(self, url, path):
        """
        Stream url to path, skipping if path already exists
        """
        if os.path.exists(path):
            return path

        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"  fetching {os.path.basename(path)}")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(path + ".part", "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        os.rename(path + ".part", path)
        return path

    def _fetch_gridmet_file(self, var, year):
        """
        Download one GridMET yearly file if not already cached
        """
        return self._download_file(
            f"{GRIDMET_URL}/{var}_{year}.nc",
            f"{RAW_DATA_DIR}/gridmet_yearly/{var}_{year}.nc",
        )

    def download_gridmet(
        self, start_date, end_date, bbox="west", variables=GRIDMET_VARS
    ):
        """
        Download GridMET weather data into data directory

        Args:
            start_date: "2020-01-01"
            end_data: "2023-12-31"
            bbox = [west, south, east, north] or region name in REGIONS, default west
            variables: GridMET variable short names, default GRIDMET_VARS
        """
        print(f"Downloading GridMET data from {start_date} to {end_date}")

        if isinstance(bbox, str):
            bbox = REGIONS[bbox]
        west, south, east, north = bbox

        # Parse the file to get the correct region and time frame
        years = range(pd.Timestamp(start_date).year, pd.Timestamp(end_date).year + 1)
        subsets = []
        for var in variables:
            yearly = []
            for year in years:
                ds = xr.open_dataset(self._fetch_gridmet_file(var, year))
                # GridMET lat is stored north -> south, so slice north first
                yearly.append(
                    ds.sel(
                        lon=slice(west, east),
                        lat=slice(north, south),
                        day=slice(start_date, end_date),
                    )
                    .drop_vars("crs", errors="ignore")
                    .load()
                )
                ds.close()
            subsets.append(xr.concat(yearly, dim="day"))
        subset = xr.merge(subsets)

        # Save locally to raw data dir
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        output_file = f"{RAW_DATA_DIR}/gridmet_{start_date}_{end_date}.nc"
        subset.to_netcdf(output_file)
        print(f"Saved to {output_file}")

        return output_file

    def download_landfire(
        self, bbox="west", layers=LANDFIRE_LAYERS, resolution=LANDFIRE_RESOLUTION
    ):
        """
        Download LANDFIRE static layers (elevation, slope, fuel type)

        Requires LANDFIRE_EMAIL env var (LFPS requires an email per job)

        Args:
            bbox = [west, south, east, north] or region name in REGIONS, default west
            layers: LFPS layer names, default LANDFIRE_LAYERS
            resolution: output resolution in meters, multiple of 30
        """
        email = os.environ.get("LANDFIRE_EMAIL")
        if not email:
            raise RuntimeError("Set LANDFIRE_EMAIL env var to download LANDFIRE data")

        region = bbox if isinstance(bbox, str) else "custom"
        if isinstance(bbox, str):
            bbox = REGIONS[bbox]
        west, south, east, north = bbox

        out_dir = f"{RAW_DATA_DIR}/landfire_{region}"
        zip_path = f"{out_dir}.zip"
        if not os.path.exists(zip_path):
            print(f"Submitting LANDFIRE job for {layers}")
            r = requests.post(
                f"{LFPS_URL}/submit",
                json={
                    "Layer_List": ";".join(layers),
                    "Area_of_Interest": f"{west} {south} {east} {north}",
                    # EPSG:4326 so output lines up with GridMET lat/lon
                    "Output_Projection": "4326",
                    "Resample_Resolution": resolution,
                    "Email": email,
                },
                timeout=60,
            )
            r.raise_for_status()
            job_id = r.json()["jobId"]
            print(f"  job {job_id}")

            # Poll until LFPS finishes processing the job
            while True:
                status = requests.get(
                    f"{LFPS_URL}/status", params={"JobId": job_id}, timeout=60
                ).json()
                if status.get("outputFile"):
                    break
                if status.get("status") in ("Failed", "Canceled"):
                    raise RuntimeError(
                        f"LANDFIRE job {job_id} {status['status']}: {status.get('messages')}"
                    )
                print(
                    f"  status: {status.get('status')} (queue {status.get('queuePosition')})"
                )
                time.sleep(15)

            self._download_file(status["outputFile"], zip_path)

        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out_dir)
        tifs = [
            os.path.join(root, f)
            for root, _, files in os.walk(out_dir)
            for f in files
            if f.endswith(".tif")
        ]
        print(f"Saved to {tifs}")

        return tifs

    def download_mtbs(self, start_year, end_year, bbox=None):
        """
        Download MTBS burned area perimeters (fire labels)

        Args:
            start_year: 2020
            end_year: 2023
            bbox = [west, south, east, north] or region name in REGIONS, default all CONUS
        """
        print(f"Downloading MTBS perimeters from {start_year} to {end_year}")

        mtbs_dir = f"{RAW_DATA_DIR}/mtbs"
        zip_path = self._download_file(MTBS_URL, f"{mtbs_dir}/mtbs_perimeter_data.zip")
        with zipfile.ZipFile(zip_path) as z:
            shp_name = next(n for n in z.namelist() if n.endswith(".shp"))
            if not os.path.exists(f"{mtbs_dir}/{shp_name}"):
                z.extractall(mtbs_dir)

        gdf = gpd.read_file(f"{mtbs_dir}/{shp_name}").to_crs("EPSG:4326")
        gdf["ig_date"] = pd.to_datetime(gdf["ig_date"])
        gdf = gdf[gdf["ig_date"].dt.year.between(start_year, end_year)]

        if bbox is not None:
            if isinstance(bbox, str):
                bbox = REGIONS[bbox]
            west, south, east, north = bbox
            gdf = gdf.cx[west:east, south:north]

        output_file = f"{RAW_DATA_DIR}/mtbs_{start_year}_{end_year}.gpkg"
        gdf.to_file(output_file, driver="GPKG")
        print(f"Saved {len(gdf)} fires to {output_file}")

        return output_file


if __name__ == "__main__":
    data = FireData()
    data.download_gridmet("2020-01-01", "2020-12-31", "east")
    data.download_landfire("east")
    data.download_mtbs(2020, 2020, "east")
