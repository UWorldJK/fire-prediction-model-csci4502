import os
from datetime import datetime

import pandas as pd
import requests
import xarray as xr

RAW_DATA_DIR = "data/raw"
LOGS = "logs"


class FireData:
    """
    Class used to wrap all methods that fetch fire data

    """

    def __init__(self):
        """
        Currently nothing in constructor
        """
        pass

    def download_gridmet(self, start_date, end_data, bbox):
        """
        Download GridMET weather data into data directory

        Args:
            start_date: "2020-01-01"
            end_data: "2023-12-31"
            bbox = [west, south, north, east] - geographic region, default west
        """
        pass

    def download_landfire(self, bbox):
        """
        Download LANDIFRE static layers (elevation, slope, fuel type)

        Args:
            bbox = [west, south, north, east] - geographic region, default west
        """
        pass

    def download_mtbs(self, start_year, end_year):
        """
        Download MTBS burned area permieters (fire labels)

        Args:
            start_date: "2020-01-01"
            end_data: "2023-12-31"
        """
        pass


if __name__ == "__main__":
    data = FireData()
