"""Silver writers for the cleaning stage. Bronze is written only by `python -m ingest`.

    from lakehouse import write_silver, write_zarr
    write_silver(features_df, "cell_year_features", partition_cols=["year"])
    write_zarr(gridmet_daily_ds, "gridmet_daily_4km")
"""
from __future__ import annotations

import re

import awswrangler as wr
import boto3
import geopandas as gpd
import pandas as pd
import xarray as xr

from lakehouse import config
from lakehouse.read import _auth_errors

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def _check_name(name: str) -> None:
    if not _NAME.match(name or ""):
        raise ValueError(f"Silver name {name!r} must match {_NAME.pattern} (lowercase, digits, underscores).")


@_auth_errors
def write_silver(
    df: pd.DataFrame, name: str, partition_cols: list[str] | None = None, mode: str = "overwrite"
) -> str:
    """Write a DataFrame as partitioned Parquet under silver/{name}/ and register Glue
    table silver_{name} (queryable in Athena). GeoDataFrame geometry is stored as WKB.
    mode: "overwrite" (default), "append" or "overwrite_partitions"."""
    _check_name(name)
    if isinstance(df, gpd.GeoDataFrame):
        geom = df.geometry.name
        wkb = df.geometry.to_wkb()
        df = pd.DataFrame(df.drop(columns=geom))
        df[geom] = wkb.values
    table = f"silver_{name}"
    wr.s3.to_parquet(
        df, path=f"s3://{config.bucket()}/silver/{name}/", dataset=True, database=config.DATABASE,
        table=table, partition_cols=partition_cols, mode=mode,
        boto3_session=boto3.Session(region_name=config.REGION),
    )
    return table


@_auth_errors
def write_zarr(ds: xr.Dataset, name: str, mode: str = "w") -> str:
    """Write a gridded dataset to silver/{name}.zarr. Returns its s3:// URL."""
    _check_name(name)
    url = f"s3://{config.bucket()}/silver/{name}.zarr"
    ds.to_zarr(url, mode=mode, storage_options=config.s3fs_options(), zarr_format=2, consolidated=True)
    return url
