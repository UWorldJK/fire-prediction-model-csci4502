"""Teammate readers: pull bronze (raw or integrated) and silver data into Python.

    export AWS_PROFILE=fire-lakehouse
    from lakehouse import open_gridmet, read_mtbs, open_landfire, query
"""
from __future__ import annotations

import functools
from collections.abc import Iterable

import awswrangler as wr
import boto3
import botocore.exceptions
import geopandas as gpd
import pandas as pd
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr

from lakehouse import config, s3
from lakehouse.errors import LakehouseAuthError

_AUTH_CODES = {
    "AccessDenied", "AccessDeniedException", "ExpiredToken", "ExpiredTokenException",
    "InvalidAccessKeyId", "InvalidClientTokenId", "SignatureDoesNotMatch", "UnrecognizedClientException",
}
_AUTH_HELP = (
    "AWS credentials are missing, expired, or for the wrong account: set AWS_PROFILE=fire-lakehouse "
    "(see docs/lakehouse.md)."
)
LANDFIRE_SERIES_VERSIONS = (2016, 2022, 2024)


def _auth_errors(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (
            botocore.exceptions.NoCredentialsError,
            botocore.exceptions.PartialCredentialsError,
            botocore.exceptions.ProfileNotFound,
            PermissionError,
        ) as err:
            raise LakehouseAuthError(f"{_AUTH_HELP} ({err})") from err
        except botocore.exceptions.ClientError as err:
            if err.response.get("Error", {}).get("Code") in _AUTH_CODES:
                raise LakehouseAuthError(f"{_AUTH_HELP} ({err})") from err
            raise

    return wrapper


def _is_missing(err: botocore.exceptions.ClientError) -> bool:
    return err.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound")


def _fetch(key: str, what: str, available, revalidate: bool = False):
    """download_file, but a missing object is a ValueError naming what *is* in the lake."""
    try:
        return s3.download_file(key, revalidate=revalidate)
    except botocore.exceptions.ClientError as err:
        if not _is_missing(err):
            raise
        have = available()
        raise ValueError(
            f"{what} is not in the lake yet. Available: {', '.join(map(str, have)) or 'nothing yet'} "
            "(open the viewer: python -m lakehouse.viewer)."
        ) from None


def _session() -> boto3.Session:
    return boto3.Session(region_name=config.REGION)


def _s3_url(key: str) -> str:
    return f"s3://{config.bucket()}/{key}"


# ---------------- gridMET ----------------

def _check_gridmet_request(variables, years) -> None:
    unknown = [v for v in variables if v not in config.GRIDMET_VARS]
    if unknown:
        raise ValueError(f"Unknown gridMET variable(s) {unknown}. Available: {', '.join(config.GRIDMET_VARS)}")
    lake = config.gridmet_years()
    outside = [y for y in years if y not in lake]
    if outside:
        raise ValueError(
            f"Years {outside} are not in the lake. Available: {lake.start}-{lake.stop - 1} "
            "(change StudyStartYear/StudyEndYear in the stack to add more)."
        )


def _clip_gridmet(ds: xr.Dataset, bbox) -> xr.Dataset:
    if bbox is None:
        return ds
    west, south, east, north = bbox
    return ds.sel(lon=slice(west, east), lat=slice(north, south))


def _open_gridmet_integrated(var: str, years: list[int]) -> xr.Dataset | None:
    key = config.integrated_gridmet_key(var)
    if not s3.list_keys(key + "/"):
        return None
    z = xr.open_zarr(_s3_url(key), storage_options=config.s3fs_options())
    have = set(pd.to_datetime(z["day"].values).year)
    if not set(years) <= have:
        return None
    return z.sel(day=pd.to_datetime(z["day"].values).year.isin(years))


def _gridmet_years_in_lake(var: str) -> list[int]:
    years = set()
    for key in s3.list_keys(f"bronze/gridmet/variable={var}/"):
        part = key.split("/year=")
        if len(part) > 1:
            years.add(int(part[1][:4]))
    return sorted(years)


def _open_gridmet_raw(var: str, years: list[int]) -> xr.Dataset:
    have = _gridmet_years_in_lake(var)
    missing = [y for y in years if y not in have]
    if missing:
        raise ValueError(
            f"gridMET {var} years {missing} are not in the lake yet. Available: "
            f"{', '.join(map(str, have)) or 'nothing yet'} (open the viewer: python -m lakehouse.viewer)."
        )
    parts = []
    for year in years:
        ds = xr.open_dataset(s3.download_file(config.gridmet_key(var, year)))
        ds = ds.drop_vars([v for v in ("crs",) if v in ds.variables])
        (name,) = list(ds.data_vars)
        parts.append(ds.rename({name: var}))
    return xr.concat(parts, dim="day")


@_auth_errors
def open_gridmet(variables, years: Iterable[int] | None = None, bbox=None) -> xr.Dataset:
    """Daily gridMET for the given short codes (e.g. ["tmmx", "erc"]) in physical units.

    Uses the integrated all-years Zarr (lazy; only touched chunks download) when it
    covers the request, otherwise the raw yearly files (cached locally).
    bbox is (west, south, east, north) in lon/lat.
    """
    variables = [variables] if isinstance(variables, str) else list(variables)
    years = sorted(years) if years is not None else list(config.gridmet_years())
    _check_gridmet_request(variables, years)
    out = []
    for var in variables:
        ds = _open_gridmet_integrated(var, years)
        if ds is None:
            ds = _open_gridmet_raw(var, years)
        out.append(_clip_gridmet(ds, bbox))
    return xr.merge(out, compat="override", join="outer")


# ---------------- MTBS ----------------

@_auth_errors
def read_mtbs(snapshot: str | None = None) -> gpd.GeoDataFrame:
    """MTBS perimeters (source CRS EPSG:4269). Default: all years integrated and deduplicated."""
    if snapshot is None:
        path = _fetch(config.INTEGRATED_MTBS_KEY, "Integrated MTBS", _mtbs_snapshots, revalidate=True)
    else:
        path = _fetch(config.mtbs_key(snapshot, "mtbs_perimeters_west.parquet"),
                      f"MTBS snapshot {snapshot}", _mtbs_snapshots)
    return gpd.read_parquet(path)


def _mtbs_snapshots() -> list[str]:
    return sorted({k.split("snapshot=")[1].split("/")[0] for k in s3.list_keys("bronze/mtbs/snapshot=")})


# ---------------- LANDFIRE ----------------

def _landfire_layers_in_lake() -> list[str]:
    return [k.rsplit("/", 1)[1].removesuffix(".tif") for k in s3.list_keys("bronze/integrated/landfire/")
            if k.endswith(".tif")]


@_auth_errors
def open_landfire(layer: str, bbox=None) -> xr.DataArray:
    """One LANDFIRE layer mosaicked over the western US (EPSG:5070). bbox in lon/lat."""
    config.layer_location(layer)  # raises ValueError listing available layers
    path = _fetch(config.integrated_landfire_key(layer), f"LANDFIRE layer {layer}", _landfire_layers_in_lake,
                  revalidate=True)
    da = rioxarray.open_rasterio(path, masked=True).squeeze("band", drop=True)
    da.name = layer
    if bbox is not None:
        da = da.rio.clip_box(*bbox, crs="EPSG:4326")
    return da


@_auth_errors
def open_landfire_series(acronym: str, bbox=None) -> xr.DataArray:
    """A fuels/vegetation layer (e.g. "FBFM40", "CC") stacked over LF2016, LF2022, LF2024."""
    layers = [f"LF{v}_{acronym}" for v in LANDFIRE_SERIES_VERSIONS]
    arrays = [open_landfire(name, bbox=bbox) for name in layers]
    base = arrays[0]
    aligned = [base] + [a if a.rio.transform() == base.rio.transform() and a.shape == base.shape
                        else a.rio.reproject_match(base) for a in arrays[1:]]
    series = xr.concat(aligned, dim=pd.Index(list(LANDFIRE_SERIES_VERSIONS), name="version"))
    series.name = acronym
    return series


# ---------------- SQL and silver ----------------

@_auth_errors
def query(sql: str) -> pd.DataFrame:
    """Run Athena SQL against the fire_lakehouse database (10 GB scan cap per query)."""
    return wr.athena.read_sql_query(
        sql, database=config.DATABASE, workgroup=config.WORKGROUP, ctas_approach=False,
        boto3_session=_session(),
    )


@_auth_errors
def read_silver(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    """A silver Parquet table written with write_silver()."""
    return wr.s3.read_parquet(
        path=_s3_url(f"silver/{name}/"), dataset=True, columns=columns, boto3_session=_session()
    )


@_auth_errors
def open_zarr(name: str) -> xr.Dataset:
    """A gridded silver dataset written with write_zarr()."""
    return xr.open_zarr(_s3_url(f"silver/{name}.zarr"), storage_options=config.s3fs_options())
