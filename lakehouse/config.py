"""Lakehouse constants, study-year/bucket resolution, S3 key layout and AWS clients.

Precedence for bucket and study years: environment variables, then the outputs of
the `fire-lakehouse` CloudFormation stack, then the defaults below.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from lakehouse.errors import LakehouseAuthError

# --- region of interest and data selection (Design Decisions D1-D6) ---
WEST_BBOX: tuple[float, float, float, float] = (-125.0, 31.0, -102.0, 49.5)  # W, S, E, N (EPSG:4326)
DEFAULT_STUDY_YEARS = (2002, 2025)

GRIDMET_VARS = [
    "pr", "tmmx", "tmmn", "rmax", "rmin", "vs", "th", "srad", "vpd", "erc", "bi", "fm100", "fm1000",
]

LANDFIRE_GROUPS: dict[str, list[str]] = {
    "topo": ["LF2020_Elev", "LF2020_SlpD", "LF2020_Asp"],
    "regime": ["LF2016_FRG", "LF2016_FRI", "LF2016_BPS"],
    "fuels_LF2016": ["LF2016_FBFM40", "LF2016_EVT", "LF2016_EVC", "LF2016_EVH", "LF2016_CC"],
    "fuels_LF2022": ["LF2022_FBFM40", "LF2022_EVT", "LF2022_EVC", "LF2022_EVH", "LF2022_CC"],
    "fuels_LF2024": ["LF2024_FBFM40", "LF2024_EVT", "LF2024_EVC", "LF2024_EVH", "LF2024_CC"],
}
LANDFIRE_RESOLUTION_M = 90
LANDFIRE_TILE_DEG = 5.0

# --- AWS resources (created by infra/lakehouse.yaml) ---
REGION = "us-east-1"
DATABASE = "fire_lakehouse"
WORKGROUP = "fire-lakehouse"
STACK_NAME = "fire-lakehouse"

INTEGRATED_MTBS_KEY = "bronze/integrated/mtbs/mtbs_perimeters_west.parquet"


def cache_dir() -> Path:
    """Local download cache (env FIRE_LAKEHOUSE_CACHE, default ~/.cache/fire-lakehouse)."""
    return Path(os.environ.get("FIRE_LAKEHOUSE_CACHE", Path.home() / ".cache" / "fire-lakehouse"))


def endpoint_url() -> str | None:
    return os.environ.get("FIRE_LAKEHOUSE_ENDPOINT") or None


def client(name: str):
    return boto3.client(name, region_name=REGION, endpoint_url=endpoint_url())


def s3fs_options() -> dict:
    """storage_options for xarray/zarr/pandas reads and writes through s3fs."""
    kwargs: dict = {"region_name": REGION}
    if endpoint_url():
        kwargs["endpoint_url"] = endpoint_url()
    return {"client_kwargs": kwargs}


_STACK_OUTPUTS_CACHE: dict[str, str] = {}


def stack_outputs() -> dict[str, str]:
    """Outputs of the fire-lakehouse stack, or {} when it cannot be reached.

    Only a successful lookup is cached, so fixing AWS_PROFILE mid-notebook works
    without restarting the kernel.
    """
    if _STACK_OUTPUTS_CACHE:
        return dict(_STACK_OUTPUTS_CACHE)
    try:
        stacks = client("cloudformation").describe_stacks(StackName=STACK_NAME)["Stacks"]
    except (ClientError, BotoCoreError):
        return {}
    _STACK_OUTPUTS_CACHE.update({o["OutputKey"]: o["OutputValue"] for o in stacks[0].get("Outputs", [])})
    return dict(_STACK_OUTPUTS_CACHE)


def _setting(env: str, output: str, default: str | None) -> str | None:
    if os.environ.get(env):
        return os.environ[env]
    return stack_outputs().get(output, default)


def bucket() -> str:
    name = _setting("FIRE_LAKEHOUSE_BUCKET", "BucketName", None)
    if not name:
        raise LakehouseAuthError(
            "Lakehouse bucket unknown: AWS credentials are missing, expired, or for the wrong account. "
            "Set AWS_PROFILE=fire-lakehouse (see docs/lakehouse.md)."
        )
    return name


def study_years() -> range:
    start = int(_setting("STUDY_START_YEAR", "StudyStartYear", str(DEFAULT_STUDY_YEARS[0])))
    end = int(_setting("STUDY_END_YEAR", "StudyEndYear", str(DEFAULT_STUDY_YEARS[1])))
    return range(start, end + 1)


def gridmet_years() -> range:
    """Study years plus the year before, so January fires get a 30-day lookback."""
    years = study_years()
    return range(years.start - 1, years.stop)


# --- S3 key layout ---
def gridmet_key(var: str, year: int) -> str:
    return f"bronze/gridmet/variable={var}/year={year}/{var}_{year}_west.nc"


def mtbs_key(snapshot: str, filename: str) -> str:
    return f"bronze/mtbs/snapshot={snapshot}/{filename}"


def landfire_key(group: str, tile_id: str, filename: str) -> str:
    return f"bronze/landfire/group={group}/tile={tile_id}/{filename}"


def manifest_key(source: str, run_id: str) -> str:
    return f"bronze/_manifest/{source}_{run_id}.jsonl"


def integrated_gridmet_key(var: str) -> str:
    return f"bronze/integrated/gridmet/{var}.zarr"


def integrated_landfire_key(layer: str) -> str:
    return f"bronze/integrated/landfire/{layer}.tif"


def layer_location(layer: str) -> tuple[str, int]:
    """(group, 1-based band) for a LANDFIRE layer name."""
    for group, layers in LANDFIRE_GROUPS.items():
        if layer in layers:
            return group, layers.index(layer) + 1
    known = ", ".join(name for layers in LANDFIRE_GROUPS.values() for name in layers)
    raise ValueError(f"Unknown LANDFIRE layer {layer!r}. Available: {known}")


def make_tiles(
    bbox: tuple[float, float, float, float], tile_deg: float
) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Split bbox into tiles `r{row}c{col}` (row 0 at the south), edges clipped to bbox."""
    west, south, east, north = bbox
    n_cols = math.ceil((east - west) / tile_deg)
    n_rows = math.ceil((north - south) / tile_deg)
    tiles = []
    for r in range(n_rows):
        for c in range(n_cols):
            w = west + c * tile_deg
            s = south + r * tile_deg
            tiles.append((f"r{r}c{c}", (w, s, min(w + tile_deg, east), min(s + tile_deg, north))))
    return tiles
