"""gridMET: download yearly CONUS NetCDF files, clip to the western US, upload to bronze.

Values stay exactly as gridMET stores them (packed integers with scale_factor);
only the spatial subset and compression change.
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import xarray as xr

from ingest import IngestResult
from lakehouse import config, s3
from lakehouse.manifest import ManifestRecord, git_sha, new_run_id, utc_now, write_manifest
from lakehouse.s3 import stream_download

log = logging.getLogger(__name__)
GRIDMET_URL = "https://www.northwestknowledge.net/metdata/data"


def gridmet_url(var: str, year: int) -> str:
    return f"{GRIDMET_URL}/{var}_{year}.nc"


def clip_to_bbox(ds: xr.Dataset, bbox) -> xr.Dataset:
    west, south, east, north = bbox
    # gridMET latitude runs north -> south
    return ds.sel(lon=slice(west, east), lat=slice(north, south))


def _clip_file(src: Path, dest: Path) -> None:
    with xr.open_dataset(src, mask_and_scale=False) as ds:
        clipped = clip_to_bbox(ds, config.WEST_BBOX).load()
    encoding = {v: {"zlib": True, "complevel": 4} for v in clipped.data_vars if clipped[v].ndim}
    clipped.to_netcdf(dest, engine="netcdf4", encoding=encoding)


def ingest_gridmet(variables=None, years=None, force: bool = False) -> IngestResult:
    variables = list(variables or config.GRIDMET_VARS)
    years = list(years if years is not None else config.gridmet_years())
    result = IngestResult()
    run_id = new_run_id()
    sha = git_sha()
    for var in variables:
        for year in years:
            key = config.gridmet_key(var, year)
            url = gridmet_url(var, year)
            if not force and s3.object_exists(key):
                result.skipped.append(key)
                continue
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    raw = stream_download(url, Path(tmp) / f"{var}_{year}.nc")
                    out = Path(tmp) / Path(key).name
                    _clip_file(raw, out)
                    s3.upload_file(out, key, content_type="application/x-netcdf")
                    result.records.append(
                        ManifestRecord(
                            source="gridmet", s3_key=key, source_url=url, sha256=s3.sha256_file(out),
                            bytes=out.stat().st_size, downloaded_at=utc_now(),
                            bbox=list(config.WEST_BBOX),
                            params=json.dumps({"variable": var, "year": year}), git_sha=sha,
                        )
                    )
                write_manifest(result.records, "gridmet", run_id)
                log.info("gridmet %s %s uploaded", var, year)
            except Exception as err:  # keep going; report at the end
                log.warning("gridmet %s %s failed: %s", var, year, err)
                result.failed.append((url, f"{type(err).__name__}: {err}"))
    write_manifest(result.records, "gridmet", run_id)
    return result
