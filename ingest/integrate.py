"""Integration: combine each dataset's raw bronze files across all years.

Outputs live under bronze/integrated/ and follow the bronze rule: values are only
concatenated or mosaicked, never cleaned. Everything here can be rebuilt from bronze
raw at any time.

- gridMET: one Zarr store per variable with every year on the `day` dimension.
- MTBS: every snapshot unioned, deduplicated on event_id (latest snapshot wins).
- LANDFIRE: one Cloud-Optimized GeoTIFF per layer, mosaicked from all tiles.
"""
from __future__ import annotations

import json
import logging
import re
import tempfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
import rasterio.merge
import rasterio.shutil
import xarray as xr

from ingest import IngestResult
from lakehouse import config, s3
from lakehouse.manifest import ManifestRecord, git_sha, new_run_id, utc_now, write_manifest

log = logging.getLogger(__name__)
GRIDMET_CHUNKS = {"day": 366, "lat": 64, "lon": 64}
_ZARR_ENCODING_KEYS = {"dtype", "scale_factor", "add_offset", "_FillValue", "units", "calendar"}


def _s3_url(key: str) -> str:
    return f"s3://{config.bucket()}/{key}"


def _prefix_bytes(prefix: str) -> int:
    paginator = config.client("s3").get_paginator("list_objects_v2")
    return sum(
        obj["Size"]
        for page in paginator.paginate(Bucket=config.bucket(), Prefix=prefix)
        for obj in page.get("Contents", [])
    )


def _record(source, key, input_prefix, params, sha256="", nbytes=None) -> ManifestRecord:
    return ManifestRecord(
        source=source, s3_key=key, source_url=_s3_url(input_prefix), sha256=sha256,
        bytes=nbytes if nbytes is not None else _prefix_bytes(key), downloaded_at=utc_now(),
        bbox=list(config.WEST_BBOX), params=json.dumps(params), git_sha=git_sha(),
    )


# ---------------- gridMET ----------------

def _raw_gridmet_years(var: str) -> list[int]:
    years = []
    for key in s3.list_keys(f"bronze/gridmet/variable={var}/"):
        m = re.search(r"/year=(\d{4})/", key)
        if m:
            years.append(int(m.group(1)))
    return sorted(set(years))


def _store_exists(var: str) -> bool:
    return bool(s3.list_keys(config.integrated_gridmet_key(var) + "/"))


def _integrated_years(var: str) -> list[int]:
    if not _store_exists(var):
        return []
    with xr.open_zarr(_s3_url(config.integrated_gridmet_key(var)), storage_options=config.s3fs_options()) as z:
        return sorted(set(pd.to_datetime(z["day"].values).year))


def _year_dataset(var: str, year: int) -> xr.Dataset:
    path = s3.download_file(config.gridmet_key(var, year))
    # Decode on read and re-encode on write with the source's own packing
    # (dtype/scale_factor/_FillValue), so the stored integers are identical to gridMET's.
    ds = xr.open_dataset(path).load()
    ds = ds.drop_vars([v for v in ("crs",) if v in ds.variables])
    (name,) = list(ds.data_vars)
    ds = ds.rename({name: var})
    for v in ds.variables:
        ds[v].encoding = {k: val for k, val in ds[v].encoding.items() if k in _ZARR_ENCODING_KEYS}
    ds.attrs["source_variable"] = name
    chunks = {d: min(n, ds.sizes[d]) for d, n in GRIDMET_CHUNKS.items() if d in ds.sizes}
    return ds.chunk(chunks)


def _write_year(var: str, year: int, first: bool) -> None:
    ds = _year_dataset(var, year)
    url = _s3_url(config.integrated_gridmet_key(var))
    if first:
        ds.to_zarr(url, mode="w", storage_options=config.s3fs_options(), zarr_format=2, consolidated=True)
    else:
        ds.to_zarr(url, append_dim="day", storage_options=config.s3fs_options(), consolidated=True)


def _delete_prefix(prefix: str) -> None:
    keys = s3.list_keys(prefix)
    client = config.client("s3")
    for i in range(0, len(keys), 1000):
        client.delete_objects(
            Bucket=config.bucket(), Delete={"Objects": [{"Key": k} for k in keys[i:i + 1000]]}
        )


def _rebuild_gridmet(var: str, years: list[int]) -> None:
    _delete_prefix(config.integrated_gridmet_key(var) + "/")
    for i, year in enumerate(years):
        _write_year(var, year, first=(i == 0))


def integrate_gridmet(variables=None, years=None, force: bool = False) -> IngestResult:
    variables = list(variables or config.GRIDMET_VARS)
    wanted_years = set(years if years is not None else config.gridmet_years())
    result = IngestResult()
    for var in variables:
        key = config.integrated_gridmet_key(var)
        try:
            want = [y for y in _raw_gridmet_years(var) if y in wanted_years]
            if not want:
                result.failed.append((key, f"no raw gridMET files for {var} in requested years"))
                continue
            have = _integrated_years(var)
            if have == want and not force:
                result.skipped.append(key)
                continue
            if not force and want[: len(have)] == have:
                for year in want[len(have):]:
                    _write_year(var, year, first=not have and year == want[0])
                action = "append"
            else:
                _rebuild_gridmet(var, want)
                action = "rebuild"
            result.records.append(
                _record("integrated_gridmet", key, f"bronze/gridmet/variable={var}/",
                        {"variable": var, "years": [want[0], want[-1]], "action": action})
            )
            log.info("integrated gridmet %s %s-%s (%s)", var, want[0], want[-1], action)
        except Exception as err:
            log.warning("integrate gridmet %s failed: %s", var, err)
            result.failed.append((key, f"{type(err).__name__}: {err}"))
    write_manifest(result.records, "integrated_gridmet", new_run_id())
    return result


# ---------------- MTBS ----------------

def integrate_mtbs() -> IngestResult:
    result = IngestResult()
    keys = [k for k in s3.list_keys("bronze/mtbs/") if k.endswith(".parquet")]
    if not keys:
        result.failed.append((config.INTEGRATED_MTBS_KEY, "no MTBS snapshots in bronze/mtbs/"))
        return result
    try:
        frames = []
        for key in keys:
            snapshot = re.search(r"snapshot=([^/]+)/", key).group(1)
            gdf = gpd.read_parquet(s3.download_file(key))
            frames.append(gdf.assign(_snapshot=snapshot))
        allfires = pd.concat(frames, ignore_index=True).sort_values("_snapshot", kind="stable")
        merged = allfires.drop_duplicates("event_id", keep="last").drop(columns="_snapshot")
        merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=frames[0].crs)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / Path(config.INTEGRATED_MTBS_KEY).name
            merged.to_parquet(out)
            s3.upload_file(out, config.INTEGRATED_MTBS_KEY, content_type="application/vnd.apache.parquet")
            result.records.append(
                _record("integrated_mtbs", config.INTEGRATED_MTBS_KEY, "bronze/mtbs/",
                        {"snapshots": len(keys), "fires": len(merged)},
                        sha256=s3.sha256_file(out), nbytes=out.stat().st_size)
            )
    except Exception as err:
        log.warning("integrate mtbs failed: %s", err)
        result.failed.append((config.INTEGRATED_MTBS_KEY, f"{type(err).__name__}: {err}"))
    write_manifest(result.records, "integrated_mtbs", new_run_id())
    return result


# ---------------- LANDFIRE ----------------

def _band_for(src, layer: str, fallback: int) -> int:
    for i, desc in enumerate(src.descriptions, start=1):
        if desc and desc.startswith(layer):
            return i
    return fallback


def _mosaic_layer(layer: str, dest: Path) -> int:
    group, band = config.layer_location(layer)
    tiles = [k for k in s3.list_keys(f"bronze/landfire/group={group}/") if k.endswith(".tif")]
    if not tiles:
        raise FileNotFoundError(f"no LANDFIRE tiles for group {group}")
    sources = [rasterio.open(s3.download_file(k)) for k in tiles]
    try:
        bands = {_band_for(src, layer, band) for src in sources}
        if len(bands) != 1:
            raise ValueError(f"{layer} is in different bands across tiles: {sorted(bands)}")
        mosaic, transform = rasterio.merge.merge(sources, indexes=[bands.pop()])
        profile = {
            "driver": "GTiff", "count": 1, "dtype": mosaic.dtype, "crs": sources[0].crs,
            "transform": transform, "width": mosaic.shape[2], "height": mosaic.shape[1],
            "nodata": sources[0].nodata,
        }
    finally:
        for src in sources:
            src.close()
    tmp_tif = dest.with_suffix(".tmp.tif")
    with rasterio.open(tmp_tif, "w", **profile) as dst:
        dst.write(mosaic)
        dst.set_band_description(1, layer)
    rasterio.shutil.copy(tmp_tif, dest, driver="COG", compress="DEFLATE")
    tmp_tif.unlink()
    return len(tiles)


def integrate_landfire(layers=None, force: bool = False) -> IngestResult:
    layers = list(layers or [name for group in config.LANDFIRE_GROUPS.values() for name in group])
    result = IngestResult()
    for layer in layers:
        key = config.integrated_landfire_key(layer)
        if not force and s3.object_exists(key):
            result.skipped.append(key)
            continue
        try:
            group, _ = config.layer_location(layer)
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / f"{layer}.tif"
                n_tiles = _mosaic_layer(layer, out)
                s3.upload_file(out, key, content_type="image/tiff")
                result.records.append(
                    _record("integrated_landfire", key, f"bronze/landfire/group={group}/",
                            {"layer": layer, "group": group, "tiles": n_tiles},
                            sha256=s3.sha256_file(out), nbytes=out.stat().st_size)
                )
        except Exception as err:
            log.warning("integrate landfire %s failed: %s", layer, err)
            result.failed.append((key, f"{type(err).__name__}: {err}"))
    write_manifest(result.records, "integrated_landfire", new_run_id())
    return result


def integrate_all() -> IngestResult:
    return integrate_gridmet().merge(integrate_mtbs()).merge(integrate_landfire())
