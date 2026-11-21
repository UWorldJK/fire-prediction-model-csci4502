import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr

from ingest import integrate
from ingest.integrate import integrate_gridmet, integrate_landfire, integrate_mtbs
from lakehouse import s3
from lakehouse.config import (
    INTEGRATED_MTBS_KEY,
    gridmet_key,
    integrated_gridmet_key,
    integrated_landfire_key,
    mtbs_key,
    s3fs_options,
)


def _upload_raw(factory, years):
    for y in years:
        s3.upload_file(factory(y), gridmet_key("tmmx", y))


def _open(bucket="test-lake"):
    return xr.open_zarr(
        f"s3://{bucket}/{integrated_gridmet_key('tmmx')}", storage_options=s3fs_options(), mask_and_scale=False
    )


def test_integrate_gridmet_builds_all_years(lake_bucket, synthetic_gridmet_factory):
    _upload_raw(synthetic_gridmet_factory, [2019, 2020])
    res = integrate_gridmet(["tmmx"], [2019, 2020])
    assert res.failed == []
    z = _open()
    assert list(z.data_vars) == ["tmmx"]
    assert z.sizes["day"] == 6
    assert z["tmmx"].dtype == np.uint16
    assert z["tmmx"].attrs["scale_factor"] == 0.1


def test_integrate_gridmet_rerun_is_noop(lake_bucket, synthetic_gridmet_factory, mocker):
    _upload_raw(synthetic_gridmet_factory, [2019, 2020])
    integrate_gridmet(["tmmx"], [2019, 2020])
    spy = mocker.spy(integrate, "_write_year")
    res = integrate_gridmet(["tmmx"], [2019, 2020])
    assert res.skipped == [integrated_gridmet_key("tmmx")]
    assert spy.call_count == 0
    assert _open().sizes["day"] == 6


def test_integrate_gridmet_appends_new_year(lake_bucket, synthetic_gridmet_factory, mocker):
    _upload_raw(synthetic_gridmet_factory, [2019])
    integrate_gridmet(["tmmx"], [2019, 2020])
    _upload_raw(synthetic_gridmet_factory, [2020])
    spy = mocker.spy(integrate, "_rebuild_gridmet")
    integrate_gridmet(["tmmx"], [2019, 2020])
    assert spy.call_count == 0
    assert _open().sizes["day"] == 6


def test_integrate_gridmet_rebuilds_when_earlier_year_added(lake_bucket, synthetic_gridmet_factory):
    _upload_raw(synthetic_gridmet_factory, [2020])
    integrate_gridmet(["tmmx"], [2019, 2020])
    _upload_raw(synthetic_gridmet_factory, [2019])
    integrate_gridmet(["tmmx"], [2019, 2020])
    days = pd.to_datetime(_open()["day"].values)
    assert len(days) == 6
    assert days.is_monotonic_increasing and days.is_unique
    assert days[0] == pd.Timestamp("2019-01-01")


def test_integrate_mtbs_dedupes_latest_snapshot_wins(lake_bucket, synthetic_mtbs, tmp_path):
    old = synthetic_mtbs.iloc[[0]].copy()
    old["incid_name"] = "OLD"
    new = synthetic_mtbs.iloc[[0, 3]].copy()
    new.loc[new.index[0], "incid_name"] = "NEW"
    for snap, gdf in (("2026-01-01", old), ("2026-11-21", new)):
        p = tmp_path / f"{snap}.parquet"
        gdf.to_parquet(p)
        s3.upload_file(p, mtbs_key(snap, "mtbs_perimeters_west.parquet"))
    integrate_mtbs()
    out = gpd.read_parquet(s3.download_file(INTEGRATED_MTBS_KEY))
    assert len(out) == 2
    assert out.set_index(out.event_id.str[:2]).loc["CO", "incid_name"] == "NEW"


def test_integrate_landfire_mosaics_one_cog_per_layer(lake_bucket, two_tile_geotiffs):
    res = integrate_landfire(["LF2020_SlpD"])
    assert res.failed == []
    with rasterio.open(s3.download_file(integrated_landfire_key("LF2020_SlpD"))) as src:
        assert src.count == 1
        assert src.width == two_tile_geotiffs["width"]
        assert src.crs.to_epsg() == 5070
        data = src.read(1)
    # band 2 of tile a is 102, of tile b is 202
    assert set(np.unique(data)) == {102, 202}


def test_integrate_landfire_skips_existing(lake_bucket, two_tile_geotiffs, mocker):
    integrate_landfire(["LF2020_SlpD"])
    res = integrate_landfire(["LF2020_SlpD"])
    assert res.skipped == [integrated_landfire_key("LF2020_SlpD")]


def test_integrated_values_identical_to_raw(lake_bucket, synthetic_gridmet_factory):
    _upload_raw(synthetic_gridmet_factory, [2019])
    integrate_gridmet(["tmmx"], [2019])
    raw = xr.open_dataset(s3.download_file(gridmet_key("tmmx", 2019)), mask_and_scale=False)
    z = _open()
    np.testing.assert_array_equal(z["tmmx"].values, raw["air_temperature"].values)
