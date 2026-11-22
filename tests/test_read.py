import botocore.exceptions
import pandas as pd
import pytest

from lakehouse import read, s3
from lakehouse.config import INTEGRATED_MTBS_KEY, gridmet_key, mtbs_key
from lakehouse.errors import LakehouseAuthError
from lakehouse.read import (
    open_gridmet,
    open_landfire,
    open_landfire_series,
    query,
    read_mtbs,
)


def test_open_gridmet_uses_integrated_zarr(integrated_tmmx_2019_2020, mocker):
    spy = mocker.spy(read, "_open_gridmet_raw")
    ds = open_gridmet(["tmmx"], [2020])
    assert list(ds.data_vars) == ["tmmx"]
    assert ds.sizes["day"] == 3
    assert ds["tmmx"].dtype.kind == "f"
    assert spy.call_count == 0


def test_open_gridmet_falls_back_to_raw(lake_bucket, synthetic_gridmet):
    s3.upload_file(synthetic_gridmet, gridmet_key("tmmx", 2020))
    ds = open_gridmet(["tmmx"], [2020])
    assert list(ds.data_vars) == ["tmmx"] and ds.sizes["day"] == 3


def test_open_gridmet_bbox(integrated_tmmx_2019_2020):
    ds = open_gridmet(["tmmx"], [2020], bbox=(-109.0, 37.0, -102.0, 41.0))
    assert float(ds.lon.min()) >= -109 and float(ds.lon.max()) <= -102
    assert float(ds.lat.min()) >= 37 and float(ds.lat.max()) <= 41


def test_open_gridmet_rejects_years_outside_lake(lake_bucket):
    with pytest.raises(ValueError, match="2018.*2020"):  # gridmet_years() for study years 2019-2020
        open_gridmet(["tmmx"], [1990])


def test_open_gridmet_rejects_unknown_variable(lake_bucket):
    with pytest.raises(ValueError, match="fm1000"):
        open_gridmet(["temperature"], [2020])


def test_read_mtbs_default_is_integrated(lake_bucket, synthetic_mtbs, tmp_path):
    p = tmp_path / "i.parquet"
    synthetic_mtbs.iloc[:2].to_parquet(p)
    s3.upload_file(p, INTEGRATED_MTBS_KEY)
    assert len(read_mtbs()) == 2


def test_read_mtbs_snapshot(lake_bucket, synthetic_mtbs, tmp_path):
    p = tmp_path / "s.parquet"
    synthetic_mtbs.to_parquet(p)
    s3.upload_file(p, mtbs_key("2026-11-21", "mtbs_perimeters_west.parquet"))
    assert len(read_mtbs("2026-11-21")) == 4


def test_open_landfire_rejects_unknown_layer():
    with pytest.raises(ValueError, match="LF2020_Elev"):
        open_landfire("LF1999_Nope")


def test_open_landfire_reads_integrated_layer(two_tile_geotiffs):
    from ingest.integrate import integrate_landfire

    integrate_landfire(["LF2020_Elev"])
    da = open_landfire("LF2020_Elev")
    assert da.rio.crs.to_epsg() == 5070
    assert da.shape[-1] == two_tile_geotiffs["width"]


def test_open_landfire_series_stacks_versions(integrated_cc_three_versions):
    da = open_landfire_series("CC")
    assert da["version"].values.tolist() == [2016, 2022, 2024]
    assert da.sizes["version"] == 3


def test_query_passes_workgroup(mocker):
    rq = mocker.patch("lakehouse.read.wr.athena.read_sql_query", return_value=pd.DataFrame({"a": [1]}))
    assert query("select 1").a.tolist() == [1]
    assert rq.call_args.kwargs["workgroup"] == "fire-lakehouse"
    assert rq.call_args.kwargs["database"] == "fire_lakehouse"
    assert rq.call_args.kwargs["ctas_approach"] is False


def test_reader_without_credentials_raises_helpful_error(lake_bucket, mocker):
    mocker.patch("lakehouse.s3.download_file", side_effect=botocore.exceptions.NoCredentialsError())
    with pytest.raises(LakehouseAuthError, match="AWS_PROFILE=fire-lakehouse"):
        read_mtbs("2026-11-21")


def test_reader_access_denied_raises_helpful_error(lake_bucket, mocker):
    err = botocore.exceptions.ClientError({"Error": {"Code": "AccessDenied", "Message": "no"}}, "GetObject")
    mocker.patch("lakehouse.s3.download_file", side_effect=err)
    with pytest.raises(LakehouseAuthError, match="AWS_PROFILE=fire-lakehouse"):
        read_mtbs("2026-11-21")


def test_open_gridmet_missing_year_in_lake_raises_value_error(lake_bucket, synthetic_gridmet):
    s3.upload_file(synthetic_gridmet, gridmet_key("tmmx", 2020))
    with pytest.raises(ValueError, match="2019.*not in the lake yet.*2020"):
        open_gridmet("tmmx", years=[2019, 2020])


def test_open_landfire_layer_not_integrated_yet_raises_value_error(lake_bucket):
    with pytest.raises(ValueError, match="LF2024_CC.*not in the lake yet"):
        open_landfire("LF2024_CC")


def test_read_mtbs_unknown_snapshot_raises_value_error(lake_bucket, synthetic_mtbs, tmp_path):
    p = tmp_path / "m.parquet"
    synthetic_mtbs.to_parquet(p)
    s3.upload_file(p, mtbs_key("2026-11-21", "mtbs_perimeters_west.parquet"))
    with pytest.raises(ValueError, match="2001-01-01.*2026-11-21"):
        read_mtbs("2001-01-01")
