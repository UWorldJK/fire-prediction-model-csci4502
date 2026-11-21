import geopandas as gpd
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Point

from lakehouse.read import open_zarr
from lakehouse.write import write_silver, write_zarr


def test_write_silver_calls_to_parquet(mocker, monkeypatch):
    monkeypatch.setenv("FIRE_LAKEHOUSE_BUCKET", "b")
    tp = mocker.patch("lakehouse.write.wr.s3.to_parquet")
    name = write_silver(pd.DataFrame({"a": [1], "year": [2020]}), "cell_year_features", partition_cols=["year"])
    assert name == "silver_cell_year_features"
    kw = tp.call_args.kwargs
    assert kw["path"] == "s3://b/silver/cell_year_features/"
    assert kw["partition_cols"] == ["year"]
    assert kw["database"] == "fire_lakehouse" and kw["table"] == "silver_cell_year_features"
    assert kw["dataset"] is True and kw["mode"] == "overwrite"


def test_write_silver_geodataframe_geometry_as_wkb(mocker, monkeypatch):
    monkeypatch.setenv("FIRE_LAKEHOUSE_BUCKET", "b")
    tp = mocker.patch("lakehouse.write.wr.s3.to_parquet")
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(-105, 39)], crs="EPSG:4326")
    write_silver(gdf, "fires")
    df = tp.call_args.args[0] if tp.call_args.args else tp.call_args.kwargs["df"]
    assert not isinstance(df, gpd.GeoDataFrame)
    assert isinstance(df["geometry"].iloc[0], bytes)


@pytest.mark.parametrize("bad", ["Bad-Name", "1abc", "", "silver/x"])
def test_bad_name_refused(bad):
    with pytest.raises(ValueError):
        write_silver(pd.DataFrame({"a": [1]}), bad)


def test_write_zarr_round_trip(lake_bucket):
    url = write_zarr(xr.Dataset({"t": (("x",), [1.0, 2.0])}), "unit_test")
    assert url == "s3://test-lake/silver/unit_test.zarr"
    assert open_zarr("unit_test")["t"].values.tolist() == [1.0, 2.0]
