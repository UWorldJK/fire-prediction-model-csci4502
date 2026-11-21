"""Shared fixtures: an in-process moto S3 server and synthetic datasets."""
import os

import boto3
import pytest


@pytest.fixture(scope="session")
def moto_endpoint():
    from moto.server import ThreadedMotoServer

    server = ThreadedMotoServer(port=0, verbose=False)
    server.start()
    host, port = server.get_host_and_port()
    endpoint = f"http://{host}:{port}"
    saved = {k: os.environ.get(k) for k in _AWS_ENV}
    os.environ.update(
        {
            "FIRE_LAKEHOUSE_ENDPOINT": endpoint,
            "AWS_ACCESS_KEY_ID": "testing",
            "AWS_SECRET_ACCESS_KEY": "testing",
            "AWS_SESSION_TOKEN": "testing",
            "AWS_DEFAULT_REGION": "us-east-1",
        }
    )
    os.environ.pop("AWS_PROFILE", None)
    yield endpoint
    server.stop()
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


_AWS_ENV = (
    "FIRE_LAKEHOUSE_ENDPOINT",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_DEFAULT_REGION",
    "AWS_PROFILE",
)


@pytest.fixture
def lake_bucket(moto_endpoint, monkeypatch, tmp_path):
    """Fresh, empty bucket `test-lake` with study years 2019-2020."""
    s3 = boto3.client("s3", endpoint_url=moto_endpoint, region_name="us-east-1")
    s3.create_bucket(Bucket="test-lake")  # us-east-1 takes no LocationConstraint
    monkeypatch.setenv("FIRE_LAKEHOUSE_BUCKET", "test-lake")
    monkeypatch.setenv("STUDY_START_YEAR", "2019")
    monkeypatch.setenv("STUDY_END_YEAR", "2020")
    monkeypatch.setenv("FIRE_LAKEHOUSE_CACHE", str(tmp_path / "cache"))
    yield "test-lake"
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket="test-lake"):
        for obj in page.get("Contents", []):
            s3.delete_object(Bucket="test-lake", Key=obj["Key"])
    s3.delete_bucket(Bucket="test-lake")


@pytest.fixture
def tmp_cache(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    monkeypatch.setenv("FIRE_LAKEHOUSE_CACHE", str(cache))
    return cache


# ---------- synthetic gridMET ----------

def _make_gridmet(path, year):
    import numpy as np
    import pandas as pd
    import xarray as xr

    lat = np.arange(50.0, 23.0, -1.0)  # north -> south like gridMET
    lon = np.arange(-130.0, -59.0, 1.0)
    day = pd.date_range(f"{year}-01-01", periods=3, freq="D")
    data = (np.random.default_rng(year).uniform(250, 320, (3, lat.size, lon.size))).astype("float32")
    ds = xr.Dataset(
        {"air_temperature": (("day", "lat", "lon"), data, {"units": "K", "description": "Daily Maximum Temperature"})},
        coords={"day": day, "lat": lat, "lon": lon},
        attrs={"source": "synthetic gridMET"},
    )
    ds["crs"] = xr.DataArray(np.uint16(3), attrs={"spatial_ref": "EPSG:4326"})
    ds.to_netcdf(
        path,
        encoding={"air_temperature": {"dtype": "uint16", "scale_factor": 0.1, "_FillValue": 32767}},
    )
    return path


@pytest.fixture
def synthetic_gridmet_factory(tmp_path):
    def make(year: int):
        return _make_gridmet(tmp_path / f"synthetic_{year}.nc", year)

    return make


@pytest.fixture
def synthetic_gridmet(synthetic_gridmet_factory):
    return synthetic_gridmet_factory(2020)


# ---------- synthetic MTBS ----------

def _square(lon, lat, d=0.1):
    from shapely.geometry import box

    return box(lon, lat, lon + d, lat + d)


@pytest.fixture
def synthetic_mtbs():
    import geopandas as gpd

    rows = [
        ("CO3900010500020190601", "CO FIRE", "Wildfire", 1500, "2019-06-01", _square(-105.0, 39.0)),
        ("FL2600008100020190301", "FL FIRE", "Wildfire", 2500, "2019-03-01", _square(-81.0, 26.0)),
        ("CA3700012000019990801", "OLD CA", "Wildfire", 9000, "1999-08-01", _square(-120.0, 37.0)),
        ("UT3900011100020200415", "UT RX", "Prescribed Fire", 700, "2020-04-15", _square(-111.0, 39.0)),
    ]
    return gpd.GeoDataFrame(
        [r[:5] for r in rows],
        columns=["event_id", "incid_name", "incid_type", "burnbndac", "ig_date"],
        geometry=[r[5] for r in rows],
        crs="EPSG:4269",
    )


@pytest.fixture
def mtbs_zip(tmp_path, synthetic_mtbs):
    """A zip shaped like the MTBS composite download."""
    import zipfile

    shp_dir = tmp_path / "mtbs_src"
    shp_dir.mkdir()
    synthetic_mtbs.to_file(shp_dir / "mtbs_perims_DD.shp")
    (shp_dir / "mtbs_perims_DD_metadata.xml").write_text("<metadata/>")
    zpath = tmp_path / "mtbs_perimeter_data.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for f in shp_dir.iterdir():
            z.write(f, f.name)
    return zpath


# ---------- synthetic LANDFIRE tiles ----------

TOPO_DESCRIPTIONS = ("LF2020_Elev_CONUS", "LF2020_SlpD_CONUS", "LF2020_Asp_CONUS")


def _write_tile(path, x0, y0, width, height, value, descriptions, res=90.0):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    transform = from_origin(x0, y0, res, res)
    with rasterio.open(
        path, "w", driver="GTiff", width=width, height=height, count=len(descriptions),
        dtype="int16", crs="EPSG:5070", transform=transform, nodata=-9999,
    ) as dst:
        for b, desc in enumerate(descriptions, start=1):
            dst.write(np.full((height, width), value + b, dtype="int16"), b)
            dst.set_band_description(b, desc)
    return path


@pytest.fixture
def two_tile_geotiffs(lake_bucket, tmp_path):
    """Two adjacent 3-band topo tiles (r0c0 west, r0c1 east), 4x3 pixels each."""
    from lakehouse import s3
    from lakehouse.config import landfire_key

    a = _write_tile(tmp_path / "a.tif", -2000000.0, 2000000.0, 4, 3, 100, TOPO_DESCRIPTIONS)
    b = _write_tile(tmp_path / "b.tif", -2000000.0 + 4 * 90.0, 2000000.0, 4, 3, 200, TOPO_DESCRIPTIONS)
    s3.upload_file(a, landfire_key("topo", "r0c0", "ja.tif"))
    s3.upload_file(b, landfire_key("topo", "r0c1", "jb.tif"))
    return {"width": 8, "height": 3}


@pytest.fixture
def cc_three_versions(lake_bucket, tmp_path):
    """One-tile fuels groups for LF2016/LF2022/LF2024 with CC in band 5."""
    from lakehouse import s3
    from lakehouse.config import LANDFIRE_GROUPS, landfire_key

    for i, version in enumerate(("LF2016", "LF2022", "LF2024")):
        group = f"fuels_{version}"
        descs = tuple(f"{name}_CONUS" for name in LANDFIRE_GROUPS[group])
        tif = _write_tile(tmp_path / f"{group}.tif", -2000000.0, 2000000.0, 4, 3, 10 * i, descs)
        s3.upload_file(tif, landfire_key(group, "r0c0", f"{group}.tif"))
