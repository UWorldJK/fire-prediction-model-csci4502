import shutil

import numpy as np
import requests
import xarray as xr

from ingest.gridmet import clip_to_bbox, gridmet_url, ingest_gridmet
from lakehouse import s3
from lakehouse.config import WEST_BBOX, gridmet_key
from lakehouse.manifest import read_manifests


def _fake_download(src):
    def fake(url, dest, **kwargs):
        shutil.copy(src, dest)
        return dest

    return fake


def test_gridmet_url():
    assert gridmet_url("erc", 2020) == "https://www.northwestknowledge.net/metdata/data/erc_2020.nc"


def test_clip_keeps_bbox_and_raw_values(synthetic_gridmet):
    ds = xr.open_dataset(synthetic_gridmet, mask_and_scale=False)
    out = clip_to_bbox(ds, WEST_BBOX)
    assert -125 <= float(out.lon.min()) and float(out.lon.max()) <= -102
    assert 31 <= float(out.lat.min()) and float(out.lat.max()) <= 49.5
    assert out.sizes["lon"] == 24 and out.sizes["lat"] == 19
    assert out["air_temperature"].dtype == np.uint16
    assert out["air_temperature"].attrs["scale_factor"] == 0.1


def test_ingest_uploads_and_records(lake_bucket, synthetic_gridmet, mocker):
    mocker.patch("ingest.gridmet.stream_download", side_effect=_fake_download(synthetic_gridmet))
    res = ingest_gridmet(["tmmx"], [2020])
    key = gridmet_key("tmmx", 2020)
    assert s3.object_exists(key)
    assert res.failed == [] and len(res.records) == 1
    assert res.records[0].source_url.endswith("tmmx_2020.nc")
    assert res.records[0].s3_key == key
    assert read_manifests() == res.records
    stored = xr.open_dataset(s3.download_file(key), mask_and_scale=False)
    assert stored["air_temperature"].dtype == np.uint16
    assert float(stored.lon.min()) >= -125


def test_ingest_skips_existing(lake_bucket, tmp_path, mocker):
    p = tmp_path / "empty"
    p.write_bytes(b"")
    s3.upload_file(p, gridmet_key("tmmx", 2020))
    dl = mocker.patch("ingest.gridmet.stream_download")
    res = ingest_gridmet(["tmmx"], [2020])
    assert res.skipped == [gridmet_key("tmmx", 2020)]
    dl.assert_not_called()


def test_ingest_gridmet_continues_after_http_error(lake_bucket, synthetic_gridmet, mocker):
    ok = _fake_download(synthetic_gridmet)

    def fake(url, dest, **kwargs):
        if "2019" in url:
            raise requests.HTTPError("404")
        return ok(url, dest)

    mocker.patch("ingest.gridmet.stream_download", side_effect=fake)
    res = ingest_gridmet(["tmmx"], [2019, 2020])
    assert len(res.records) == 1
    assert res.failed[0][0].endswith("tmmx_2019.nc")
    assert "404" in res.failed[0][1]
