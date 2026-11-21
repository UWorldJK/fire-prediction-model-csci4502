import shutil

import geopandas as gpd

from ingest.mtbs import MTBS_URL, filter_mtbs, ingest_mtbs
from lakehouse import s3
from lakehouse.config import mtbs_key


def test_filter_keeps_west_study_years_all_types(synthetic_mtbs):
    out = filter_mtbs(synthetic_mtbs, years=range(2002, 2026))
    assert sorted(out.event_id.str[:2]) == ["CO", "UT"]
    assert set(out.incid_type) == {"Wildfire", "Prescribed Fire"}
    assert out.crs.to_epsg() == 4269
    assert out["ig_date"].dtype == synthetic_mtbs["ig_date"].dtype
    assert list(out.columns) == list(synthetic_mtbs.columns)


def test_ingest_mtbs_writes_geoparquet_and_metadata(lake_bucket, mocker, mtbs_zip):
    def fake(url, dest, **kwargs):
        assert url == MTBS_URL
        shutil.copy(mtbs_zip, dest)
        return dest

    mocker.patch("ingest.mtbs.stream_download", side_effect=fake)
    res = ingest_mtbs(snapshot="2026-11-21")
    pq_key = mtbs_key("2026-11-21", "mtbs_perimeters_west.parquet")
    assert s3.object_exists(pq_key)
    assert s3.object_exists(mtbs_key("2026-11-21", "mtbs_perims_DD_metadata.xml"))
    assert len(res.records) == 2 and res.failed == []
    back = gpd.read_parquet(s3.download_file(pq_key))
    # lake_bucket study years are 2019-2020: CO 2019 and UT 2020 stay, FL is east, CA is 1999
    assert sorted(back.event_id.str[:2]) == ["CO", "UT"]
    assert back.crs.to_epsg() == 4269


def test_ingest_mtbs_skips_existing_snapshot(lake_bucket, tmp_path, mocker):
    p = tmp_path / "x"
    p.write_bytes(b"")
    s3.upload_file(p, mtbs_key("2026-11-21", "mtbs_perimeters_west.parquet"))
    dl = mocker.patch("ingest.mtbs.stream_download")
    res = ingest_mtbs(snapshot="2026-11-21")
    dl.assert_not_called()
    assert res.skipped == [mtbs_key("2026-11-21", "mtbs_perimeters_west.parquet")]
