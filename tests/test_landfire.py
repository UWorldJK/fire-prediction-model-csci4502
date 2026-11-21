import json
import zipfile

import pytest

from ingest.landfire import LFPS_URL, ingest_landfire, job_payload, make_tiles, run_lfps_job
from lakehouse import s3
from lakehouse.config import WEST_BBOX, landfire_key


def test_make_tiles_covers_bbox():
    tiles = make_tiles(WEST_BBOX, 5.0)
    assert len(tiles) == 20
    assert tiles[0] == ("r0c0", (-125.0, 31.0, -120.0, 36.0))
    assert max(t[1][2] for t in tiles) == -102.0
    assert max(t[1][3] for t in tiles) == 49.5
    assert ("r3c4", (-105.0, 46.0, -102.0, 49.5)) in tiles


def test_job_payload_native_crs():
    p = job_payload(["LF2020_Elev", "LF2020_SlpD"], (-125.0, 31.0, -120.0, 36.0), 90, "e@x")
    assert p["Layer_List"] == "LF2020_Elev;LF2020_SlpD"
    assert p["Area_of_Interest"] == "-125.0 31.0 -120.0 36.0"
    assert p["Resample_Resolution"] == 90
    assert p["Email"] == "e@x"
    assert "Output_Projection" not in p


def test_run_lfps_job_returns_output_url(requests_mock):
    requests_mock.post(f"{LFPS_URL}/submit", json={"jobId": "j1"})
    requests_mock.get(
        f"{LFPS_URL}/status",
        [{"json": {"status": "Executing"}}, {"json": {"status": "Succeeded", "outputFile": "https://o/j1.zip"}}],
    )
    assert run_lfps_job({"x": 1}, poll_seconds=0) == ("https://o/j1.zip", "j1")


def test_run_lfps_job_raises_on_failure(requests_mock):
    requests_mock.post(f"{LFPS_URL}/submit", json={"jobId": "j1"})
    requests_mock.get(f"{LFPS_URL}/status", json={"status": "Failed", "messages": ["too big"]})
    with pytest.raises(RuntimeError, match="too big"):
        run_lfps_job({"x": 1}, poll_seconds=0)


def test_ingest_landfire_uploads_tile_files(lake_bucket, mocker, monkeypatch, tmp_path):
    monkeypatch.setenv("LANDFIRE_EMAIL", "e@x")
    zpath = tmp_path / "out.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("j1.tif", b"tif")
        z.writestr("j1.tfw", b"tfw")
    mocker.patch("ingest.landfire.run_lfps_job", return_value=("https://o/j1.zip", "j1"))
    mocker.patch("ingest.landfire.stream_download", side_effect=lambda url, dest, **k: zpath.replace(dest) or dest)
    res = ingest_landfire({"topo": ["LF2020_Elev"]}, tile_deg=30.0)  # one tile
    assert res.failed == []
    assert s3.object_exists(landfire_key("topo", "r0c0", "j1.tif"))
    assert s3.object_exists(landfire_key("topo", "r0c0", "j1.tfw"))
    params = json.loads(res.records[0].params)
    assert params["layers"] == ["LF2020_Elev"] and params["lfps_job_id"] == "j1"


def test_ingest_landfire_skips_completed_tiles(lake_bucket, mocker, monkeypatch, tmp_path):
    monkeypatch.setenv("LANDFIRE_EMAIL", "e@x")
    p = tmp_path / "a.tif"
    p.write_bytes(b"")
    s3.upload_file(p, landfire_key("topo", "r0c0", "a.tif"))
    job = mocker.patch("ingest.landfire.run_lfps_job", side_effect=RuntimeError("stop"))
    res = ingest_landfire({"topo": ["LF2020_Elev"]}, tile_deg=5.0)
    assert landfire_key("topo", "r0c0", "") in res.skipped
    assert job.call_count == 19
    assert len(res.failed) == 19


def test_ingest_landfire_requires_email(monkeypatch):
    monkeypatch.delenv("LANDFIRE_EMAIL", raising=False)
    with pytest.raises(RuntimeError, match="LANDFIRE_EMAIL"):
        ingest_landfire({"topo": ["LF2020_Elev"]})
