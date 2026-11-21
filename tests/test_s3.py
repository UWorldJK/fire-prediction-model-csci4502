import pytest
import requests

from lakehouse import s3


def test_upload_then_exists_and_list(lake_bucket, tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hi")
    assert not s3.object_exists("bronze/a.txt")
    s3.upload_file(p, "bronze/a.txt")
    assert s3.object_exists("bronze/a.txt")
    assert s3.list_keys("bronze/") == ["bronze/a.txt"]


def test_download_is_cached(lake_bucket, tmp_path, mocker):
    p = tmp_path / "x.txt"
    p.write_text("payload")
    s3.upload_file(p, "bronze/x")
    first = s3.download_file("bronze/x")
    assert first.read_text() == "payload"
    spy = mocker.spy(s3, "_download_raw")
    second = s3.download_file("bronze/x")
    assert first == second
    assert spy.call_count == 0


def test_download_refresh_refetches(lake_bucket, tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("v1")
    s3.upload_file(p, "bronze/x")
    s3.download_file("bronze/x")
    p.write_text("v2")
    s3.upload_file(p, "bronze/x")
    assert s3.download_file("bronze/x", refresh=True).read_text() == "v2"


def test_sha256_known_value(tmp_path):
    p = tmp_path / "b"
    p.write_bytes(b"abc")
    assert s3.sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_stream_download_writes_file(tmp_path, requests_mock):
    requests_mock.get("https://x/f.nc", content=b"data")
    out = s3.stream_download("https://x/f.nc", tmp_path / "f.nc")
    assert out.read_bytes() == b"data"
    assert not (tmp_path / "f.nc.part").exists()


def test_failed_download_uploads_nothing(tmp_path, requests_mock):
    requests_mock.get("https://x/f.nc", exc=requests.ConnectionError)
    with pytest.raises(requests.ConnectionError):
        s3.stream_download("https://x/f.nc", tmp_path / "f.nc")
    assert not (tmp_path / "f.nc").exists()
    assert not (tmp_path / "f.nc.part").exists()


def test_http_error_raises(tmp_path, requests_mock):
    requests_mock.get("https://x/missing.nc", status_code=404)
    with pytest.raises(requests.HTTPError):
        s3.stream_download("https://x/missing.nc", tmp_path / "m.nc")
    assert not (tmp_path / "m.nc").exists()
