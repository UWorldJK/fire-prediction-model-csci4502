import re

from lakehouse.manifest import ManifestRecord, new_run_id, read_manifests, write_manifest


def _record(key="bronze/k"):
    return ManifestRecord(
        "gridmet", key, "https://u", "abc", 3, "2026-11-21T00:00:00Z",
        [-125.0, 31.0, -102.0, 49.5], "{}", "deadbeef",
    )


def test_manifest_round_trip(lake_bucket):
    r = _record()
    key = write_manifest([r, r], "gridmet", "20261121T000000Z")
    assert key == "bronze/_manifest/gridmet_20261121T000000Z.jsonl"
    assert read_manifests() == [r, r]


def test_empty_manifest_not_written(lake_bucket):
    assert write_manifest([], "gridmet", "x") is None
    assert read_manifests() == []


def test_run_id_format():
    assert re.fullmatch(r"\d{8}T\d{6}Z", new_run_id())
