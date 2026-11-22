import pytest

from lakehouse import config


def test_study_years_default(monkeypatch):
    monkeypatch.delenv("STUDY_START_YEAR", raising=False)
    monkeypatch.delenv("STUDY_END_YEAR", raising=False)
    monkeypatch.setattr(config, "stack_outputs", lambda: {})
    assert list(config.study_years()) == list(range(2002, 2026))
    assert config.gridmet_years()[0] == 2001
    assert config.gridmet_years()[-1] == 2025


def test_study_years_from_env(monkeypatch):
    monkeypatch.setenv("STUDY_START_YEAR", "1995")
    monkeypatch.setenv("STUDY_END_YEAR", "2000")
    assert list(config.study_years()) == list(range(1995, 2001))


def test_study_years_from_stack(monkeypatch):
    for v in ("STUDY_START_YEAR", "STUDY_END_YEAR"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(
        config, "stack_outputs", lambda: {"StudyStartYear": "2010", "StudyEndYear": "2012"}
    )
    assert list(config.study_years()) == [2010, 2011, 2012]


def test_bucket_from_env(monkeypatch):
    monkeypatch.setenv("FIRE_LAKEHOUSE_BUCKET", "b")
    assert config.bucket() == "b"


def test_key_builders():
    assert config.gridmet_key("tmmx", 2020) == "bronze/gridmet/variable=tmmx/year=2020/tmmx_2020_west.nc"
    assert (
        config.mtbs_key("2026-11-21", "mtbs_perimeters_west.parquet")
        == "bronze/mtbs/snapshot=2026-11-21/mtbs_perimeters_west.parquet"
    )
    assert config.landfire_key("topo", "r0c0", "a.tif") == "bronze/landfire/group=topo/tile=r0c0/a.tif"
    assert config.manifest_key("gridmet", "20261121T120000Z") == "bronze/_manifest/gridmet_20261121T120000Z.jsonl"
    assert config.integrated_gridmet_key("erc") == "bronze/integrated/gridmet/erc.zarr"
    assert config.INTEGRATED_MTBS_KEY == "bronze/integrated/mtbs/mtbs_perimeters_west.parquet"
    assert config.integrated_landfire_key("LF2022_FBFM40") == "bronze/integrated/landfire/LF2022_FBFM40.tif"


def test_layer_location():
    assert config.layer_location("LF2020_SlpD") == ("topo", 2)
    assert config.layer_location("LF2024_CC") == ("fuels_LF2024", 5)
    assert sum(len(v) for v in config.LANDFIRE_GROUPS.values()) == 21


def test_constants_match_design():
    assert config.WEST_BBOX == (-125.0, 31.0, -102.0, 49.5)
    assert len(config.GRIDMET_VARS) == 13
    assert config.LANDFIRE_RESOLUTION_M == 90
    assert config.REGION == "us-east-1"


def test_bucket_without_profile_raises_auth_error(monkeypatch):
    from lakehouse.errors import LakehouseAuthError

    monkeypatch.delenv("FIRE_LAKEHOUSE_BUCKET", raising=False)
    monkeypatch.setattr(config, "stack_outputs", lambda: {})
    with pytest.raises(LakehouseAuthError, match="AWS_PROFILE=fire-lakehouse"):
        config.bucket()


def test_failed_stack_lookup_is_not_cached(monkeypatch, mocker):
    import botocore.exceptions

    config._STACK_OUTPUTS_CACHE.clear()
    cf = mocker.Mock()
    cf.describe_stacks.side_effect = [
        botocore.exceptions.NoCredentialsError(),
        {"Stacks": [{"Outputs": [{"OutputKey": "BucketName", "OutputValue": "lake"}]}]},
    ]
    mocker.patch.object(config, "client", return_value=cf)
    assert config.stack_outputs() == {}
    assert config.stack_outputs() == {"BucketName": "lake"}
    config._STACK_OUTPUTS_CACHE.clear()
