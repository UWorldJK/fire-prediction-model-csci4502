import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Point, box

from cleaning.build_regions import (
    build_region_training,
    clean_mtbs,
    extract_gridmet_features,
    extract_landfire_features,
    landfire_version_for,
    sample_negatives,
)
from lakehouse import regions


@pytest.fixture
def states():
    rows = {
        "CA": (-124.5, 32.5, -120.0, 42.0),
        "NV": (-120.0, 35.0, -114.05, 42.0),
        "UT": (-114.05, 37.0, -109.05, 42.0),
        "CO": (-109.05, 37.0, -102.05, 41.0),
        "AZ": (-114.8, 31.3, -109.05, 37.0),
        "NM": (-109.05, 31.3, -103.0, 37.0),
    }
    gdf = gpd.GeoDataFrame({"STUSPS": list(rows)}, geometry=[box(*b) for b in rows.values()], crs="EPSG:4269")
    return regions.prepare_states(gdf)


@pytest.fixture
def wx():
    lat = np.arange(49.0, 30.9, -0.25)
    lon = np.arange(-125.0, -101.9, 0.25)
    day = pd.date_range("2019-12-01", "2020-12-31", freq="D")
    shape = (day.size, lat.size, lon.size)
    erc = np.broadcast_to(np.arange(day.size, dtype="float32")[:, None, None], shape)  # erc = day index
    return xr.Dataset(
        {
            "erc": (("day", "lat", "lon"), erc.copy()),
            "tmmx": (("day", "lat", "lon"), np.full(shape, 300.0, "float32")),
            "pr": (("day", "lat", "lon"), np.ones(shape, "float32")),
        },
        coords={"day": day, "lat": lat, "lon": lon},
    )


def _fires():
    rows = [
        ("CO3900010500020200701", "Wildfire", 1500, "2020-07-01", box(-105.0, 39.0, -104.9, 39.1)),
        ("CO3800010600020200801", "Wildfire", 300, "2020-08-01", box(-106.0, 38.0, -105.95, 38.05)),
        ("CA3700012100020200815", "Wildfire", 9000, "2020-08-15", box(-121.0, 37.0, -120.9, 37.1)),
        ("UT3900011100020200415", "Prescribed Fire", 700, "2020-04-15", box(-111.0, 39.0, -110.9, 39.1)),
        ("FL2600008100020200301", "Wildfire", 2500, "2020-03-01", box(-81.0, 26.0, -80.9, 26.1)),
        ("XX0000000000020200601", "Wildfire", 800, "2020-06-01", box(-112.0, 33.0, -111.9, 33.1)),  # AZ by location
        ("NM3300010700020200501", "Wildfire", 0, "2020-05-01", box(-107.0, 33.0, -106.9, 33.1)),     # 0 acres
        ("NM3300010700020200502", "Wildfire", 50, None, box(-107.0, 33.0, -106.9, 33.1)),            # no date
    ]
    return gpd.GeoDataFrame(
        [r[:4] for r in rows], columns=["event_id", "incid_type", "burnbndac", "ig_date"],
        geometry=[r[4] for r in rows], crs="EPSG:4269",
    )


# ---------- schemes ----------

def test_builtin_schemes_cover_every_state():
    for name, mapping in regions.SCHEMES.items():
        assert set(mapping) == set(regions.STATE_NAMES), name


def test_scheme_mapping_errors():
    with pytest.raises(ValueError, match="gacc"):
        regions.scheme_mapping("nope")
    with pytest.raises(ValueError, match="ZZ"):
        regions.scheme_mapping({"ZZ": "x"})
    assert regions.scheme_mapping({"CA": "coast", "CO": "rockies"}) == {"CA": "coast", "CO": "rockies"}


def test_region_codes_are_stable_and_nonzero():
    codes = regions.region_codes("state")
    assert min(codes.values()) == 1 and codes["arizona"] == 1


def test_state_from_event_id():
    s = regions.state_from_event_id(pd.Series(["CA3456712345620200815", "co1", "XX1", None]))
    assert s.tolist()[:2] == ["CA", "CO"] and s.isna().tolist()[2:] == [True, True]


def test_region_polygons_dissolve(states):
    polys = regions.region_polygons("gacc", states)
    assert set(polys["region"]) == {"california", "great_basin", "rocky_mountain", "southwest"}
    gb = polys.set_index("region").geometry["great_basin"]  # NV + UT
    assert gb.contains(Point(-117, 38)) and gb.contains(Point(-111, 39))


def test_region_bbox(states):
    assert regions.region_bbox("colorado", "state", states) == pytest.approx((-109.05, 37.0, -102.05, 41.0))
    with pytest.raises(ValueError, match="colorado"):
        regions.region_bbox("texas", "state", states)


def test_assign_regions_spatial(states):
    pts = gpd.GeoDataFrame(geometry=[Point(-105, 39), Point(-122, 38), Point(-130, 40)], crs="EPSG:4326")
    assert regions.assign_regions(pts, "state", states).tolist()[:2] == ["colorado", "california"]
    assert pd.isna(regions.assign_regions(pts, "state", states).iloc[2])
    assert regions.assign_regions(pts, "macro", states).tolist()[:2] == ["rockies", "pacific_coast"]


def test_region_mask_on_gridmet_grid(wx, states):
    mask = regions.region_mask(wx, "state", states)
    codes = regions.region_codes("state")
    assert int(mask.sel(lat=39.0, lon=-105.0)) == codes["colorado"]
    assert int(mask.sel(lat=37.5, lon=-122.0)) == codes["california"]
    assert int(mask.sel(lat=45.0, lon=-124.0)) == 0  # no Oregon polygon in this fixture
    # ascending latitude gives the same answer
    flipped = regions.region_mask(wx.isel(lat=slice(None, None, -1)), "state", states)
    assert int(flipped.sel(lat=39.0, lon=-105.0)) == codes["colorado"]


def test_subset_region_masks_outside(wx, states):
    co = regions.subset_region(wx[["erc"]].isel(day=slice(0, 2)), "colorado", "state", states)
    assert float(co.lon.min()) >= -109.05 and float(co.lon.max()) <= -102.05
    assert np.isnan(co["erc"].sel(lat=39.0, lon=-105.0).values).sum() == 0
    full = regions.subset_region(wx[["erc"]].isel(day=slice(0, 1)), "great_basin", "gacc", states)
    assert np.isnan(full["erc"].sel(lat=35.5, lon=-114.5, method="nearest").values).all()  # AZ corner


# ---------- cleaning ----------

def test_clean_mtbs(states):
    out = clean_mtbs(_fires(), states)
    assert sorted(out["event_id"]) == sorted(
        ["CO3900010500020200701", "CO3800010600020200801", "CA3700012100020200815", "XX0000000000020200601"]
    )
    assert out.set_index("event_id").loc["XX0000000000020200601", "state"] == "AZ"  # spatial fallback
    assert out.crs.to_epsg() == 4326
    co = out[out["event_id"] == "CO3900010500020200701"].iloc[0]
    assert co["region_state"] == "colorado" and co["region_gacc"] == "rocky_mountain"
    assert co["lon"] == pytest.approx(-104.95, abs=1e-3) and co["month"] == 7


def test_clean_mtbs_can_keep_prescribed(states):
    out = clean_mtbs(_fires(), states, incident_types=None)
    assert "UT3900011100020200415" in set(out["event_id"])


def test_extract_gridmet_window_is_before_ignition(wx):
    samples = pd.DataFrame({"lat": [39.0], "lon": [-105.0], "date": [pd.Timestamp("2020-07-01")]})
    f = extract_gridmet_features(wx, samples, window_days=7)
    day_ix = int(np.nonzero(wx.indexes["day"] == pd.Timestamp("2020-07-01"))[0][0])
    assert f["erc_mean7d"].iloc[0] == pytest.approx(np.mean(range(day_ix - 7, day_ix)))
    assert f["erc_max7d"].iloc[0] == day_ix - 1  # never the ignition day itself
    assert f["tmmx_mean7d"].iloc[0] == pytest.approx(26.85, abs=1e-3)  # K -> °C
    assert f["pr_sum7d"].iloc[0] == 7


def test_extract_gridmet_january_uses_previous_december(wx):
    samples = pd.DataFrame({"lat": [39.0], "lon": [-105.0], "date": [pd.Timestamp("2020-01-03")]})
    f = extract_gridmet_features(wx, samples, window_days=7)
    assert f["pr_sum7d"].iloc[0] == 7


def test_extract_gridmet_missing_days_are_nan(wx):
    samples = pd.DataFrame({"lat": [39.0], "lon": [-105.0], "date": [pd.Timestamp("2022-07-01")]})
    f = extract_gridmet_features(wx, samples)
    assert np.isnan(f["erc_mean7d"].iloc[0]) and np.isnan(f["pr_sum7d"].iloc[0])


def test_sample_negatives_inside_region_and_unburned(wx, states):
    fires = clean_mtbs(_fires(), states)
    co = fires[fires["state"] == "CO"]
    mask = regions.region_mask(wx, "state", states)
    code = regions.region_codes("state")["colorado"]
    big = co.copy()
    big["geometry"] = [box(-109.05, 37.0, -106.0, 41.0)] * len(big)  # burns the western 40% of CO
    neg = sample_negatives(co, mask, code, 200, perimeters=big)
    assert len(neg) > 0
    assert neg["lon"].between(-109.05, -102.05).all() and neg["lat"].between(37, 41).all()
    assert (neg["lon"] > -106.0).all()
    assert set(neg["date"]) <= set(co["ig_date"])


def test_landfire_version_for():
    assert [landfire_version_for(y) for y in (2005, 2019, 2020, 2022, 2023, 2025)] == [2016] * 2 + [2022] * 2 + [2024] * 2


def _fake_landfire(layer, bbox):
    import rioxarray  # noqa: F401

    x = np.arange(-1_300_000, -500_000, 9000.0)
    y = np.arange(2_200_000, 1_400_000, -9000.0)
    value = {"Asp": 90.0, "FBFM40": 102.0}.get(layer.split("_", 1)[1], 1.0)
    if layer.startswith("LF2022"):
        value = 165.0
    da = xr.DataArray(np.full((y.size, x.size), value), coords={"y": y, "x": x}, dims=("y", "x"))
    return da.rio.write_crs("EPSG:5070")


def test_extract_landfire_features():
    samples = pd.DataFrame({"lat": [39.0, 39.0], "lon": [-105.0, -105.0],
                            "date": pd.to_datetime(["2018-07-01", "2021-07-01"])})
    f = extract_landfire_features(samples, None, opener=_fake_landfire)
    assert f["aspect_sin"].iloc[0] == pytest.approx(1.0)
    assert f["fbfm40"].tolist() == [102, 165]  # LF2016 for 2018, LF2022 for 2021
    assert str(f["frg"].dtype) == "Int32"


def test_build_region_training_end_to_end(wx, states):
    fires = clean_mtbs(_fires(), states)
    calls = {}

    def fake_open_gridmet(variables, years, bbox):
        calls["bbox"], calls["years"] = bbox, years
        w, s, e, n = bbox
        return wx[variables].sel(lon=slice(w, e), lat=slice(n, s))

    df = build_region_training("colorado", fires, "state", years=[2020], n_neg=3, states=states,
                               gridmet_vars=("erc", "tmmx", "pr"),
                               open_gridmet=fake_open_gridmet, landfire_opener=_fake_landfire)
    assert calls["years"] == [2019, 2020]
    assert df["label"].sum() == 2 and (df["label"] == 0).sum() == 6
    assert set(df["region"]) == {"colorado"} and set(df["state"]) == {"CO"}
    assert {"erc_mean7d", "pr_sum7d", "elev", "fbfm40", "month_sin"} <= set(df.columns)
    assert df["erc_mean7d"].notna().all()


def test_build_region_training_empty_region(wx, states):
    fires = clean_mtbs(_fires(), states)
    df = build_region_training("utah", fires, "state", states=states, open_gridmet=lambda *a, **k: wx)
    assert df.empty


# ---------- boundaries from Natural Earth (cartopy) ----------

def test_read_states_from_natural_earth(mocker, tmp_path, states):
    ne = gpd.GeoDataFrame(
        {"iso_3166_2": ["US-" + s for s in states["state"]] + ["US-FL", "CA-BC", "MX-SON"]},
        geometry=list(states.geometry) + [box(-87, 25, -80, 31), box(-125, 49, -114, 55), box(-115, 26, -109, 32)],
        crs="EPSG:4326",
    )
    ne.to_file(tmp_path / "ne_admin1.shp")
    mocker.patch("cartopy.io.shapereader.natural_earth", return_value=str(tmp_path / "ne_admin1.shp"))
    regions._STATES_CACHE.clear()
    got = regions.read_states()
    regions._STATES_CACHE.clear()
    assert sorted(got["state"]) == ["AZ", "CA", "CO", "NM", "NV", "UT"]  # FL, Canada, Mexico dropped
    assert got.crs.to_epsg() == 4326
