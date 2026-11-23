"""Clean bronze data into per-region silver tables for model training.

    export AWS_PROFILE=fire-lakehouse
    # all western fires, cleaned and tagged with every region scheme
    python -m cleaning.build_regions fires
    # training samples (fires + sampled non-fire cells) for chosen regions
    python -m cleaning.build_regions training --scheme state --regions california colorado --years 2010-2020
    python -m cleaning.build_regions training --scheme gacc            # every GACC region
    # write local Parquet instead of silver (no AWS writes)
    python -m cleaning.build_regions training --regions colorado --out-dir out/

Outputs (silver):
    silver/mtbs_fires_regions/                 one row per cleaned fire, partitioned by state
    silver/training_{scheme}/region=.../       label 1 = fire, 0 = sampled non-fire cell-day

In Python:
    from cleaning.build_regions import clean_mtbs, build_region_training
    from lakehouse import read_silver
    co = read_silver("training_state").query("region == 'colorado'")
"""
from __future__ import annotations

import argparse
import logging
import warnings
from collections.abc import Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from lakehouse import regions

log = logging.getLogger(__name__)

DEFAULT_GRIDMET_VARS = ("tmmx", "rmin", "vs", "vpd", "erc", "bi", "fm100", "fm1000", "pr")
# Static layers; categorical ones (FRG) stay integer codes, aspect becomes sin/cos.
DEFAULT_LANDFIRE_STATIC = ("LF2020_Elev", "LF2020_SlpD", "LF2020_Asp", "LF2016_FRG", "LF2016_FRI")
# Time-varying fuels: each sample uses the LANDFIRE version closest to its year.
DEFAULT_LANDFIRE_FUELS = ("FBFM40", "CC")
CATEGORICAL_LANDFIRE = {"frg", "bps", "fbfm40", "evt"}  # integer codes, not quantities
KELVIN_VARS = {"tmmx", "tmmn"}
SUM_VARS = {"pr"}


# ---------------- MTBS ----------------

def clean_mtbs(fires: gpd.GeoDataFrame, states: gpd.GeoDataFrame | None = None,
               incident_types: Sequence[str] | None = ("Wildfire",)) -> gpd.GeoDataFrame:
    """Cleaned MTBS perimeters in EPSG:4326 with state, region_{scheme}, centroid and date columns.

    - drops rows with no ignition date, non-positive acres or empty geometry; repairs invalid polygons
    - keeps only `incident_types` (default wildfire only; None keeps everything, incl. prescribed)
    - one row per event_id
    - state from the event_id prefix, falling back to a spatial join; fires outside the
      western states (Mexico, Canada, or states not in regions.STATE_NAMES) are dropped
    """
    df = fires.copy()
    n0 = len(df)
    df["ig_date"] = pd.to_datetime(df["ig_date"], errors="coerce")
    df["burnbndac"] = pd.to_numeric(df["burnbndac"], errors="coerce")
    df = df[df["ig_date"].notna() & (df["burnbndac"] > 0)]
    df = df[df.geometry.notna() & ~df.geometry.is_empty]
    df["geometry"] = df.geometry.make_valid()
    if incident_types is not None and "incid_type" in df:
        df = df[df["incid_type"].isin(list(incident_types))]
    if "event_id" in df:
        df = df.drop_duplicates("event_id", keep="last")
    df = df.to_crs("EPSG:4326")

    state = regions.state_from_event_id(df["event_id"]) if "event_id" in df else pd.Series(pd.NA, index=df.index)
    missing = state.isna()
    if missing.any():
        state[missing] = regions.assign_states(df[missing], states)
    df["state"] = state.astype("string")
    df = df[df["state"].notna()]

    centroid = df.geometry.to_crs("EPSG:5070").centroid.to_crs("EPSG:4326")  # equal-area centroid
    df["lon"], df["lat"] = centroid.x.values, centroid.y.values
    df["year"] = df["ig_date"].dt.year.astype("int32")
    df["month"] = df["ig_date"].dt.month.astype("int8")
    df["doy"] = df["ig_date"].dt.dayofyear.astype("int16")
    df["log_acres"] = np.log1p(df["burnbndac"])
    for name in regions.SCHEMES:
        df[f"region_{name}"] = df["state"].map(regions.SCHEMES[name])
    log.info("clean_mtbs: kept %d of %d rows", len(df), n0)
    return df.reset_index(drop=True)


# ---------------- negatives ----------------

def sample_negatives(positives: gpd.GeoDataFrame, mask: xr.DataArray, code: int, n_per_positive: int,
                     perimeters: gpd.GeoDataFrame | None = None, seed: int = 42) -> pd.DataFrame:
    """Random cell-days inside the region (mask == code) with ignition dates drawn from the
    region's own fires, so season and year mix match the positives. Points that fall
    inside a fire perimeter of the same year are dropped."""
    rng = np.random.default_rng(seed)
    iy, ix = np.nonzero(mask.values == code)
    if iy.size == 0 or positives.empty:
        return pd.DataFrame(columns=["lat", "lon", "date"])
    n = len(positives) * n_per_positive
    pick = rng.integers(0, iy.size, n)
    dates = rng.choice(positives["ig_date"].values, n)
    neg = pd.DataFrame({
        "lat": mask["lat"].values[iy[pick]], "lon": mask["lon"].values[ix[pick]], "date": pd.to_datetime(dates),
    })
    neg["year"] = neg["date"].dt.year
    if perimeters is not None and not perimeters.empty:
        pts = gpd.GeoDataFrame(neg, geometry=gpd.points_from_xy(neg["lon"], neg["lat"]), crs="EPSG:4326")
        hit = gpd.sjoin(pts, perimeters[["year", "geometry"]].rename(columns={"year": "fire_year"}),
                        how="inner", predicate="intersects")
        burned = hit.index[hit["year"] == hit["fire_year"]].unique()
        neg = neg.drop(index=burned)
    return neg.drop(columns="year").reset_index(drop=True)


# ---------------- gridMET features ----------------

def extract_gridmet_features(ds: xr.Dataset, samples: pd.DataFrame, window_days: int = 7) -> pd.DataFrame:
    """Weather over the `window_days` days *before* each sample's date (no same-day or
    post-ignition weather leaks in). Mean per variable, total for precipitation,
    plus max for erc/vpd/tmmx. Temperatures converted to °C.

    Works year by year so only one year of the region is in memory at a time.
    """
    out = pd.DataFrame(index=samples.index)
    if samples.empty:
        return out
    lat_ix = ds.indexes["lat"].get_indexer(samples["lat"].values, method="nearest")
    lon_ix = ds.indexes["lon"].get_indexer(samples["lon"].values, method="nearest")
    dates = pd.to_datetime(samples["date"]).values.astype("datetime64[D]")
    lags = np.arange(1, window_days + 1)
    wanted = dates[:, None] - lags[None, :].astype("timedelta64[D]")  # (n, window)
    day_values = ds.indexes["day"].values.astype("datetime64[D]")
    years = pd.to_datetime(samples["date"]).dt.year.values

    cols: dict[str, np.ndarray] = {}
    for var in ds.data_vars:  # every column exists even if no year is in the lake
        names = ([f"{var}_sum{window_days}d"] if var in SUM_VARS else []) + [f"{var}_mean{window_days}d"]
        names += [f"{var}_max{window_days}d"] if var in ("erc", "vpd", "tmmx") else []
        for name in names:
            cols[name] = np.full(len(samples), np.nan)
    for year in np.unique(years):
        rows = np.nonzero(years == year)[0]
        lo, hi = np.datetime64(f"{year - 1}-12-01"), np.datetime64(f"{year}-12-31")
        sel = (day_values >= lo) & (day_values <= hi)
        if not sel.any():
            continue
        chunk = ds.isel(day=np.nonzero(sel)[0]).load()
        chunk_days = day_values[sel]
        d_ix = np.searchsorted(chunk_days, wanted[rows])
        d_ix_c = np.clip(d_ix, 0, chunk_days.size - 1)
        valid = chunk_days[d_ix_c] == wanted[rows]
        la = np.broadcast_to(lat_ix[rows, None], d_ix.shape)
        lo_ = np.broadcast_to(lon_ix[rows, None], d_ix.shape)
        for var in ds.data_vars:
            arr = chunk[var].transpose("day", "lat", "lon").values
            vals = arr[d_ix_c, la, lo_].astype("float64")
            vals[~valid | (la < 0) | (lo_ < 0)] = np.nan
            if var in KELVIN_VARS:
                vals = vals - 273.15
            agg = {f"{var}_sum{window_days}d": np.nansum(vals, axis=1)} if var in SUM_VARS else {}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                agg[f"{var}_mean{window_days}d"] = np.nanmean(vals, axis=1)
                if var in ("erc", "vpd", "tmmx"):
                    agg[f"{var}_max{window_days}d"] = np.nanmax(vals, axis=1)
            if var in SUM_VARS:
                agg[f"{var}_sum{window_days}d"][np.isnan(vals).all(axis=1)] = np.nan
            for name, v in agg.items():
                cols[name][rows] = v
    for name, v in cols.items():
        out[name] = v
    return out


# ---------------- LANDFIRE features ----------------

def landfire_version_for(year: int) -> int:
    """Closest LANDFIRE fuels version: <=2019 -> LF2016, 2020-2022 -> LF2022, >=2023 -> LF2024."""
    return 2016 if year <= 2019 else 2022 if year <= 2022 else 2024


def _sample_raster(da: xr.DataArray, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    from pyproj import Transformer

    x, y = Transformer.from_crs("EPSG:4326", da.rio.crs, always_xy=True).transform(lon, lat)
    xi = da.indexes["x"].get_indexer(x, method="nearest")
    yi = da.indexes["y"].get_indexer(y, method="nearest")
    vals = da.values[np.clip(yi, 0, None), np.clip(xi, 0, None)].astype("float64")
    vals[(xi < 0) | (yi < 0)] = np.nan
    return vals


def extract_landfire_features(samples: pd.DataFrame, bbox, static=DEFAULT_LANDFIRE_STATIC,
                              fuels=DEFAULT_LANDFIRE_FUELS, opener=None) -> pd.DataFrame:
    """Nearest-pixel LANDFIRE values at each sample. `opener(layer, bbox)` defaults to open_landfire."""
    if opener is None:
        from lakehouse import open_landfire as opener
    out = pd.DataFrame(index=samples.index)
    lon, lat = samples["lon"].values, samples["lat"].values
    for layer in static:
        vals = _sample_raster(opener(layer, bbox), lon, lat)
        name = layer.split("_", 1)[1].lower()
        if layer.endswith("_Asp"):
            rad = np.deg2rad(vals)
            out["aspect_sin"], out["aspect_cos"] = np.sin(rad), np.cos(rad)
        else:
            out[name] = vals
    version = pd.to_datetime(samples["date"]).dt.year.map(landfire_version_for).values
    for acronym in fuels:
        col = np.full(len(samples), np.nan)
        for v in np.unique(version):
            rows = version == v
            col[rows] = _sample_raster(opener(f"LF{v}_{acronym}", bbox), lon[rows], lat[rows])
        out[acronym.lower()] = col
    for c in out.columns.intersection(list(CATEGORICAL_LANDFIRE)):
        out[c] = out[c].round().astype("Int32")
    return out


# ---------------- per-region training table ----------------

def build_region_training(region: str, fires: gpd.GeoDataFrame, scheme: str | Mapping[str, str] = "state",
                          years: Sequence[int] | None = None, n_neg: int = 3, window_days: int = 7,
                          gridmet_vars=DEFAULT_GRIDMET_VARS, landfire: bool = True,
                          states: gpd.GeoDataFrame | None = None, open_gridmet=None, landfire_opener=None,
                          seed: int = 42) -> pd.DataFrame:
    """Fire (label 1) and sampled non-fire (label 0) cell-days for one region, with features.

    `fires` is the output of clean_mtbs(). Readers are injectable for testing.
    """
    if open_gridmet is None:
        from lakehouse import open_gridmet
    states = regions.read_states() if states is None else states
    mapping = regions.scheme_mapping(scheme)
    pos = fires[fires["state"].map(mapping) == region]
    if years is not None:
        pos = pos[pos["year"].isin(list(years))]
    if pos.empty:
        log.warning("%s: no fires in the requested years", region)
        return pd.DataFrame()

    bbox = regions.region_bbox(region, scheme, states)
    yrs = sorted(pos["year"].unique())
    wx_years = sorted(set(yrs) | {y - 1 for y in yrs})
    try:
        wx = open_gridmet(list(gridmet_vars), years=wx_years, bbox=bbox)
    except ValueError:  # year before the first study year is not in the lake
        wx = open_gridmet(list(gridmet_vars), years=yrs, bbox=bbox)
    mask = regions.region_mask(wx, scheme, states)
    code = regions.region_codes(scheme)[region]

    positives = pd.DataFrame({
        "event_id": pos["event_id"].values, "lat": pos["lat"].values, "lon": pos["lon"].values,
        "date": pos["ig_date"].values, "state": pos["state"].values, "burnbndac": pos["burnbndac"].values,
        "label": 1,
    })
    neg = sample_negatives(pos, mask, code, n_neg, perimeters=fires, seed=seed)
    neg["event_id"] = None
    neg["label"] = 0
    neg["burnbndac"] = 0.0
    neg_pts = gpd.GeoDataFrame(neg, geometry=gpd.points_from_xy(neg["lon"], neg["lat"]), crs="EPSG:4326")
    neg["state"] = regions.assign_states(neg_pts, states).values
    samples = pd.concat([positives, neg], ignore_index=True)
    samples["date"] = pd.to_datetime(samples["date"])

    feats = [extract_gridmet_features(wx, samples, window_days)]
    if landfire:
        feats.append(extract_landfire_features(samples, bbox, opener=landfire_opener))
    df = pd.concat([samples, *feats], axis=1)
    df["region"] = region
    df["year"] = df["date"].dt.year.astype("int32")
    df["month"] = df["date"].dt.month.astype("int8")
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    weather_cols = [c for c in feats[0].columns]
    before = len(df)
    df = df.dropna(subset=weather_cols, how="all")
    log.info("%s: %d fires, %d non-fire samples, %d dropped without weather",
             region, int(df["label"].sum()), int((df["label"] == 0).sum()), before - len(df))
    return df.reset_index(drop=True)


# ---------------- CLI ----------------

def _parse_years(text: str | None):
    if not text:
        return None
    if "-" in text:
        a, b = (int(x) for x in text.split("-"))
        return range(a, b + 1)
    return [int(x) for x in text.split(",")]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m cleaning.build_regions", description=__doc__.splitlines()[0])
    p.add_argument("target", choices=("fires", "training"))
    p.add_argument("--scheme", default="state", choices=list(regions.SCHEMES))
    p.add_argument("--regions", nargs="*", help="region names (default: every region in the scheme)")
    p.add_argument("--years", help="e.g. 2010-2020 or 2018,2020 (default: all study years)")
    p.add_argument("--neg", type=int, default=3, help="non-fire samples per fire")
    p.add_argument("--window", type=int, default=7, help="days of weather before ignition")
    p.add_argument("--include-prescribed", action="store_true", help="keep prescribed fires / WFU")
    p.add_argument("--no-landfire", action="store_true")
    p.add_argument("--out-dir", help="write local Parquet here instead of silver")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from lakehouse import read_mtbs, write_silver

    states = regions.read_states()
    types = None if args.include_prescribed else ("Wildfire",)
    fires = clean_mtbs(read_mtbs(), states, incident_types=types)

    def save(df, name, partition_cols, mode="overwrite"):
        if args.out_dir:
            from pathlib import Path
            path = Path(args.out_dir) / name
            path.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, partition_cols=partition_cols)
            print(f"wrote {len(df)} rows to {path}")
        else:
            print(f"wrote {len(df)} rows to silver table {write_silver(df, name, partition_cols, mode=mode)}")

    if args.target == "fires":
        save(fires, "mtbs_fires_regions", ["state"])
        return 0

    wanted = args.regions or regions.region_names(args.scheme)
    unknown = set(wanted) - set(regions.region_names(args.scheme))
    if unknown:
        p.error(f"unknown region(s) {sorted(unknown)} for scheme {args.scheme}: "
                f"choose from {', '.join(regions.region_names(args.scheme))}")
    for region in wanted:
        df = build_region_training(region, fires, args.scheme, _parse_years(args.years), args.neg, args.window,
                                   landfire=not args.no_landfire, states=states)
        if df.empty:
            continue
        if not args.out_dir:
            df["event_id"] = df["event_id"].astype("string")
        save(df, f"training_{args.scheme}", ["region"], mode="overwrite_partitions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
