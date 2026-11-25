import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import random
import os

GRIDMET_VARS = ["pr", "tmmx", "rmin", "vs", "erc", "fm100"]

def extract_window_features(ds, lat, lon, date, short_window=7, long_windows=[30, 90]):
    """
    Extract temporal features for a single (lat, lon, date) sample.
    Returns a flat dict of features, or None if extraction fails.
    """
    end = pd.Timestamp(date)
    features = {}

    try:
        point = ds.sel(lat=lat, lon=lon, method="nearest")
    except Exception:
        return None

    # --- Short window (7-day) stats ---
    short_start = end - pd.Timedelta(days=short_window)
    try:
        window = point.sel(day=slice(str(short_start.date()), str(end.date())))
        for var in GRIDMET_VARS:
            vals = window[var].values
            features[f"{var}_mean7"]  = float(np.mean(vals))
            features[f"{var}_std7"]   = float(np.std(vals))
            features[f"{var}_min7"]   = float(np.min(vals))
            features[f"{var}_max7"]   = float(np.max(vals))
            if len(vals) > 1:
                features[f"{var}_trend7"] = float(np.polyfit(range(len(vals)), vals, 1)[0])
            else:
                features[f"{var}_trend7"] = 0.0
    except Exception:
        return None

    # --- Long window rolling means (30, 90 day) for precip only ---
    for w in long_windows:
        long_start = end - pd.Timedelta(days=w)
        try:
            long_window = point.sel(day=slice(str(long_start.date()), str(end.date())))
            features[f"pr_mean{w}"] = float(long_window["pr"].mean())
        except Exception:
            features[f"pr_mean{w}"] = np.nan

    # --- Calendar features (cyclical encoding) ---
    day_of_year = end.day_of_year
    features["month"]   = end.month
    features["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365)
    features["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365)

    # --- Interaction features ---
    features["temp_x_wind"]    = features["tmmx_mean7"] * features["vs_mean7"]
    features["dry_fire_index"] = features["erc_mean7"] / (features["fm100_mean7"] + 1e-6)

    return features


def build_feature_matrix(gridmet_path, mtbs_path, n_negatives_per_positive=3):
    """
    Build a labeled feature matrix from GridMET and MTBS files.
    Returns a DataFrame ready for model training.
    """
    print("Loading datasets...")
    ds = xr.open_dataset(gridmet_path)
    fires = gpd.read_file(mtbs_path).to_crs("EPSG:4326")
    fires = fires.dropna(subset=["ig_date"])
    fires["ig_date"] = pd.to_datetime(fires["ig_date"])

    all_dates = pd.to_datetime(ds.day.values)
    lat_min, lat_max = float(ds.lat.min()), float(ds.lat.max())
    lon_min, lon_max = float(ds.lon.min()), float(ds.lon.max())
    date_min, date_max = all_dates.min(), all_dates.max()

    records = []

    # --- Positive samples (actual fires) ---
    print(f"Extracting features for {len(fires)} fire events...")
    for _, fire in fires.iterrows():
        lat  = fire.geometry.centroid.y
        lon  = fire.geometry.centroid.x
        date = fire["ig_date"]
        if date < date_min + pd.Timedelta(days=90):
            continue  # not enough history for long-window features
        feats = extract_window_features(ds, lat, lon, date)
        if feats:
            feats["label"] = 1
            records.append(feats)

    # --- Negative samples (random non-fire locations/dates) ---
    n_negatives = len(records) * n_negatives_per_positive
    print(f"Sampling {n_negatives} negative examples...")
    attempts  = 0
    neg_count = 0
    while neg_count < n_negatives and attempts < n_negatives * 10:
        attempts += 1
        lat  = random.uniform(lat_min, lat_max)
        lon  = random.uniform(lon_min, lon_max)
        date = date_min + pd.Timedelta(
            days=random.randint(90, (date_max - date_min).days)
        )
        feats = extract_window_features(ds, lat, lon, date)
        if feats:
            feats["label"] = 0
            records.append(feats)
            neg_count += 1

    df = pd.DataFrame(records).dropna()
    print(f"Feature matrix shape: {df.shape}")
    print(f"Label balance:\n{df['label'].value_counts()}")
    return df


if __name__ == "__main__":
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = build_feature_matrix(
        gridmet_path=os.path.join(BASE, "data/raw/gridmet_2020-01-01_2020-12-31.nc"),
        mtbs_path=os.path.join(BASE, "data/raw/mtbs_2020_2020.gpkg"),
    )
    out_path = os.path.join(BASE, "data/processed/features_2020.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")