import os
import random

import geopandas as gpd
import pandas as pd
import xarray as xr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class WildfireClassifier:
    def __init__(self, gridmet_path, mtbs_path, n_negatives_per_positive=3):
        self.gridmet_path = gridmet_path
        self.mtbs_path = mtbs_path
        self.n_neg = n_negatives_per_positive
        self.model = None
        self.feature_cols = None

    def _extract_features(self, ds, lat, lon, date, window_days=7):
        end = pd.Timestamp(date)
        start = end - pd.Timedelta(days=window_days)
        try:
            window = ds.sel(
                lat=lat, lon=lon, method="nearest"
            ).sel(day=slice(str(start.date()), str(end.date())))
            return {var: float(window[var].mean()) for var in ds.data_vars}
        except Exception:
            return None

    def build_features(self):
        print("Loading data...")
        ds = xr.open_dataset(self.gridmet_path)
        fires = gpd.read_file(self.mtbs_path)
        fires["ig_date"] = pd.to_datetime(fires["ig_date"])
        fires = fires.dropna(subset=["ig_date"])

        lat_min, lat_max = float(ds.lat.min()), float(ds.lat.max())
        lon_min, lon_max = float(ds.lon.min()), float(ds.lon.max())

        rows = []

        print(f"Extracting features for {len(fires)} fires...")
        for _, fire in fires.iterrows():
            centroid = fire.geometry.centroid
            feats = self._extract_features(ds, centroid.y, centroid.x, fire["ig_date"])
            if feats:
                feats["label"] = 1
                rows.append(feats)

        print(f"Sampling {len(fires) * self.n_neg} negative examples...")
        dates = fires["ig_date"].tolist()
        for _ in range(len(fires) * self.n_neg):
            lat = random.uniform(lat_min, lat_max)
            lon = random.uniform(lon_min, lon_max)
            date = random.choice(dates)
            feats = self._extract_features(ds, lat, lon, date)
            if feats:
                feats["label"] = 0
                rows.append(feats)

        ds.close()
        df = pd.DataFrame(rows).dropna()
        self.feature_cols = [c for c in df.columns if c != "label"]
        print(f"Feature matrix: {df.shape} — fires={df.label.sum()}, non-fires={(df.label==0).sum()}")
        return df

    def train(self, df):
        X = df[self.feature_cols]
        y = df["label"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print("Training Random Forest...")
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)
        print("\n--- Evaluation ---")
        print(classification_report(y_test, preds, target_names=["No Fire", "Fire"]))

        importances = sorted(
            zip(self.feature_cols, self.model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
        print("\nFeature importances:")
        for name, imp in importances:
            print(f"  {name}: {imp:.3f}")

        return self.model


if __name__ == "__main__":
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    clf = WildfireClassifier(
        gridmet_path=os.path.join(BASE, "data/raw/gridmet_2020-01-01_2020-12-31.nc"),
        mtbs_path=os.path.join(BASE, "data/raw/mtbs_2020_2020.gpkg"),
    )
    df = clf.build_features()
    clf.train(df)