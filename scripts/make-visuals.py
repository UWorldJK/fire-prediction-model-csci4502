from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rasterio.features import rasterize
from rasterio.plot import plotting_extent

REPO = Path(__file__).resolve().parent.parent
RAW = Path(__file__).resolve().parent / "data" / "raw" 
FIGS = REPO / "figures"

# Scott & Burgan 40 fuel model codes (FBFM40) grouped by fuel type
FUEL_GROUPS = [
    (91, 99, "Non-burnable", "#bdbdbd"),
    (101, 109, "Grass", "#f2e26b"),
    (121, 124, "Grass-shrub", "#c8d96f"),
    (141, 149, "Shrub", "#8fb35b"),
    (161, 165, "Timber-understory", "#3f7f4a"),
    (181, 189, "Timber litter", "#1f4d2b"),
    (201, 204, "Slash-blowdown", "#8c5a3c"),
]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# loading

def load_mtbs():
    """Load the most recent MTBS GeoPackage written by download_raw_data.py."""
    files = sorted(RAW.glob("mtbs_*.gpkg"))
    if not files:
        return None
    gdf = gpd.read_file(files[-1])
    gdf = gdf.rename(columns=str.lower)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    gdf["ig_date"] = pd.to_datetime(gdf["ig_date"])
    gdf["type"] = gdf["incid_type"] if "incid_type" in gdf else "Fire"
    gdf["acres"] = pd.to_numeric(gdf.get("burnbndac"), errors="coerce")
    print(f"MTBS: {len(gdf)} fires from {files[-1].name}")
    return gdf


def _layer_key(text):
    t = (text or "").lower()
    for key, tag in (("elev", "elev"), ("slope", "slp"), ("fuel", "fbfm")):
        if tag in t:
            return key
    return None


def load_landfire():
    """
    Return {"elev"|"slope"|"fuel": dict(data, transform, crs, extent)}.

    LFPS returns one multi-band GeoTIFF with bands in the order requested
    (Elev, SlpD, FBFM40); band descriptions are used when present.
    """
    layers = {}
    order = ["elev", "slope", "fuel"]
    for tif in sorted(RAW.glob("landfire_*/**/*.tif")):
        with rasterio.open(tif) as src:
            for b in range(1, src.count + 1):
                key = _layer_key(src.descriptions[b - 1]) or _layer_key(tif.name)
                if key is None and src.count == len(order):
                    key = order[b - 1]
                if key is None or key in layers:
                    continue
                arr = src.read(b, masked=True).astype("float64")
                # LANDFIRE fills missing cells with -9999 / 32767 even when nodata isn't set
                arr = np.ma.masked_where(
                    np.ma.getmaskarray(arr) | (arr <= -9999) | (arr == 32767), arr
                )
                layers[key] = dict(
                    data=arr,
                    transform=src.transform,
                    crs=src.crs,
                    extent=plotting_extent(src),
                )
    if layers:
        print(f"LANDFIRE layers: {', '.join(layers)}")
    return layers


# helpers

def save(fig, name):
    FIGS.mkdir(exist_ok=True)
    path = FIGS / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(REPO)}")


def geo_aspect(ax, crs, ymin, ymax):
    """Keep lat/lon maps from looking stretched."""
    if crs is None or crs.is_geographic:
        ax.set_aspect(1 / np.cos(np.radians((ymin + ymax) / 2)))
    else:
        ax.set_aspect("equal")


def marker_sizes(acres):
    return np.clip(np.sqrt(acres.fillna(0).to_numpy()) / 4, 2, 300)


def fuel_group_codes(fuel):
    """Map FBFM40 codes to indexes into FUEL_GROUPS (masked where unknown)."""
    codes = fuel.filled(-1)
    out = np.full(codes.shape, -1, dtype=int)
    for i, (lo, hi, _, _) in enumerate(FUEL_GROUPS):
        out[(codes >= lo) & (codes <= hi)] = i
    return np.ma.masked_where(out < 0, out)


def burned_mask(layer, gdf):
    """True for raster cells touched by any fire perimeter."""
    fires = gdf.to_crs(layer["crs"]) if layer["crs"] else gdf
    return rasterize(
        ((geom, 1) for geom in fires.geometry),
        out_shape=layer["data"].shape,
        transform=layer["transform"],
        fill=0,
        all_touched=True,  # fires are often smaller than one ~4 km cell
        dtype="uint8",
    ).astype(bool)


# MTBS plots

def plot_fire_map(gdf):
    fig, ax = plt.subplots(figsize=(10, 8))
    for fire_type, sub in gdf.groupby("type"):
        pts = sub.geometry.representative_point()
        ax.scatter(pts.x, pts.y, s=marker_sizes(sub["acres"]), alpha=0.5,
                   edgecolor="none", label=f"{fire_type} ({len(sub)})")
    leg = ax.legend(title="Incident type (count)", loc="lower right")
    for h in getattr(leg, "legend_handles", None) or leg.legendHandles:
        h.set_sizes([40])
    ymin, ymax = ax.get_ylim()
    geo_aspect(ax, gdf.crs, ymin, ymax)
    year = f"{gdf['ig_date'].dt.year.min()}–{gdf['ig_date'].dt.year.max()}"
    ax.set(title=f"MTBS fires {year} (marker size ~ burned acres)",
           xlabel="Longitude", ylabel="Latitude")
    ax.grid(alpha=0.3)
    save(fig, "01_fire_locations")


def plot_fires_by_month(gdf):
    month = gdf["ig_date"].dt.month
    counts = (gdf.groupby([month, "type"]).size()
              .unstack(fill_value=0).reindex(range(1, 13), fill_value=0))
    acres = (gdf.groupby([month, "type"])["acres"].sum()
             .unstack(fill_value=0).reindex(range(1, 13), fill_value=0))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    counts.plot(kind="bar", stacked=True, ax=axes[0], width=0.85)
    axes[0].set(title="Fires by ignition month", ylabel="Number of fires")
    (acres / 1000).plot(kind="bar", stacked=True, ax=axes[1], width=0.85, legend=False)
    axes[1].set(title="Area burned by ignition month", ylabel="Thousand acres", xlabel="")
    axes[1].set_xticklabels(MONTHS, rotation=0)
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(title="Incident type")
    save(fig, "02_fires_by_month")


def plot_fire_sizes(gdf):
    sizes = gdf[gdf["acres"] > 0]
    if sizes.empty:
        return
    bins = np.logspace(np.log10(sizes["acres"].min()), np.log10(sizes["acres"].max()), 30)
    groups = list(sizes.groupby("type"))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist([g["acres"] for _, g in groups], bins=bins, stacked=True,
            label=[t for t, _ in groups])
    ax.set_xscale("log")
    ax.set(title="Fire size distribution", xlabel="Burned area (acres, log scale)",
           ylabel="Number of fires")
    ax.legend(title="Incident type")
    ax.grid(alpha=0.3)
    save(fig, "03_fire_sizes")


# LANDFIRE plots

def plot_landfire(layers, gdf):
    panels = [k for k in ("elev", "slope", "fuel") if k in layers]
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 6), squeeze=False)
    for ax, key in zip(axes[0], panels):
        layer = layers[key]
        ext = layer["extent"]
        if key == "elev":
            im = ax.imshow(layer["data"], extent=ext, cmap="terrain")
            fig.colorbar(im, ax=ax, shrink=0.7, label="Elevation (m)")
            ax.set_title("Elevation")
        elif key == "slope":
            im = ax.imshow(layer["data"], extent=ext, cmap="magma")
            fig.colorbar(im, ax=ax, shrink=0.7, label="Slope (degrees)")
            ax.set_title("Slope")
        else:
            cmap = ListedColormap([c for *_, c in FUEL_GROUPS])
            ax.imshow(fuel_group_codes(layer["data"]), extent=ext, cmap=cmap,
                      vmin=-0.5, vmax=len(FUEL_GROUPS) - 0.5, interpolation="nearest")
            ax.legend(handles=[Patch(color=c, label=n) for _, _, n, c in FUEL_GROUPS],
                      loc="lower right", fontsize=8)
            ax.set_title("Fuel type (FBFM40 groups)")
        if gdf is not None:
            fires = gdf.to_crs(layer["crs"]) if layer["crs"] else gdf
            pts = fires.geometry.representative_point()
            ax.scatter(pts.x, pts.y, s=4, c="red", alpha=0.6, edgecolor="none")
        geo_aspect(ax, layer["crs"], ext[2], ext[3])
    if gdf is not None:
        fig.suptitle("LANDFIRE layers (red dots = MTBS fires)")
    fig.tight_layout()
    save(fig, "04_landfire_layers")


def plot_fuel_shares(fuel_layer, burned):
    groups = fuel_group_codes(fuel_layer["data"])
    valid = ~np.ma.getmaskarray(groups)
    n = len(FUEL_GROUPS)
    region = np.bincount(groups.data[valid], minlength=n) / max(valid.sum(), 1)
    in_fire = valid & burned
    fires = np.bincount(groups.data[in_fire], minlength=n) / max(in_fire.sum(), 1)

    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.2, region * 100, 0.4, label="Whole region", color="#9e9e9e")
    ax.bar(x + 0.2, fires * 100, 0.4, label="Burned cells", color="#d9480f")
    ax.set_xticks(x, [name for _, _, name, _ in FUEL_GROUPS], rotation=20)
    ax.set(title="Fuel type: burned cells vs. the whole region",
           ylabel="Share of cells (%)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save(fig, "05_fuel_burned_vs_region")


def plot_terrain_burned_vs_not(layers, gdf):
    panels = [(k, label) for k, label in
              (("elev", "Elevation (m)"), ("slope", "Slope (degrees)")) if k in layers]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 4.5), squeeze=False)
    for ax, (key, label) in zip(axes[0], panels):
        layer = layers[key]
        burned = burned_mask(layer, gdf)
        valid = ~np.ma.getmaskarray(layer["data"])
        data = layer["data"].data
        bins = np.linspace(np.percentile(data[valid], 0.5), np.percentile(data[valid], 99.5), 50)
        ax.hist(data[valid & ~burned], bins=bins, density=True, alpha=0.6,
                label="Unburned", color="#9e9e9e")
        ax.hist(data[valid & burned], bins=bins, density=True, alpha=0.6,
                label="Burned", color="#d9480f")
        ax.set(xlabel=label, ylabel="Density")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Terrain of burned vs. unburned cells")
    fig.tight_layout()
    save(fig, "06_terrain_burned_vs_unburned")


# main

def main():
    gdf = load_mtbs()
    layers = load_landfire()
    if gdf is None and not layers:
        print(f"No data found in {RAW}. Run download_raw_data.py first.")
        return

    if gdf is not None:
        plot_fire_map(gdf)
        plot_fires_by_month(gdf)
        plot_fire_sizes(gdf)

    if layers:
        plot_landfire(layers, gdf)
        if gdf is not None:
            if "fuel" in layers:
                plot_fuel_shares(layers["fuel"], burned_mask(layers["fuel"], gdf))
            plot_terrain_burned_vs_not(layers, gdf)

    print(f"Done. Figures are in {FIGS}")


if __name__ == "__main__":
    main()