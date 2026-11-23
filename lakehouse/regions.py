"""Group the western US into regions for per-region modelling.

A *scheme* maps state abbreviations to region names. Three are built in:

- "state":  one region per state (california, colorado, ...).
- "gacc":   NIFC Geographic Area Coordination Centers, approximated on state lines.
            These are the regions fire managers actually forecast for (and NFDRS
            ERC/BI are interpreted per GACC). Approximation notes: the real GACC
            lines split California (north/south ops), Idaho (panhandle -> Northern
            Rockies) and Wyoming (west -> Great Basin); here each state goes whole.
- "macro":  four large regions for when per-state samples are too small.

A scheme can also be a plain dict {"CA": "pacific", ...} for your own grouping.

    from lakehouse.regions import read_states, assign_regions, region_mask, region_polygons

State lines come from Natural Earth via cartopy (no extra ingest step). Fires do not
even need them: MTBS event_id starts with the state abbreviation.
    fires["region"] = assign_regions(fires, scheme="gacc")
    mask = region_mask(wx, scheme="state")          # region code per gridMET cell
"""
from __future__ import annotations

from collections.abc import Mapping

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box

from lakehouse import config

# Every state that touches config.WEST_BBOX. The last six only contribute slivers
# east of the Rockies; most schemes fold them into a neighbour.
STATE_NAMES = {
    "AZ": "arizona", "CA": "california", "CO": "colorado", "ID": "idaho", "MT": "montana",
    "NV": "nevada", "NM": "new_mexico", "OR": "oregon", "UT": "utah", "WA": "washington",
    "WY": "wyoming",
    "TX": "texas", "OK": "oklahoma", "KS": "kansas", "NE": "nebraska", "SD": "south_dakota",
    "ND": "north_dakota",
}
CORE_WEST = ("AZ", "CA", "CO", "ID", "MT", "NV", "NM", "OR", "UT", "WA", "WY")

SCHEMES: dict[str, dict[str, str]] = {
    "state": dict(STATE_NAMES),
    "gacc": {
        "WA": "northwest", "OR": "northwest",
        "CA": "california",
        "NV": "great_basin", "UT": "great_basin", "ID": "great_basin",
        "MT": "northern_rockies", "ND": "northern_rockies",
        "CO": "rocky_mountain", "WY": "rocky_mountain", "SD": "rocky_mountain",
        "NE": "rocky_mountain", "KS": "rocky_mountain",
        "AZ": "southwest", "NM": "southwest", "TX": "southwest", "OK": "southwest",
    },
    "macro": {
        "WA": "pacific_coast", "OR": "pacific_coast", "CA": "pacific_coast",
        "NV": "intermountain", "UT": "intermountain", "ID": "intermountain",
        "MT": "rockies", "WY": "rockies", "CO": "rockies",
        "ND": "rockies", "SD": "rockies", "NE": "rockies", "KS": "rockies",
        "AZ": "southwest", "NM": "southwest", "TX": "southwest", "OK": "southwest",
    },
}


def scheme_mapping(scheme: str | Mapping[str, str]) -> dict[str, str]:
    """State abbreviation -> region name for a built-in scheme name or a custom dict."""
    if isinstance(scheme, Mapping):
        unknown = set(scheme) - set(STATE_NAMES)
        if unknown:
            raise ValueError(f"Unknown state abbreviation(s) {sorted(unknown)}. Known: {', '.join(STATE_NAMES)}")
        return dict(scheme)
    if scheme not in SCHEMES:
        raise ValueError(f"Unknown region scheme {scheme!r}. Built in: {', '.join(SCHEMES)} (or pass a dict).")
    return SCHEMES[scheme]


def region_names(scheme: str | Mapping[str, str]) -> list[str]:
    return sorted(set(scheme_mapping(scheme).values()))


def region_codes(scheme: str | Mapping[str, str]) -> dict[str, int]:
    """Stable integer code per region (1-based; 0 = outside every region)."""
    return {name: i + 1 for i, name in enumerate(region_names(scheme))}


# ---------------- boundaries ----------------

_STATES_CACHE: dict[str, gpd.GeoDataFrame] = {}


def read_states(states: list[str] | None = None, resolution: str = "50m") -> gpd.GeoDataFrame:
    """Western state polygons (EPSG:4326), columns `state`, `geometry`, clipped to the lake bbox.

    Source: Natural Earth admin-1 boundaries via cartopy (already in requirements.txt).
    The first call downloads a few MB and cartopy caches it locally; no AWS needed.
    Or pass your own polygons to prepare_states().
    """
    if resolution not in _STATES_CACHE:
        import cartopy.io.shapereader as shpreader

        path = shpreader.natural_earth(resolution=resolution, category="cultural",
                                       name="admin_1_states_provinces")
        _STATES_CACHE[resolution] = gpd.read_file(path)
    return prepare_states(_STATES_CACHE[resolution], states)


def _state_codes(gdf: gpd.GeoDataFrame) -> pd.Series:
    if "STUSPS" in gdf.columns:  # Census
        return gdf["STUSPS"]
    if "iso_3166_2" in gdf.columns:  # Natural Earth: "US-CA"
        code = gdf["iso_3166_2"].astype("string")
        return code.str[3:].where(code.str.startswith("US-"))
    return gdf["state"]


def prepare_states(gdf: gpd.GeoDataFrame, states: list[str] | None = None) -> gpd.GeoDataFrame:
    """Normalise any state polygon layer (Census, Natural Earth or columns state/geometry)."""
    gdf = gdf if gdf.crs is not None else gdf.set_crs("EPSG:4326")
    out = gpd.GeoDataFrame({"state": _state_codes(gdf).values}, geometry=gdf.geometry.values, crs=gdf.crs)
    out = out[out["state"].isin(states or list(STATE_NAMES))].to_crs("EPSG:4326")
    out["geometry"] = out.geometry.intersection(box(*config.WEST_BBOX))
    return out[~out.geometry.is_empty].reset_index(drop=True)


def region_polygons(scheme: str | Mapping[str, str] = "state", states: gpd.GeoDataFrame | None = None
                    ) -> gpd.GeoDataFrame:
    """One (multi)polygon per region, EPSG:4326, columns `region`, `code`, `geometry`."""
    mapping = scheme_mapping(scheme)
    states = read_states() if states is None else states
    s = states[states["state"].isin(mapping)].copy()
    s["region"] = s["state"].map(mapping)
    out = s.dissolve(by="region", as_index=False)[["region", "geometry"]]
    out["code"] = out["region"].map(region_codes(scheme))
    return out.sort_values("code").reset_index(drop=True)


def region_bbox(region: str, scheme: str | Mapping[str, str] = "state",
                states: gpd.GeoDataFrame | None = None) -> tuple[float, float, float, float]:
    """(west, south, east, north) of one region: pass to open_gridmet / open_landfire."""
    polys = region_polygons(scheme, states)
    hit = polys[polys["region"] == region]
    if hit.empty:
        raise ValueError(f"Unknown region {region!r} for this scheme. Available: {', '.join(polys['region'])}")
    return tuple(float(v) for v in hit.total_bounds)


# ---------------- assigning things to regions ----------------

def state_from_event_id(event_id: pd.Series) -> pd.Series:
    """MTBS event_id starts with the state abbreviation (e.g. CA3456712345620200815)."""
    st = event_id.astype("string").str[:2].str.upper()
    return st.where(st.isin(list(STATE_NAMES)))


def assign_states(gdf: gpd.GeoDataFrame, states: gpd.GeoDataFrame | None = None) -> pd.Series:
    """State of each feature's representative point (points inside polygons), NaN outside the West."""
    states = read_states() if states is None else states
    pts = gpd.GeoDataFrame(geometry=gdf.geometry.to_crs("EPSG:4326").representative_point(), index=gdf.index)
    joined = gpd.sjoin(pts, states[["state", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    return joined["state"].reindex(gdf.index)


def assign_regions(gdf: gpd.GeoDataFrame, scheme: str | Mapping[str, str] = "state",
                   states: gpd.GeoDataFrame | None = None, state_col: str | None = None) -> pd.Series:
    """Region name per row. Uses `state_col` if given, else a spatial join on the geometry."""
    st = gdf[state_col] if state_col else assign_states(gdf, states)
    return st.map(scheme_mapping(scheme))


def region_mask(ds: xr.Dataset | xr.DataArray, scheme: str | Mapping[str, str] = "state",
                states: gpd.GeoDataFrame | None = None) -> xr.DataArray:
    """Integer region code for every cell of a lat/lon grid (gridMET). 0 = no region.

    Cell centres decide membership. Use region_codes(scheme) to map names to codes:
        mask = region_mask(wx, "state"); co = wx.where(mask == region_codes("state")["colorado"])
    """
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    lat, lon = ds["lat"].values, ds["lon"].values
    dx = float(abs(lon[1] - lon[0])) if lon.size > 1 else 1 / 24
    dy = float(abs(lat[1] - lat[0])) if lat.size > 1 else 1 / 24
    descending = lat.size < 2 or lat[0] > lat[-1]
    north = lat.max() + dy / 2
    transform = from_origin(lon.min() - dx / 2, north, dx, dy)
    polys = region_polygons(scheme, states)
    shapes = list(zip(polys.geometry, polys["code"].astype(int)))
    grid = rasterize(shapes, out_shape=(lat.size, lon.size), transform=transform, fill=0, dtype="int16")
    if not descending:
        grid = grid[::-1]
    mask = xr.DataArray(grid, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"), name="region_code")
    mask.attrs["regions"] = ", ".join(f"{c}={n}" for n, c in region_codes(scheme).items())
    return mask


def subset_region(ds: xr.Dataset, region: str, scheme: str | Mapping[str, str] = "state",
                  states: gpd.GeoDataFrame | None = None) -> xr.Dataset:
    """gridMET clipped to the region's bbox and masked (NaN) outside its polygon."""
    states = read_states() if states is None else states
    west, south, east, north = region_bbox(region, scheme, states)
    clipped = ds.sel(lon=slice(west, east), lat=slice(north, south))
    mask = region_mask(clipped, scheme, states)
    return clipped.where(mask == region_codes(scheme)[region])


__all__ = [
    "SCHEMES", "STATE_NAMES", "CORE_WEST", "assign_regions", "assign_states", "prepare_states",
    "read_states", "region_bbox", "region_codes", "region_mask", "region_names", "region_polygons",
    "scheme_mapping", "state_from_event_id", "subset_region",
]
