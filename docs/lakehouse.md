# Fire lakehouse: teammate guide

All project data lives in one S3 bucket in AWS (us-east-1). You do not download
anything by hand. You read the lake from Python with the `lakehouse` package in
this repo. Cleaned outputs go back into the lake with `write_silver` or `write_zarr`.

Scope: western US only (bbox lon -125 to -102, lat 31 to 49.5), study years
2002-2025, three sources (gridMET, MTBS, LANDFIRE). The goal is fire *proneness*
(susceptibility of an area), not predicting a single ignition.

## 1. Architecture

```
                 sources (internet)
   gridMET (THREDDS)   MTBS (USGS zip)   LANDFIRE (LFPS API)
          |                  |                  |
          v                  v                  v
   +----------------------------------------------------+
   | CodeBuild job "fire-lakehouse-ingest"               |  python -m ingest all
   |   1. ingest   2. integrate   3. catalog   4. viewer |
   +----------------------------------------------------+
          |
          v
   s3://fire-lakehouse-<account>/
     bronze/            raw, clipped to the West, values untouched      <- collection
     bronze/integrated/ all years combined per dataset, still raw       <- integration
     bronze/_manifest/  one JSON line per file: source URL, sha256, ...  <- lineage
     silver/            cleaned tables and grids written by teammates   <- cleaning,
                                                                           transformation,
                                                                           reduction
     viewer/index.html  what is in the lake
          |
          +--> Glue Data Catalog (database fire_lakehouse) --> Athena SQL
          +--> lakehouse.open_gridmet / read_mtbs / open_landfire --> your notebook
```

There are two layers:

- **Bronze** is written only by the ingest job. *Bronze is raw*: the only changes
  are spatial clipping to the West, the year filter, and compression. gridMET keeps
  its packed integers and scale factors. MTBS keeps its source CRS (EPSG:4269) and
  every incident type, prescribed burns included. LANDFIRE keeps its native
  projection (EPSG:5070) at 90 m.
- **Silver** is written by teammates. This is where cleaning, feature engineering,
  joins and reduction happen.

How this maps onto the course's preprocessing stages:

| Stage | Where it lives |
|---|---|
| Data collection | `bronze/gridmet`, `bronze/mtbs`, `bronze/landfire` + `bronze/_manifest` |
| Data integration | `bronze/integrated/` (all years in one object per variable or layer) |
| Cleaning, transformation, reduction | `silver/` |

### S3 layout

```
bronze/gridmet/variable={var}/year={year}/{var}_{year}_west.nc
bronze/mtbs/snapshot={YYYY-MM-DD}/mtbs_perimeters_west.parquet
bronze/mtbs/snapshot={YYYY-MM-DD}/_mtbs_perims_DD_metadata.xml
bronze/landfire/group={group}/tile={tile}/{lfps file}.tif
bronze/integrated/gridmet/{var}.zarr/
bronze/integrated/mtbs/mtbs_perimeters_west.parquet
bronze/integrated/landfire/{layer}.tif          (Cloud-Optimized GeoTIFF)
bronze/_manifest/{source}_{run_id}.jsonl
silver/{name}/                                  (Parquet + Glue table silver_{name})
silver/{name}.zarr/
viewer/index.html
```

## 2. Setup

1. Get your access key from the project owner through a private channel. Never put
   it in the repo, a notebook, `.env`, Slack or the report.
2. Add it to `~/.aws/credentials`:

   ```ini
   [fire-lakehouse-user]
   aws_access_key_id = ...
   aws_secret_access_key = ...
   ```

3. Add this to `~/.aws/config`. The role ARN comes with your key.

   ```ini
   [profile fire-lakehouse-user]
   region = us-east-1

   [profile fire-lakehouse]
   role_arn = arn:aws:iam::<account-id>:role/FireLakehouseRole
   source_profile = fire-lakehouse-user
   region = us-east-1
   ```

4. Install and select the profile:

   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   export AWS_PROFILE=fire-lakehouse
   aws sts get-caller-identity   # should show assumed-role/FireLakehouseRole
   ```

If a reader raises `LakehouseAuthError`, your profile is missing, expired or wrong.
Re-check step 4.

## 3. What's in the lake

```bash
python -m lakehouse.viewer
```

This downloads `viewer/index.html` and opens it in your browser. It shows:

- a gridMET variable x year matrix (raw, integrated or missing);
- MTBS fires per year by incident type;
- the LANDFIRE tile map and which groups are present;
- silver tables;
- the lineage log, searchable.

The ingest job republishes the viewer at the end of every run.

## 4. Reading

All readers take an optional `bbox = (west, south, east, north)` in lon/lat.
Downloads are cached in `~/.cache/fire-lakehouse` (override with
`FIRE_LAKEHOUSE_CACHE`), so the second call is local.

```python
from lakehouse import open_gridmet, read_mtbs, open_landfire, open_landfire_series, query

FOUR_CORNERS = (-114.1, 31.3, -102.0, 42.0)   # UT, CO, AZ, NM
CALIFORNIA   = (-124.5, 32.5, -114.1, 42.0)

# gridMET: xarray Dataset, dims (day, lat, lon), physical units.
# Reads the integrated Zarr lazily; only the chunks you touch are downloaded.
wx = open_gridmet(["tmmx", "erc", "vpd"], years=range(2015, 2021), bbox=FOUR_CORNERS)
summer_erc = wx["erc"].sel(day=wx.day.dt.month.isin([6, 7, 8])).mean("day").compute()

# MTBS: GeoDataFrame (EPSG:4269), all snapshots merged, deduplicated on event_id.
fires = read_mtbs()
wild = fires[fires["incid_type"] == "Wildfire"]

# LANDFIRE: xarray DataArray in EPSG:5070, 90 m.
elev = open_landfire("LF2020_Elev", bbox=CALIFORNIA)
fuel_models = open_landfire_series("FBFM40", bbox=CALIFORNIA)   # dim version = 2016, 2022, 2024

# SQL over the catalog (Athena; 10 GB scan cap per query).
query("SELECT * FROM bronze_inventory")
query("""
    SELECT year(date(ig_date)) AS yr, incid_type, count(*) AS n, sum(burnbndac) AS acres
    FROM bronze_mtbs_integrated GROUP BY 1, 2 ORDER BY 1
""")
```

Errors are explicit. An unknown variable or layer, or a year outside the lake,
raises `ValueError` and lists what is available.

## 5. Writing silver

```python
from lakehouse import write_silver, write_zarr, read_silver, open_zarr

write_silver(features_df, "fire_features", partition_cols=["year"])   # -> Glue table silver_fire_features
write_zarr(summer_erc_ds, "erc_summer_mean")                          # -> silver/erc_summer_mean.zarr

df = read_silver("fire_features")
query("SELECT count(*) FROM silver_fire_features")
```

Rules:

- Names are lowercase snake case: `^[a-z][a-z0-9_]*$`.
- `write_silver` overwrites by default. Pass `mode="append"` to add partitions.
- GeoDataFrames are stored with geometry as WKB. Rebuild them with
  `gpd.GeoSeries.from_wkb(df.geometry)`.
- Never write to `bronze/`. Bronze is rebuilt only by the ingest job, and silver
  must always be reproducible from it.

## 5b. Regions (per-region training data)

State lines come from Natural Earth through cartopy (already in `requirements.txt`; the
first call downloads a few MB and caches it). Fires don't need them at all: the MTBS
`event_id` starts with the state code. `lakehouse.regions` groups states into regions:

| Scheme | Regions |
|---|---|
| `state` | one per state: `california`, `colorado`, `arizona`, ... |
| `gacc` | NIFC coordination areas on state lines: `northwest` (WA, OR), `california`, `great_basin` (NV, UT, ID), `northern_rockies` (MT), `rocky_mountain` (CO, WY), `southwest` (AZ, NM) |
| `macro` | `pacific_coast`, `intermountain`, `rockies`, `southwest` |
| a dict | your own, e.g. `{"CA": "coast", "OR": "coast", "CO": "rockies"}` |

Plains/Texas slivers inside the bbox (TX, OK, KS, NE, SD, ND) fold into the nearest region.

```python
from lakehouse import open_gridmet, read_silver
from lakehouse.regions import read_states, region_bbox, subset_region, region_mask

co_box = region_bbox("colorado")                       # pass to any reader
wx = open_gridmet(["erc", "vpd"], years=[2020], bbox=co_box)
co_wx = subset_region(wx, "colorado")                  # NaN outside the state line
```

Build the silver tables (fires are cleaned once; training rows are 1 = fire,
0 = random unburned cell-day in the same region, dates drawn from that region's fires,
weather from the 7 days *before* the date):

```bash
python -m cleaning.build_regions fires                                   # silver_mtbs_fires_regions
python -m cleaning.build_regions training --scheme state --regions california colorado
python -m cleaning.build_regions training --scheme gacc --years 2010-2025
python -m cleaning.build_regions training --regions colorado --out-dir out/   # local only
```

```python
train = read_silver("training_state")
co = train[train.region == "colorado"]
```

Split train/test **by year** (or spatial blocks), not randomly: neighbouring cell-days
share weather, so a random split overstates accuracy.

## 6. Changing study years

The years are CloudFormation parameters (`StudyStartYear`, `StudyEndYear`). The
project owner changes them and refills the lake:

```bash
infra/deploy.sh fire-admin <budget-email> <landfire-email> 1995 2025
infra/run_ingest.sh fire-admin all --medium
```

The job is idempotent. Files that already exist are skipped. New years are
downloaded. The integrated Zarr is appended to (later years) or rebuilt (earlier
years). Readers pick the new range up from the stack outputs automatically.

## 7. Cost and lifetime

- Storage is an estimated 30-45 GB of S3 once full (about $0.70-1.00/month).
  Athena is $5 per TB scanned, capped at 10 GB per query. CodeBuild is billed per
  build minute (about $0.005/min small, $0.01/min medium).
- An AWS Budget emails the owner at $5/month. If actual spend passes $5, a budget
  action automatically blocks new ingest builds and Athena queries (reading files
  from S3 keeps working). The owner resets it in the AWS Budgets console.
- `athena-results/` expires after 7 days.
- The stack is torn down on **2026-12-03** with `infra/teardown.sh`. That
  deletes every object. Copy anything you need for the report before then.

## 8. Data dictionary

### gridMET (daily, about 4 km, EPSG:4326)

| Code | Variable | Units |
|---|---|---|
| `pr` | precipitation | mm |
| `tmmx` / `tmmn` | max / min air temperature | K |
| `rmax` / `rmin` | max / min relative humidity | % |
| `vs` | wind speed at 10 m | m/s |
| `th` | wind direction (from) | degrees clockwise from north |
| `srad` | downward shortwave radiation | W/m² |
| `vpd` | mean vapor pressure deficit | kPa |
| `erc` | energy release component (NFDRS) | index |
| `bi` | burning index (NFDRS) | index |
| `fm100` / `fm1000` | 100-hour / 1000-hour dead fuel moisture | % |

Bronze stores the source's packed `uint16` values with `scale_factor` and
`add_offset`. xarray unpacks them on read, so you see physical units.

### LANDFIRE (90 m, EPSG:5070, 5° tiles mosaicked per layer)

| Group | Layers | Meaning |
|---|---|---|
| `topo` | `LF2020_Elev`, `LF2020_SlpD`, `LF2020_Asp` | elevation (m), slope (degrees), aspect (degrees) |
| `regime` | `LF2016_FRG`, `LF2016_FRI`, `LF2016_BPS` | fire regime group, mean fire return interval (years), biophysical setting |
| `fuels_LF{2016,2022,2024}` | `FBFM40`, `EVT`, `EVC`, `EVH`, `CC` | Scott & Burgan 40 fuel model, existing vegetation type / cover / height, canopy cover |

`FBFM40`, `EVT`, `EVC`, `EVH`, `FRG` and `BPS` are categorical codes. Look up their
meanings in the LANDFIRE attribute tables before treating them as numbers.
LANDFIRE has no fuels before 2016, so earlier years use LF2016 as an
approximation. Say so in the report.

### MTBS perimeters (GeoParquet, EPSG:4269)

| Column | Meaning |
|---|---|
| `event_id` | unique fire ID (dedup key) |
| `irwinid` | IRWIN incident ID, when known |
| `incid_name` | incident name |
| `incid_type` | `Wildfire`, `Prescribed Fire`, `Wildland Fire Use`, `Unknown`, ... |
| `ig_date` | ignition date |
| `burnbndac` | burned area within the burn boundary (acres) |
| `burnbndlat` / `burnbndlon` | burn boundary centroid |
| `map_id`, `map_prog`, `asmnt_type`, `perim_id` | MTBS mapping program and assessment type |
| `pre_id` / `post_id` | Landsat scenes used for the dNBR |
| `dnbr_offst`, `dnbr_stddv` | dNBR offset and standard deviation |
| `nodata_t`, `incgreen_t`, `low_t`, `mod_t`, `high_t` | dNBR severity class thresholds |
| `comment` | analyst comment |
| `snapshot` | (Athena table `bronze_mtbs_perimeters` only) download date partition |

Prescribed fires are **included** in bronze. Filter them out (or keep them as a
separate class) in silver.

### Lineage (`bronze_manifest` table)

One row per file written: `source`, `s3_key`, `source_url`, `sha256`, `bytes`,
`downloaded_at`, `bbox`, `params` (JSON), `git_sha` (the code version that wrote it).

## Legacy local data

`scripts/download_raw_data.py` is gone. `classifier/WildfireClassifier.py`,
`scripts/make-visuals.py` and the notebooks still read `data/raw/` and will stop
working once local data is deleted. Port them to the readers in section 4.
