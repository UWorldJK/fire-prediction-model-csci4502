# Fire lakehouse: how everything works

This is the long version. `docs/lakehouse.md` is the short guide teammates need
to use the lake. This document explains every piece: what it does, why it was
built that way, how data moves through it, what it costs, and what is still left
to do. It is written for the project owner and for the "data preprocessing"
section of the course report.

---

## 1. The problem it solves

Before the lakehouse, the project worked like this:

- Every teammate ran `scripts/download_raw_data.py` on their own laptop.
- That script pulled gridMET, MTBS and LANDFIRE data into `data/raw/`. It covered
  only a year or two, for the whole country, and ended up at about 1.6 GB locally.
- Each notebook had its own path conventions (`data/raw/gridmet_2020-01-01_2020-12-31.nc`,
  `mtbs_2020_2020.gpkg`, ...) and its own cleaning.
- Nobody could say exactly which file version a result came from.

The project's goal is **fire proneness**: which areas of the western US are
susceptible to fire given their weather, fuels and terrain. The goal is not to
predict one ignition on one day. That needs long records: many years of weather,
every mapped fire, and fuels and terrain everywhere. So the lake holds about 24
years (2002-2025) for the western US only.

The lakehouse replaces the local scripts with one shared, versioned, documented
copy of the data in AWS. It also shows, step by step, the preprocessing stages
the course asks for:

| Course stage | Lakehouse piece |
|---|---|
| Data collection | `bronze/` raw files + lineage manifest |
| Data integration | `bronze/integrated/` (all years combined per dataset) |
| Data cleaning | `silver/` (teammate scripts) |
| Data transformation | `silver/` |
| Data reduction | `silver/` (and the spatial/temporal clip in bronze) |

---

## 2. The big picture

```
   Internet sources                       AWS account (us-east-1)
   ----------------                       -------------------------------------------------
   gridMET  (THREDDS, NetCDF)    --+
   MTBS     (USGS, shapefile zip) --+-->  CodeBuild project  fire-lakehouse-ingest
   LANDFIRE (LFPS job API, GeoTIFF)-+       runs:  python -m ingest all
                                              1. ingest   -> bronze/<source>/...
                                              2. integrate-> bronze/integrated/...
                                              3. catalog  -> Glue tables + Athena view
                                              4. viewer   -> viewer/index.html
                                              every write -> bronze/_manifest/*.jsonl
                                                       |
                                                       v
                                          S3 bucket  fire-lakehouse-<account-id>
                                                       |
                         +-----------------------------+-----------------------------+
                         |                             |                             |
                  Glue Data Catalog              lakehouse Python pkg          viewer/index.html
                  db fire_lakehouse              open_gridmet, read_mtbs,      (static HTML
                         |                       open_landfire, query,          snapshot of lake)
                  Athena workgroup               write_silver, write_zarr
                  fire-lakehouse                        |
                  (10 GB/query cap)              teammates' notebooks and models
```

Everything is **serverless**. No server, database instance or cluster runs
between jobs, so nothing bills by the hour while idle:

- **S3** stores all data and bills per GB-month.
- **Glue Data Catalog** stores table definitions only. It is free at this size.
- **Athena** runs SQL directly on S3 files and bills per byte scanned.
- **CodeBuild** is the compute for ingest. It starts a container, runs the job,
  then stops. It bills per minute of runtime.
- **CloudWatch Logs** keeps the job logs for 7 days.
- **AWS Budgets** emails you before spending gets close to $5 in a month, and
  blocks new builds and Athena queries once actual spend passes $5 (section 14).

Everything is created by one CloudFormation template, `infra/lakehouse.yaml`. You
apply it once and the whole lake exists. You delete the stack and it is gone.

### Why us-east-1

The AWS account belongs to an AWS Organization. One of its Service Control
Policies (a guardrail set by the org admin) blocks CloudFormation in us-west-2.
us-east-1 is allowed. The lake does not need to sit near the West: the data is
pulled from the internet, and S3 reads from Colorado are fast enough in any US
region.

### Why not Iceberg, Redshift or Lake Formation

Your requirements were: cheapest possible, two layers, until Dec 3, and people
just pull data. Those tools add cost or setup without helping with that:

- **Iceberg tables** add ACID transactions, time travel and schema evolution.
  Only two writers exist (the ingest job and teammates' silver writes), and they
  never write the same table. Plain Parquet is enough.
- **Redshift** bills per hour.
- **Lake Formation** permissions add complexity that one shared role does not need.

---

## 3. The two layers in detail

### 3.1 Bronze: raw, but only what we need

**Rule: bronze is raw.** Only three changes are allowed:

1. **Spatial subsetting** to the western US bounding box: lon -125 to -102,
   lat 31 to 49.5. That covers WA, OR, CA, NV, ID, MT, WY, UT, CO, AZ, NM. It
   excludes Alaska, Hawaii and everything east of roughly the Colorado/Kansas line.
2. **Temporal subsetting** to the study years.
3. **Compression** and a cloud-friendly file format.

No unit conversion, no cleaning, no dropping rows, no reprojection. Why so strict:

- The cleaning teammate sees exactly what the source published and makes every
  cleaning decision explicitly in silver, where it is visible and reviewable.
- If a cleaning choice turns out to be wrong, silver can be rebuilt from bronze
  without downloading anything again.
- For the report, you can state the preprocessing honestly: every
  transformation lives in one layer.

Concretely per source:

| Source | What's kept exactly as published |
|---|---|
| gridMET | packed `uint16` values with `scale_factor`/`add_offset`, source variable names and attributes, `day` coordinate |
| MTBS | every column, the source CRS (EPSG:4269, NAD83), every incident type (wildfire, prescribed fire, wildland fire use, unknown) |
| LANDFIRE | native CRS (EPSG:5070, CONUS Albers), source codes (categorical layers stay as codes) |

The one compromise is **LANDFIRE at 90 m** rather than its native 30 m. At 30 m
the western US is about 9x larger, roughly 6 billion pixels per layer. That would
mean far more LFPS jobs and far more storage and memory for each mosaic. The
LANDFIRE Product Service resamples on its side, so the values are still source
values, just at a coarser grid. gridMET is 4 km and the modeling unit will be
coarser than 90 m anyway. The resolution is one constant (`LANDFIRE_RESOLUTION_M`
in `lakehouse/config.py`) if you ever want 30 m.

### 3.2 Bronze/integrated: all years combined

The course separates **integration** from collection, and so does the lake.
Integrated objects are still raw values. They are only concatenated or mosaicked,
never cleaned, and they can be rebuilt from bronze raw at any time.

- **gridMET**: one **Zarr** store per variable (`bronze/integrated/gridmet/tmmx.zarr`)
  with every year on the `day` dimension. Zarr splits the array into small chunks
  (366 days x 64 x 64 cells). Reading a summer of ERC over Colorado downloads only
  the chunks that cover it, not 24 years of files.
- **MTBS**: one GeoParquet. It is the union of every snapshot ever ingested,
  deduplicated on `event_id`, with the latest snapshot winning. MTBS republishes
  and corrects its composite file, so snapshots preserve history while the
  integrated file gives one clean "current" view.
- **LANDFIRE**: one **Cloud-Optimized GeoTIFF (COG)** per layer, mosaicked from
  all 20 tiles. It is one file to open instead of 20.

### 3.3 Silver: cleaned, teammate-owned

Teammates write silver with two functions:

- `write_silver(df, name)` writes a Parquet dataset to `silver/{name}/` **and**
  registers the Glue table `silver_{name}`, so it is immediately queryable in
  Athena. GeoDataFrames are stored with geometry as WKB bytes.
- `write_zarr(ds, name)` writes a gridded dataset to `silver/{name}.zarr`.

Names must be lowercase snake case. That keeps S3 keys and Glue table names valid
and consistent. Bronze is never written by teammates.

---

## 4. The S3 layout, key by key

```
s3://fire-lakehouse-<account-id>/
  bronze/
    gridmet/variable={var}/year={year}/{var}_{year}_west.nc
    mtbs/snapshot={YYYY-MM-DD}/mtbs_perimeters_west.parquet
    mtbs/snapshot={YYYY-MM-DD}/_mtbs_perims_DD_metadata.xml
    landfire/group={group}/tile={tile}/{lfps file}.tif
    integrated/gridmet/{var}.zarr/
    integrated/mtbs/mtbs_perimeters_west.parquet
    integrated/landfire/{layer}.tif
    _manifest/{source}_{run_id}.jsonl
  silver/{name}/
  silver/{name}.zarr/
  viewer/index.html
  _code/ingest.zip
  athena-results/
```

Why it looks like this:

- **`key=value` folder names** (`variable=tmmx/year=2020`) are Hive-style
  partitions. Athena and Glue read them as columns. `snapshot` becomes a column
  of the `bronze_mtbs_perimeters` table automatically.
- **`_west` suffix**: a file name alone tells you it is the clipped version.
- **Leading underscore** on the MTBS metadata XML: Athena ignores files starting
  with `_` or `.`. Without it, Athena would try to read the XML as Parquet and
  every MTBS query would fail.
- **`_manifest/`, `_code/`** are underscored for the same reason, and to sort them
  apart from data.
- **`athena-results/`** holds Athena's query outputs. A lifecycle rule deletes
  them after 7 days so they never pile up.

The key builders live in one place, `lakehouse/config.py` (`gridmet_key`,
`mtbs_key`, `landfire_key`, `manifest_key`, `integrated_*_key`). No code builds a
key by hand.

---

## 5. Each dataset, end to end

### 5.1 gridMET (`ingest/gridmet.py`)

**Source:** `https://www.northwestknowledge.net/metdata/data/{var}_{year}.nc`, one
CONUS-wide NetCDF per variable per year, daily, 1/24° (about 4 km).

**Variables (13):** `pr tmmx tmmn rmax rmin vs th srad vpd erc bi fm100 fm1000`.
These are the weather and NFDRS fire-danger drivers of fire proneness: heat,
dryness, wind, fuel moisture, energy release and burning index.

**Years:** study years **plus the year before** (2001-2025 for a 2002 start).
This lets a fire in early January get a 30-day pre-ignition weather window.

**Steps for each (variable, year):**

1. If `bronze/gridmet/variable=.../year=.../..._west.nc` already exists, skip it.
   This is what makes re-runs cheap and safe.
2. Stream the file to a temp folder (`stream_download`: writes to `.part` and
   renames only on success, so a half-download never looks finished).
3. Open it with `mask_and_scale=False`, which keeps the packed integers untouched.
4. Clip to the West bbox (latitude runs north to south in gridMET, so the slice
   is reversed).
5. Write NetCDF4 with zlib level 4 compression and upload it.
6. Record a manifest line: source URL, S3 key, sha256, byte size, time, bbox,
   params, git commit.

If one file fails (404, timeout), the failure is recorded and the loop **keeps
going**. At the end the CLI prints every failure and exits with code 1, so
CodeBuild marks the build failed. Re-running fetches only what is missing.

**Integration (`ingest/integrate.py`):**

- Each year is opened *decoded* (physical units), renamed to the short code
  (`air_temperature` becomes `tmmx`), then written with the **source's own packing**
  (`uint16`, same `scale_factor`/`add_offset`/`_FillValue`). The stored integers
  are therefore identical to the source. A test proves this byte for byte
  (`test_integrated_values_identical_to_raw`).
- Re-runs compare the years already in the Zarr with the years wanted:
  - **same years**: nothing happens;
  - **later years added** (e.g. 2026): those years are *appended* on `day`;
  - **earlier years added** (e.g. start moves to 1995): the store is *rebuilt*,
    because Zarr can only append at the end. Days are never duplicated.
- **Crash safety.** After each year is fully written, the job updates a small
  marker next to the store (`{var}.zarr.complete.json`) listing the finished years.
  On the next run the store is checked against it: the `day` axis must be unique,
  increasing, the same length as every array, and cover exactly the marked years.
  A write cut off by a timeout, an out-of-memory kill or an S3 error fails that
  check, and the store is rebuilt from bronze instead of being trusted with
  half-written (NaN) chunks.
- Written as Zarr format 2 with consolidated metadata, so opening it is one small
  read instead of one request per array.

### 5.2 MTBS (`ingest/mtbs.py`)

**Source:** the USGS composite perimeter shapefile zip
(`mtbs_perimeter_data.zip`), all mapped fires in the US since 1984.

**Steps:**

1. Stream the zip and extract it.
2. Read the shapefile with GeoPandas.
3. Keep rows whose ignition year (`ig_date`) is in the study years **and** whose
   perimeter intersects the West bbox. The intersection is tested on a temporary
   EPSG:4326 copy. The rows saved keep the original EPSG:4269 geometry.
4. Write GeoParquet to `bronze/mtbs/snapshot=<today>/mtbs_perimeters_west.parquet`
   and the metadata XML next to it as `_mtbs_perims_DD_metadata.xml`.
5. Manifest lines for both files.

Prescribed fires stay in bronze. Silver decides whether to drop them or keep them
as a separate class. Treating a prescribed burn as wildfire proneness would bias
the model toward places land managers chose to burn.

**Integration:** union every snapshot, sort by snapshot, drop duplicate
`event_id`s keeping the last (newest), and write
`bronze/integrated/mtbs/mtbs_perimeters_west.parquet`.

### 5.3 LANDFIRE (`ingest/landfire.py`)

**Source:** the LANDFIRE Product Service (LFPS) at `https://lfps.usgs.gov/api`.
You do not download LANDFIRE files directly. You submit a job ("these layers,
this area, this resolution") and LFPS builds a zip, which you poll for and then
download. LFPS requires an email address on each job. That email is a deploy
parameter (`LandfireEmail`, hidden in CloudFormation) and is never stored in the
repo.

**Layers (21, in 5 groups):**

| Group | Layers | Why it matters for proneness |
|---|---|---|
| `topo` | `LF2020_Elev`, `LF2020_SlpD`, `LF2020_Asp` | fire runs faster upslope; south-facing slopes dry out |
| `regime` | `LF2016_FRG`, `LF2016_FRI`, `LF2016_BPS` | historical fire frequency/severity, the strongest prior for proneness |
| `fuels_LF2016/2022/2024` | `FBFM40`, `EVT`, `EVC`, `EVH`, `CC` | what burns, and how much: fuel model, vegetation type/cover/height, canopy cover |

Fuels come in three versions (2016, 2022, 2024) because vegetation changes after
fires and treatments. `open_landfire_series("FBFM40")` stacks them on a `version`
dimension. LANDFIRE has nothing before 2016, so years 2002-2015 use LF2016 as an
approximation. State that in the report.

**Tiling:** one LFPS job for the whole West is too big. The bbox is cut into 5°
tiles (`make_tiles`), 5 columns x 4 rows = 20 tiles named `r{row}c{col}`, with
row 0 at the south. Each (group, tile) is one job, so 5 x 20 = 100 jobs.

**Steps for each (group, tile):**

1. If a `.tif` already exists under `bronze/landfire/group=.../tile=.../`, skip it, so a
   failed backfill resumes where it stopped.
2. Submit the job (layers joined with `;`, tile bbox, 90 m, email,
   `Output_Projection=5070`). The projection must be requested explicitly: without
   it LFPS returns a custom Albers centered on each request, so all 20 tiles
   would have different projections and could not be mosaicked. The first live
   tile showed exactly that; the build was stopped and the tile deleted.
3. Poll every 15 s for up to 1 h. A job that fails or times out is recorded as a
   failure and the loop moves to the next tile.
4. Download the zip, upload the GeoTIFF(s) to the tile folder, and write a manifest line.

**Integration:** for each layer, find the band in every tile file, merge all tiles
with `rasterio.merge`, and write a COG to `bronze/integrated/landfire/{layer}.tif`.

LFPS snaps each tile to LANDFIRE's native 30 m grid, so tile origins differ by
multiples of 30 m rather than 90 m. The mosaic therefore puts every tile onto one
90 m grid with nearest-neighbor sampling: no values are averaged or invented, but
a cell can sit up to 30 m from its source position. That is negligible next to
gridMET's 4 km cells. The raw tiles in `bronze/landfire/` are untouched.

---

## 6. Lineage: the manifest

Every file the pipeline writes gets one JSON line in
`bronze/_manifest/{source}_{run_id}.jsonl`:

```json
{"source": "gridmet", "s3_key": "bronze/gridmet/variable=erc/year=2020/erc_2020_west.nc",
 "source_url": "https://www.northwestknowledge.net/metdata/data/erc_2020.nc",
 "sha256": "…", "bytes": 31457280, "downloaded_at": "2026-11-22T03:14:00Z",
 "bbox": [-125.0, 31.0, -102.0, 49.5], "params": "{…}", "git_sha": "aa62151"}
```

It answers, for any file: where did it come from, when, which code wrote it, and
whether it has changed (sha256). In Athena:

```sql
SELECT * FROM bronze_inventory;                       -- files and GB per source
SELECT s3_key, downloaded_at FROM bronze_manifest
WHERE source = 'landfire' ORDER BY downloaded_at DESC;
```

Integration runs write manifest lines too (`source` = `integrated_gridmet`, `integrated_mtbs`,
`integrated_landfire`),
so even the combined objects have lineage.

---

## 7. Catalog and SQL (`ingest/catalog.py`, `sql/`)

`register_catalog()` runs at the end of every ingest:

1. Runs every `sql/*.sql` file in name order through Athena (numbered, so tables
   come before the views built on them), with `${bucket}`
   filled in:
   - `01_bronze_manifest.sql` creates the `bronze_manifest` table (JSON SerDe over
     the manifest folder).
   - `02_bronze_inventory_view.sql` creates the `bronze_inventory` view.
2. Registers the MTBS Parquet folders as Glue tables from their own schema:
   `bronze_mtbs_perimeters` (partitioned by `snapshot`) and `bronze_mtbs_integrated`.

Every statement is `IF NOT EXISTS` or `OR REPLACE`, so running it again is harmless.

gridMET (Zarr/NetCDF) and LANDFIRE (GeoTIFF) are not SQL tables. Arrays are read
with xarray, which is the right tool for gridded data. Their inventory is still in
SQL through the manifest.

**The workgroup** `fire-lakehouse` enforces Athena engine v3, writes results to
`athena-results/`, and **cancels any query that would scan more than 10 GB**
(about $0.05). A runaway `SELECT *` cannot cost real money.

---

## 8. The ingest CLI and CodeBuild

`python -m ingest <target> [--only X] [--force]`:

| Target | Does |
|---|---|
| `gridmet` | download missing gridMET files (`--only erc` for one variable) |
| `mtbs` | today's MTBS snapshot |
| `landfire` | missing LANDFIRE tiles (`--only topo` for one group) |
| `integrate` | rebuild/append all integrated objects |
| `catalog` | Glue tables + Athena view |
| `viewer` | regenerate and upload `viewer/index.html` |
| `all` | all of the above, in that order |

A step that throws is caught and reported, and the remaining steps still run. The
summary lists every failure, and the exit code is 1 if anything failed.

**Why CodeBuild as compute:** it is the cheapest serverless way to run a
multi-hour Python job. Lambda stops at 15 minutes, and Fargate/Batch need VPCs,
task definitions and more IAM. A CodeBuild project is one resource. It runs a
container from a zip of our code, bills per minute, and stops when done.

How it runs:

1. `infra/deploy.sh` zips the committed code (`git archive HEAD lakehouse ingest sql
   requirements-ingest.txt`) and uploads it to `_code/ingest.zip`. **It ships what is
   committed, not your working copy.** Commit before deploying code changes.
2. `infra/run_ingest.sh <profile> <target> [--medium]` starts a build with
   `INGEST_TARGET` set and tails the logs live.
3. Inside the build, Python 3.12 runs `pip install -r requirements-ingest.txt`, then
   `python -m ingest $INGEST_TARGET`.
4. Bucket, study years and the LANDFIRE email come in as environment variables
   set by the stack.
5. The project allows 8 hours per build, but this AWS account enforces a 45-minute
   cap on every build (an account-level restriction; the override flag is ignored).
   Only one build runs at a time. Because
   every step skips work already done, a timed-out build is simply re-run. The
   lineage manifest is rewritten after every file, so a killed build keeps the
   lineage of everything it finished. The deploy script also writes the commit SHA
   into the bundle, so every lineage row records the exact code version.

`--medium` (BUILD_GENERAL1_MEDIUM, 7 GB RAM) is the safe choice for MTBS and
integration. The first MTBS smoke test ran out of memory on the small instance
(3 GB) reading the US-wide shapefile. MTBS is now clipped to the West while the
file is read, but medium costs only $0.60/hour.

---

## 9. Configuration: one source of truth

The **study years are CloudFormation parameters** (`StudyStartYear`, default 2002,
min 1984 because MTBS starts there; `StudyEndYear`, default 2025). They show up in
three places, all read from the stack:

- CodeBuild environment variables (what ingest downloads);
- stack outputs (what readers think the lake contains);
- `lakehouse.config.study_years()`.

`lakehouse/config.py` resolves every setting in this order:

1. **environment variable** (`FIRE_LAKEHOUSE_BUCKET`, `STUDY_START_YEAR`, ...): used
   inside CodeBuild and in tests;
2. **CloudFormation stack outputs**: used on teammates' laptops, read once per
   session with your AWS profile;
3. **defaults** (2002-2025).

So teammates configure nothing beyond `AWS_PROFILE`. The bucket name and years
come from the stack.

---

## 10. Security and access

**One shared role, `FireLakehouseRole`, does everything**, as requested. It has
full access to *this* bucket, *this* Glue database and *this* Athena workgroup, plus:

- starting and reading builds of the one CodeBuild project;
- writing and reading its logs;
- reading its own stack's outputs.

It has nothing outside the lake. It cannot touch other buckets, create users or
change IAM.

Who can use the role:

- **CodeBuild** (the ingest job runs as this role);
- **IAM users in the group `fire-lakehouse-team`**. That group's only permission is
  `sts:AssumeRole` on this role.

A teammate's own key therefore can do nothing until it assumes the role. Their
profile `fire-lakehouse` does that automatically (`role_arn` + `source_profile` in
`~/.aws/config`), and sessions last up to 12 hours.

`infra/add_teammate.sh fire-admin <username>` creates the user, adds them to the
group, makes an access key, and prints the exact config blocks to send them.

Bucket protection:

- All public access is blocked.
- Server-side encryption (SSE-S3) is on by default.
- There is no versioning. Versioning would keep every overwritten integrated file
  and cost more, and bronze can always be re-downloaded from the sources.

Secrets rules you follow:

- Keys live only in `~/.aws/credentials` via named profiles. They never go in the
  repo, `.env`, notebooks, Slack or the report.
- Teammate keys go through a private channel (password-manager share, Signal).
- The admin profile is `fire-admin`. Your default CLI identity is a different
  account, so always pass `--profile fire-admin` (the scripts require it).
- Your email and the account ID are deploy-time parameters and are never written
  to the repo.
- On Dec 3, after teardown, delete the admin access key in the IAM console.

---

## 11. Reading data (what teammates actually do)

`lakehouse/read.py`:

- **`open_gridmet(vars, years, bbox)`** returns a lazy xarray Dataset in physical
  units.
  - It uses the integrated Zarr when it covers the requested years, otherwise the
    yearly raw files.
  - Unknown variables or years outside the lake raise `ValueError` listing what is
    available.
- **`read_mtbs(snapshot=None)`** returns the integrated GeoDataFrame. Pass a
  snapshot date for a historical version.
- **`open_landfire(layer, bbox)`** and **`open_landfire_series(acronym, bbox)`**
  return rioxarray DataArrays, clipped by a lon/lat bbox.
- **`query(sql)`** runs Athena SQL and returns a pandas DataFrame.
- **`read_silver(name)`** and **`open_zarr(name)`** read silver.

**Local cache:** every downloaded object is stored under
`~/.cache/fire-lakehouse/<bucket>/<key>`, so a second run reads from disk.
Integrated files that can be rebuilt are revalidated against their S3 ETag. If the
lake copy changed, the cache refreshes automatically.

**Friendly auth errors:** missing, expired or wrong-account credentials raise
`LakehouseAuthError` with "set AWS_PROFILE=fire-lakehouse (see docs/lakehouse.md)"
instead of a long botocore traceback.

---

## 12. The viewer (`lakehouse/viewer.py`)

`viewer/index.html` is a single self-contained HTML page with no external
scripts. Dark mode follows the OS, and it works at phone width. It is built from
S3 listings and the manifest, with no Athena cost.

- Summary cards: files, GB, sources, last run.
- **gridMET matrix**: variable x year. Each cell is blue (raw only), orange
  (integrated) or empty (missing).
- **MTBS**: stacked bars of fires per year by incident type.
- **LANDFIRE**: a map of the 20 tiles, a group x tile matrix, and the integrated
  layer list.
- **Silver tables** present.
- **Lineage table** of manifest rows, with a filter box.
- A copy-paste "how to load" snippet.

`python -m lakehouse.viewer` downloads and opens it. The ingest job republishes it
at the end of every run.

---

## 13. Testing

86 tests (`pytest`), all offline:

- **moto** runs a fake S3/Glue/CloudFormation server locally. Code is pointed at it
  with `FIRE_LAKEHOUSE_ENDPOINT`, so tests exercise real boto3/s3fs calls without AWS.
- **requests-mock** fakes gridMET, MTBS and LFPS HTTP responses, including 404s,
  timeouts and failed LFPS jobs.
- **Synthetic data fixtures**: a tiny packed gridMET cube, MTBS perimeters in and
  out of the West and in and out of the years, and LANDFIRE GeoTIFF tiles.
- **cfn-lint** validates the CloudFormation template. It was also validated by AWS
  in us-east-1.

Key behaviors pinned by tests:

- re-runs skip finished files and tiles;
- one failed download does not stop the run, but does fail the exit code;
- a dead download leaves no object behind;
- integrated gridMET integers equal the source;
- appending vs rebuilding never duplicates days;
- MTBS dedup keeps the newest snapshot;
- readers give clear errors for bad requests and bad credentials;
- the CLI runs steps in order and reports failures.

Run them:

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -q
```

---

## 14. Cost

Estimates once the lake is full:

| Item | Estimate |
|---|---|
| S3 storage, about 30-45 GB (raw + integrated) | about $0.70-1.00 / month |
| Initial backfill on CodeBuild (roughly 6-12 build hours across several builds) | about $3-6 one time |
| Athena | cents (manifest and MTBS tables are MB-sized; 10 GB cap per query) |
| S3 requests; downloads to laptops (first 100 GB/month out of AWS are free, then $0.09/GB) | cents |
| Glue catalog, CloudWatch Logs (7 days), Budget | free tier / negligible |

**Expected total through Dec 3: about $5.** Hard limits keep it from running away:

- **Per build:** at most 8 hours (worst case $4.80 on medium).
- **Concurrency:** one build at a time (`ConcurrentBuildLimit: 1`).
- **Per query:** Athena cancels anything scanning over 10 GB ($0.05).
- **Budget alerts:** email at 80% of $5 actual and at 100% forecast.
- **Budget stop:** at 100% of $5 actual, a budget action attaches the deny policy
  `fire-lakehouse-stop-spend` to the shared role, the team group and the admin
  user. It blocks `codebuild:StartBuild` and `athena:StartQueryExecution`. AWS
  billing data lags by several hours, so this is a backstop; the per-build timeout
  is the fast cap. To resume, reset the action in AWS Budgets >
  `fire-lakehouse-monthly` > Actions.

---

## 15. Repo map

```
infra/
  lakehouse.yaml       the whole stack
  deploy.sh            lint + deploy + upload code bundle
  run_ingest.sh        start a CodeBuild ingest and tail logs
  add_teammate.sh      create a teammate IAM user + print their config
  teardown.sh          empty the bucket, delete users and the stack (asks you to type the bucket name)
lakehouse/             teammate-facing package (never imports ingest)
  config.py            constants, settings resolution, S3 key builders, tiles
  s3.py                exists/list/upload, cached downloads, streaming source downloads
  manifest.py          lineage records
  read.py              open_gridmet, read_mtbs, open_landfire(_series), query, read_silver, open_zarr
  write.py             write_silver, write_zarr
  viewer.py            builds and publishes viewer/index.html
  errors.py            LakehouseAuthError
ingest/                runs in CodeBuild
  gridmet.py, mtbs.py, landfire.py   collection into bronze
  integrate.py         bronze/integrated
  catalog.py           Glue tables + Athena view
  __main__.py          CLI
sql/                   Athena DDL run by catalog.py
tests/                 86 offline tests
docs/lakehouse.md      teammate guide
docs/lakehouse-deep-dive.md   this file
```

---

## 16. Operating it: the full runbook

These run in order and need the admin profile.

1. **Deploy** (creates the bucket, role, group, Glue database, workgroup,
   CodeBuild project and budget):

   ```bash
   infra/deploy.sh fire-admin <your-budget-email> <your-landfire-email>
   ```

   Confirm the budget subscription email AWS sends you.

2. **Smoke test** one small piece first:

   ```bash
   infra/run_ingest.sh fire-admin mtbs
   ```

3. **Backfill** one source per build, so each fits in the 8-hour limit. If a build
   times out, run the same command again; it resumes where it stopped.

   ```bash
   infra/run_ingest.sh fire-admin gridmet --medium
   infra/run_ingest.sh fire-admin landfire --medium
   ```

4. **Integrate, catalog, viewer:**

   ```bash
   infra/run_ingest.sh fire-admin integrate --medium
   infra/run_ingest.sh fire-admin catalog
   infra/run_ingest.sh fire-admin viewer
   ```

5. **Check**: open the viewer and run `query("SELECT * FROM bronze_inventory")`.
   You can also see the bucket in the S3 console (us-east-1) under
   `fire-lakehouse-<account-id>`.

6. **Add teammates**, one per person:

   ```bash
   infra/add_teammate.sh fire-admin <name>
   ```

7. **Delete local `data/`**, only after step 5 confirms the lake has everything.
8. **Dec 3: tear down** (irreversible, deletes all lake data):

   ```bash
   infra/teardown.sh fire-admin
   ```

   Then delete the `fire-admin` access key in IAM.

---

## 17. Status and what's left

Done and tested:

- infrastructure template and scripts;
- all three ingesters;
- integration;
- catalog;
- CLI;
- readers and writers;
- viewer;
- docs.

Not done yet:

- **Deploy + backfill (step 16.1-16.5).** Nothing exists in AWS yet. This needs your
  go-ahead and your two email addresses because it creates billable resources and
  IAM identities.
- **The live LFPS check.** Layer names were confirmed against the LFPS product list,
  but the first real job runs during the backfill. If LFPS rejects a layer name,
  the run reports it per tile and the fix is a one-line change in `config.py`.
- **Porting old consumers.** `classifier/WildfireClassifier.py`, `scripts/make-visuals.py`
  and the notebooks still read `data/raw/`. Their error messages now point at the
  lake, but they need to switch to `open_gridmet` / `read_mtbs` / `open_landfire`
  before local data is deleted.
- **Silver.** The cleaning teammate's scripts, using `write_silver` / `write_zarr`.
- **Teardown on 2026-12-03.**
