"""MTBS: download the composite perimeter shapefile, keep western-US fires in the study
years, and upload one GeoParquet snapshot to bronze.

Columns, values and the source CRS (EPSG:4269) are unchanged. All incident types,
including prescribed burns, are kept; the cleaning stage decides what to drop.
"""
from __future__ import annotations

import json
import logging
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
from botocore.exceptions import ClientError
from shapely.geometry import box

from ingest import IngestResult
from lakehouse import config, s3
from lakehouse.manifest import ManifestRecord, git_sha, new_run_id, utc_now, write_manifest
from lakehouse.s3 import stream_download

log = logging.getLogger(__name__)

MTBS_URL = (
    "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/MTBS_Fire/data/"
    "composite_data/burned_area_extent_shapefile/mtbs_perimeter_data.zip"
)
PARQUET_NAME = "mtbs_perimeters_west.parquet"
METADATA_NAME = "mtbs_perims_DD_metadata.xml"
# Leading underscore: Athena/Hive skip it when reading the Parquet table in this folder.
METADATA_KEY_NAME = "_" + METADATA_NAME


def filter_mtbs(gdf: gpd.GeoDataFrame, years, bbox=config.WEST_BBOX) -> gpd.GeoDataFrame:
    """Rows whose ignition year is in `years` and whose perimeter intersects `bbox` (lon/lat)."""
    year = pd.to_datetime(gdf["ig_date"], errors="coerce").dt.year
    in_years = year.isin(list(years))
    in_bbox = gdf.geometry.to_crs("EPSG:4326").intersects(box(*bbox))
    return gdf[in_years & in_bbox]


def _years_tag(years: range) -> str:
    return f"{years.start}-{years.stop - 1}"


def _snapshot_covers(key: str, years: range) -> bool:
    """True if `key` exists and was cut for (at least) these study years."""
    try:
        head = config.client("s3").head_object(Bucket=config.bucket(), Key=key)
    except ClientError as err:
        if err.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise
    tag = head.get("Metadata", {}).get("study-years", "")
    try:
        start, end = (int(x) for x in tag.split("-"))
    except ValueError:
        return False
    return start <= years.start and end >= years.stop - 1


def ingest_mtbs(snapshot: str | None = None, force: bool = False) -> IngestResult:
    snapshot = snapshot or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    pq_key = config.mtbs_key(snapshot, PARQUET_NAME)
    result = IngestResult()
    years = config.study_years()
    if not force and _snapshot_covers(pq_key, years):
        result.skipped.append(pq_key)
        return result

    sha = git_sha()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            zpath = stream_download(MTBS_URL, tmp / "mtbs_perimeter_data.zip", timeout=600)
            with zipfile.ZipFile(zpath) as z:
                z.extractall(tmp / "src")
            shp = next((tmp / "src").rglob("*.shp"))
            # bbox prefilter at read time keeps memory low (the US-wide file does not fit in 3 GB);
            # filter_mtbs still does the exact intersection and year test.
            west = filter_mtbs(gpd.read_file(shp, bbox=config.WEST_BBOX), years)
            out = tmp / PARQUET_NAME
            west.to_parquet(out)
            uploads = [(out, pq_key, "application/vnd.apache.parquet", {"study-years": _years_tag(years)})]
            meta = next((tmp / "src").rglob(METADATA_NAME), None)
            if meta is not None:
                uploads.append((meta, config.mtbs_key(snapshot, METADATA_KEY_NAME), "application/xml", None))
            params = json.dumps(
                {"snapshot": snapshot, "years": [years.start, years.stop - 1], "fires": len(west)}
            )
            for path, key, ctype, metadata in uploads:
                s3.upload_file(path, key, content_type=ctype, metadata=metadata)
                result.records.append(
                    ManifestRecord(
                        source="mtbs", s3_key=key, source_url=MTBS_URL, sha256=s3.sha256_file(path),
                        bytes=path.stat().st_size, downloaded_at=utc_now(),
                        bbox=list(config.WEST_BBOX), params=params, git_sha=sha,
                    )
                )
            log.info("mtbs snapshot %s: %d fires", snapshot, len(west))
    except Exception as err:
        log.warning("mtbs failed: %s", err)
        result.failed.append((MTBS_URL, f"{type(err).__name__}: {err}"))
    write_manifest(result.records, "mtbs", new_run_id())
    return result
