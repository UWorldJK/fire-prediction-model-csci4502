"""LANDFIRE: request layer groups per tile from the LANDFIRE Product Service (LFPS) and
upload the returned files to bronze unchanged (native CRS EPSG:5070, multi-band GeoTIFF
whose band order is the group's layer order).

Each (group, tile) is one LFPS job. Finished tiles are skipped, so a failed or
timed-out run is resumed by running it again.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import zipfile
from pathlib import Path

import requests

from ingest import IngestResult
from lakehouse import config, s3
from lakehouse.config import make_tiles  # noqa: F401  (re-exported; tiling lives in config)
from lakehouse.manifest import ManifestRecord, git_sha, new_run_id, utc_now, write_manifest
from lakehouse.s3 import stream_download

log = logging.getLogger(__name__)
LFPS_URL = "https://lfps.usgs.gov/api/job"


def job_payload(layers, tile_bbox, resolution_m: int, email: str) -> dict:
    west, south, east, north = tile_bbox
    # No Output_Projection: LFPS then returns LANDFIRE's native Albers (EPSG:5070).
    return {
        "Layer_List": ";".join(layers),
        "Area_of_Interest": f"{west} {south} {east} {north}",
        "Resample_Resolution": resolution_m,
        "Email": email,
    }


def run_lfps_job(payload: dict, poll_seconds: int = 15, max_wait_seconds: int = 3600) -> tuple[str, str]:
    """Submit a job and wait for it. Returns (output zip URL, job id)."""
    resp = requests.post(f"{LFPS_URL}/submit", json=payload, timeout=60)
    resp.raise_for_status()
    job_id = resp.json()["jobId"]
    waited = 0
    while True:
        status = requests.get(f"{LFPS_URL}/status", params={"JobId": job_id}, timeout=60).json()
        if status.get("outputFile"):
            return status["outputFile"], job_id
        if status.get("status") in ("Failed", "Canceled"):
            raise RuntimeError(f"LFPS job {job_id} {status['status']}: {status.get('messages')}")
        if waited > max_wait_seconds:
            raise TimeoutError(f"LFPS job {job_id} still {status.get('status')} after {waited}s")
        time.sleep(poll_seconds)
        waited += max(poll_seconds, 1)


def _tile_done(group: str, tile_id: str) -> bool:
    return any(k.endswith(".tif") for k in s3.list_keys(config.landfire_key(group, tile_id, "")))


def ingest_landfire(groups=None, tile_deg=None, resolution_m=None, force: bool = False) -> IngestResult:
    email = os.environ.get("LANDFIRE_EMAIL")
    if not email:
        raise RuntimeError("Set LANDFIRE_EMAIL (LFPS requires an email address on every job).")
    groups = groups or config.LANDFIRE_GROUPS
    tile_deg = tile_deg or config.LANDFIRE_TILE_DEG
    resolution_m = resolution_m or config.LANDFIRE_RESOLUTION_M
    result = IngestResult()
    sha = git_sha()

    for group, layers in groups.items():
        for tile_id, tile_bbox in make_tiles(config.WEST_BBOX, tile_deg):
            prefix = config.landfire_key(group, tile_id, "")
            if not force and _tile_done(group, tile_id):
                result.skipped.append(prefix)
                continue
            try:
                url, job_id = run_lfps_job(job_payload(layers, tile_bbox, resolution_m, email))
                params = json.dumps(
                    {"group": group, "tile_id": tile_id, "tile_bbox": list(tile_bbox), "layers": layers,
                     "resolution_m": resolution_m, "lfps_job_id": job_id}
                )
                with tempfile.TemporaryDirectory() as tmp:
                    tmp = Path(tmp)
                    zpath = stream_download(url, tmp / "lfps.zip", timeout=600)
                    with zipfile.ZipFile(zpath) as z:
                        z.extractall(tmp / "out")
                    for path in sorted(p for p in (tmp / "out").rglob("*") if p.is_file()):
                        key = config.landfire_key(group, tile_id, path.name)
                        s3.upload_file(path, key)
                        result.records.append(
                            ManifestRecord(
                                source="landfire", s3_key=key, source_url=url, sha256=s3.sha256_file(path),
                                bytes=path.stat().st_size, downloaded_at=utc_now(),
                                bbox=list(tile_bbox), params=params, git_sha=sha,
                            )
                        )
                log.info("landfire %s %s uploaded", group, tile_id)
            except Exception as err:
                log.warning("landfire %s %s failed: %s", group, tile_id, err)
                result.failed.append((prefix, f"{type(err).__name__}: {err}"))
    write_manifest(result.records, "landfire", new_run_id())
    return result
