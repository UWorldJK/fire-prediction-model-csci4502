"""Lineage manifest: one JSON line per bronze object written, queryable in Athena."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from lakehouse import config, s3


@dataclass
class ManifestRecord:
    source: str
    s3_key: str
    source_url: str
    sha256: str
    bytes: int
    downloaded_at: str
    bbox: list[float]
    params: str
    git_sha: str


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_sha() -> str:
    if os.environ.get("GIT_SHA"):
        return os.environ["GIT_SHA"]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def write_manifest(records: list[ManifestRecord], source: str, run_id: str) -> str | None:
    if not records:
        return None
    key = config.manifest_key(source, run_id)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "manifest.jsonl"
        path.write_text("".join(json.dumps(asdict(r)) + "\n" for r in records))
        s3.upload_file(path, key, content_type="application/x-ndjson")
    return key


def read_manifests() -> list[ManifestRecord]:
    """Every manifest record in the lake, read straight from S3 (no Athena cost)."""
    body = config.client("s3")
    records: list[ManifestRecord] = []
    for key in s3.list_keys("bronze/_manifest/"):
        text = body.get_object(Bucket=config.bucket(), Key=key)["Body"].read().decode()
        records.extend(ManifestRecord(**json.loads(line)) for line in text.splitlines() if line)
    return records
