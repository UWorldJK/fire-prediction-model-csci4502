"""Small S3 helpers: existence checks, listing, cached downloads, streaming source downloads."""
from __future__ import annotations

import hashlib
from pathlib import Path

import requests
from botocore.exceptions import ClientError

from lakehouse import config


def object_exists(key: str) -> bool:
    try:
        config.client("s3").head_object(Bucket=config.bucket(), Key=key)
        return True
    except ClientError as err:
        if err.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def list_keys(prefix: str) -> list[str]:
    paginator = config.client("s3").get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=config.bucket(), Prefix=prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))
    return sorted(keys)


def upload_file(path: Path, key: str, content_type: str | None = None) -> None:
    extra = {"ContentType": content_type} if content_type else None
    config.client("s3").upload_file(str(path), config.bucket(), key, ExtraArgs=extra)


def _download_raw(key: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    config.client("s3").download_file(config.bucket(), key, str(part))
    part.replace(dest)


def download_file(key: str, refresh: bool = False, revalidate: bool = False) -> Path:
    """Download an object to the local cache (once) and return its path.

    refresh: always re-download. revalidate: re-download only if the object's ETag
    changed since the cached copy (for objects that get rebuilt, e.g. integrated files).
    """
    dest = config.cache_dir() / config.bucket() / key
    etag_file = dest.with_name(dest.name + ".etag")
    etag = None
    if revalidate:
        etag = config.client("s3").head_object(Bucket=config.bucket(), Key=key)["ETag"]
        if dest.exists() and (not etag_file.exists() or etag_file.read_text() != etag):
            refresh = True
    if refresh or not dest.exists():
        _download_raw(key, dest)
        if etag:
            etag_file.write_text(etag)
    return dest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stream_download(url: str, dest: Path, timeout: int = 120) -> Path:
    """Stream url to dest via dest.part; nothing is left behind if it fails."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        with requests.get(url, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            with open(part, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        part.replace(dest)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return dest
