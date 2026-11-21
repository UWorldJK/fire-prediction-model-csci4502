"""Register bronze in the Glue Data Catalog so Athena can query it.

- sql/*.sql run in name order (the lineage manifest table and the inventory view).
  ${bucket} in each file is replaced with the lake bucket.
- MTBS Parquet folders are registered from their own Parquet schema.

Safe to re-run: every statement is IF NOT EXISTS / OR REPLACE, and Parquet tables
are re-registered in overwrite mode.
"""
from __future__ import annotations

import logging
from pathlib import Path
from string import Template

import awswrangler as wr
import boto3

from lakehouse import config, s3

log = logging.getLogger(__name__)
SQL_DIR = Path(__file__).resolve().parent.parent / "sql"
PARQUET_TABLES = {
    "bronze_mtbs_perimeters": "bronze/mtbs/",
    "bronze_mtbs_integrated": "bronze/integrated/mtbs/",
}


def _session() -> boto3.Session:
    return boto3.Session(region_name=config.REGION)


def register_catalog() -> None:
    bucket = config.bucket()
    session = _session()
    for path in sorted(SQL_DIR.glob("*.sql")):
        sql = Template(path.read_text()).substitute(bucket=bucket)
        wr.athena.start_query_execution(
            sql, database=config.DATABASE, workgroup=config.WORKGROUP, wait=True, boto3_session=session
        )
        log.info("ran %s", path.name)
    for table, prefix in PARQUET_TABLES.items():
        if not any(k.endswith(".parquet") for k in s3.list_keys(prefix)):
            log.info("skip %s: no parquet under %s", table, prefix)
            continue
        wr.s3.store_parquet_metadata(
            path=f"s3://{bucket}/{prefix}", database=config.DATABASE, table=table,
            dataset=True, mode="overwrite", path_suffix=".parquet", boto3_session=session,
        )
        log.info("registered %s", table)
