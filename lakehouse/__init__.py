"""Western US fire lakehouse: read bronze data and write silver data on S3.

    export AWS_PROFILE=fire-lakehouse
    from lakehouse import open_gridmet, read_mtbs, open_landfire, query
"""
from lakehouse.read import (  # noqa: F401
    open_gridmet,
    open_landfire,
    open_landfire_series,
    open_zarr,
    query,
    read_mtbs,
    read_silver,
)
from lakehouse.write import write_silver, write_zarr  # noqa: F401,E402
