from ingest import IngestResult
from ingest.__main__ import main


def _ok(*args, **kwargs):
    return IngestResult()


def test_cli_all_runs_in_order(mocker):
    calls = []
    for name in ("ingest_gridmet", "ingest_mtbs", "ingest_landfire", "integrate_all", "register_catalog", "publish_viewer"):
        mocker.patch(f"ingest.__main__.{name}", side_effect=lambda *a, n=name, **k: calls.append(n))
    assert main(["all"]) == 0
    assert calls == ["ingest_gridmet", "ingest_mtbs", "ingest_landfire", "integrate_all", "register_catalog", "publish_viewer"]


def test_cli_only_gridmet_variable(mocker):
    g = mocker.patch("ingest.__main__.ingest_gridmet", side_effect=_ok)
    main(["gridmet", "--only", "erc"])
    assert g.call_args.kwargs["variables"] == ["erc"]


def test_cli_only_landfire_group(mocker):
    lf = mocker.patch("ingest.__main__.ingest_landfire", side_effect=_ok)
    main(["landfire", "--only", "topo"])
    assert list(lf.call_args.kwargs["groups"]) == ["topo"]


def test_cli_nonzero_on_failures(mocker):
    mocker.patch("ingest.__main__.ingest_mtbs", return_value=IngestResult([], [], [("k", "boom")]))
    assert main(["mtbs"]) == 1


def test_cli_step_exception_is_reported_not_raised(mocker):
    mocker.patch("ingest.__main__.register_catalog", side_effect=RuntimeError("athena down"))
    assert main(["catalog"]) == 1


def test_register_catalog_runs_sql_and_registers_parquet(mocker, monkeypatch):
    from ingest import catalog

    monkeypatch.setenv("FIRE_LAKEHOUSE_BUCKET", "b")
    q = mocker.patch("ingest.catalog.wr.athena.start_query_execution")
    meta = mocker.patch("ingest.catalog.wr.s3.store_parquet_metadata")
    mocker.patch("ingest.catalog.s3.list_keys", return_value=["x.parquet"])
    catalog.register_catalog()
    sqls = [c.args[0] for c in q.call_args_list]
    assert any("CREATE EXTERNAL TABLE IF NOT EXISTS bronze_manifest" in s and "s3://b/bronze/_manifest/" in s for s in sqls)
    assert any("CREATE OR REPLACE VIEW bronze_inventory" in s for s in sqls)
    assert all(c.kwargs["workgroup"] == "fire-lakehouse" for c in q.call_args_list)
    tables = {c.kwargs["table"]: c.kwargs["path"] for c in meta.call_args_list}
    assert tables == {
        "bronze_mtbs_perimeters": "s3://b/bronze/mtbs/",
        "bronze_mtbs_integrated": "s3://b/bronze/integrated/mtbs/",
    }


def test_catalog_creates_manifest_table_before_inventory_view(mocker, monkeypatch):
    from ingest import catalog

    monkeypatch.setenv("FIRE_LAKEHOUSE_BUCKET", "b")
    q = mocker.patch("ingest.catalog.wr.athena.start_query_execution")
    mocker.patch("ingest.catalog.wr.s3.store_parquet_metadata")
    mocker.patch("ingest.catalog.s3.list_keys", return_value=[])
    catalog.register_catalog()
    sqls = [c.args[0] for c in q.call_args_list]
    table = next(i for i, s in enumerate(sqls) if "CREATE EXTERNAL TABLE IF NOT EXISTS bronze_manifest" in s)
    view = next(i for i, s in enumerate(sqls) if "CREATE OR REPLACE VIEW bronze_inventory" in s)
    assert table < view


def test_cli_integrate_force_is_passed_through(mocker):
    m = mocker.patch("ingest.__main__.integrate_all", side_effect=_ok)
    main(["integrate", "--force"])
    assert m.call_args.kwargs["force"] is True
