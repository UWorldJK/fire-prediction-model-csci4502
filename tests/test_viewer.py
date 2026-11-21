from lakehouse.config import client
from lakehouse.viewer import publish_viewer, render_html


def test_matrix_marks_missing_and_integrated(viewer_data):
    html = render_html(viewer_data)
    assert 'data-cell="tmmx-2020" class="raw integrated"' in html
    assert 'data-cell="erc-2020" class="raw"' in html
    assert 'data-cell="erc-2019" class="missing"' in html
    assert 'data-cell="tmmx-2018" class="missing"' in html  # gridMET years start one year early


def test_sections_present(viewer_data):
    html = render_html(viewer_data)
    for heading in ("gridMET", "MTBS", "LANDFIRE", "Silver tables", "Lineage", "How to load"):
        assert heading in html
    assert "silver_cell_year_features" in html
    assert 'data-tile="topo-r0c0"' in html


def test_values_escaped(viewer_data):
    viewer_data.manifests[0].source_url = "<script>x</script>"
    html = render_html(viewer_data)
    assert "<script>x</script>" not in html
    assert "&lt;script&gt;x&lt;/script&gt;" in html


def test_self_contained(viewer_data):
    html = render_html(viewer_data)
    assert "<script src=" not in html
    assert 'rel="stylesheet"' not in html
    assert "AKIA" not in html and "aws_secret" not in html.lower()


def test_publish_uploads_html(lake_bucket, mocker, viewer_data):
    mocker.patch("lakehouse.viewer.collect_viewer_data", return_value=viewer_data)
    assert publish_viewer() == "viewer/index.html"
    head = client("s3").head_object(Bucket="test-lake", Key="viewer/index.html")
    assert head["ContentType"] == "text/html; charset=utf-8"


def test_collect_viewer_data_from_lake(lake_bucket, synthetic_gridmet, synthetic_mtbs, tmp_path):
    from lakehouse import s3
    from lakehouse.config import INTEGRATED_MTBS_KEY, gridmet_key, landfire_key
    from lakehouse.viewer import collect_viewer_data

    s3.upload_file(synthetic_gridmet, gridmet_key("tmmx", 2020))
    p = tmp_path / "m.parquet"
    synthetic_mtbs.to_parquet(p)
    s3.upload_file(p, INTEGRATED_MTBS_KEY)
    s3.upload_file(synthetic_gridmet, landfire_key("topo", "r0c0", "a.tif"))
    data = collect_viewer_data()
    assert ("tmmx", 2020) in data.gridmet_raw
    assert ("topo", "r0c0") in data.landfire_tiles
    assert int(data.mtbs_counts.loc[2019, "Wildfire"]) == 2
