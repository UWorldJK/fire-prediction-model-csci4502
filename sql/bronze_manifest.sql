CREATE EXTERNAL TABLE IF NOT EXISTS bronze_manifest (
  source string,
  s3_key string,
  source_url string,
  sha256 string,
  bytes bigint,
  downloaded_at string,
  bbox array<double>,
  params string,
  git_sha string
)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
LOCATION 's3://${bucket}/bronze/_manifest/'
