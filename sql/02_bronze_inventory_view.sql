CREATE OR REPLACE VIEW bronze_inventory AS
SELECT source,
       count(*) AS files,
       round(sum(bytes) / 1e9, 2) AS gb,
       max(downloaded_at) AS last_download
FROM bronze_manifest
GROUP BY source
