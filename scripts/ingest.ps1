# Run ingestion (init DB if needed) and fetch latest articles

$env:NEWS_INSIGHTS_LOG_LEVEL='DEBUG'
$env:NEWS_INSIGHTS_REQUEST_TIMEOUT_S='10'
#poetry run news-insights init-db
poetry run news-insights ingest --limit 50 --time-budget-s 120

