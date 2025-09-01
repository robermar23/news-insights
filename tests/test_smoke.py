from news_insights.ingestion.extract import canonicalize_url


def test_canonicalize_url_strips_tracking():
    url = "https://example.com/x?utm_source=foo&utm_campaign=bar&a=1#frag"
    canon = canonicalize_url(url)
    assert "utm_" not in canon
    assert canon.endswith("a=1")

