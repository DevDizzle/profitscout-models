import os, requests, datetime as dt, pandas as pd, logging
FMP = "https://financialmodelingprep.com/api/v3"
_API = os.getenv("FMP_API_KEY")

def _get(url):
    r = requests.get(f"{url}&apikey={_API}" if "apikey=" not in url else url, timeout=30)
    r.raise_for_status()
    return r.json()

def get_prices(ticker: str, anchor: str) -> pd.DataFrame:
    """30 d day-close window ending 1 d after earnings-call date."""
    start = (dt.date.fromisoformat(anchor) - dt.timedelta(days=5)).isoformat()
    url = f"{FMP}/historical-price-full/{ticker}?from={start}&serietype=line"
    data = _get(url)["historical"][:40]
    return pd.DataFrame(data).assign(date=lambda d: pd.to_datetime(d["date"]))

def get_eps_surprise(ticker: str, anchor: str) -> float | None:
    url = f"{FMP}/earnings-surprises/{ticker}"
    for rec in _get(url):
        if rec["date"] == anchor:
            return rec["epsSurprisePercent"]
    return None
