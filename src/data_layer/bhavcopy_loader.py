from __future__ import annotations

"""Primary NSE bhavcopy ingestion and normalization utilities."""

import io
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from src.core.calendar import trading_days_between
from src.core.config import AppConfig
from src.core.logging import get_logger
from src.core.paths import RunPaths

log = get_logger(__name__)

# Persistent session with NSE-compatible headers (avoids HTTP 403)
_SESSION: requests.Session | None = None


def _get_nse_session() -> requests.Session:
    """Return a reusable requests.Session with NSE-friendly headers & cookies."""
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
        "Connection": "keep-alive",
    })
    # Hit the NSE homepage to get a valid session cookie
    try:
        s.get("https://www.nseindia.com/", timeout=15)
    except Exception:
        pass  # Proceed anyway; some archive URLs work without cookies
    _SESSION = s
    return s


@dataclass(frozen=True)
class BhavcopyURL:
    """Candidate URL for one trading day's bhavcopy archive."""
    url: str
    kind: str  # "udiff" | "historical"


def _candidate_urls(day: pd.Timestamp) -> list[BhavcopyURL]:
    """Return possible bhavcopy endpoints for one date (new and historical)."""
    ymd = day.strftime("%Y%m%d")
    dd = day.strftime("%d")
    mmm = day.strftime("%b").upper()
    yyyy = day.strftime("%Y")

    # Newer UDiFF-style
    u1 = f"https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{ymd}_F_0000.csv.zip"

    # Historical "EQUITIES/YYYY/MMM/cmDDMMMYYYYbhav.csv.zip"
    u2 = f"https://nsearchives.nseindia.com/content/historical/EQUITIES/{yyyy}/{mmm}/cm{dd}{mmm}{yyyy}bhav.csv.zip"
    return [BhavcopyURL(u1, "udiff"), BhavcopyURL(u2, "historical")]


def _http_get(url: str) -> bytes:
    """Perform HTTP GET using the NSE session with browser-like headers."""
    session = _get_nse_session()
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return r.content


def download_bhavcopy_range(cfg: AppConfig, paths: RunPaths) -> None:
    """Download bhavcopy zip files for configured date range with rate limiting."""
    days = trading_days_between(cfg.data.start_date, cfg.data.end_date)
    out_dir = paths.raw / "bhavcopy"
    out_dir.mkdir(parents=True, exist_ok=True)

    for d in tqdm(days, desc="bhavcopy"):
        ymd = d.strftime("%Y%m%d")
        out_zip = out_dir / f"bhav_{ymd}.zip"
        if out_zip.exists():
            continue

        ok = False
        for cand in _candidate_urls(d):
            try:
                content = _http_get(cand.url)
                out_zip.write_bytes(content)
                ok = True
                break
            except Exception:
                continue

        # Rate limiting: 0.5s between requests to avoid IP blocking
        time.sleep(0.5)

        if not ok:
            # not fatal: holiday/missing day
            continue


def _read_first_csv_from_zip(b: bytes) -> pd.DataFrame:
    """Read first CSV member from a ZIP byte payload."""
    zf = zipfile.ZipFile(io.BytesIO(b))
    names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
    if not names:
        raise ValueError("No CSV found in zip")
    with zf.open(names[0]) as f:
        return pd.read_csv(f)


def load_bhavcopy_ohlcv(cfg: AppConfig, paths: RunPaths) -> pd.DataFrame:
    """Load cached bhavcopy zips into canonical OHLCV rows."""
    in_dir = paths.raw / "bhavcopy"
    if not in_dir.exists():
        return pd.DataFrame()

    rows = []
    for p in sorted(in_dir.glob("bhav_*.zip")):
        ymd = p.stem.split("_")[1]
        day = pd.to_datetime(ymd, format="%Y%m%d")
        try:
            df = _read_first_csv_from_zip(p.read_bytes())
        except Exception:
            continue

        norm = _normalize_bhavcopy(df)
        norm["date"] = day
        rows.append(norm)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["ticker", "Open", "High", "Low", "Close"])
    out["Volume"] = out["Volume"].fillna(0.0).astype(float)
    if "series" in out.columns:
        out["series"] = out["series"].astype(str).str.upper()
        out = out[out["series"] == "EQ"].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out = out.sort_values(["ticker", "date", "Volume"], ascending=[True, True, False])
    out = out.drop_duplicates(subset=["ticker", "date"], keep="first")
    return out


def _normalize_bhavcopy(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize mixed bhavcopy schema variants to standard column names."""
    cols = {c.upper(): c for c in df.columns}
    def pick(*names: str) -> str | None:
        for n in names:
            if n in cols:
                return cols[n]
        return None

    sym = pick("SYMBOL", "TCKRSYMB")
    series = pick("SERIES", "SCTYSRS")
    op = pick("OPEN", "OPNPRIC")
    hi = pick("HIGH", "HGHPRIC")
    lo = pick("LOW", "LWPRIC")
    cl = pick("CLOSE", "CLSPRIC")
    vol = pick("TOTTRDQTY", "TTLTRADGVOL")
    val = pick("TOTTRDVAL", "TTLTRFVAL")

    if sym is None or op is None or hi is None or lo is None or cl is None:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["ticker"] = df[sym].astype(str).str.strip() + ".NS"
    out["series"] = df[series].astype(str).str.strip() if series else "EQ"
    out["Open"] = pd.to_numeric(df[op], errors="coerce")
    out["High"] = pd.to_numeric(df[hi], errors="coerce")
    out["Low"] = pd.to_numeric(df[lo], errors="coerce")
    out["Close"] = pd.to_numeric(df[cl], errors="coerce")
    out["Volume"] = pd.to_numeric(df[vol], errors="coerce") if vol else 0.0
    out["Turnover"] = pd.to_numeric(df[val], errors="coerce") if val else None
    return out
