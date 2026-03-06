from __future__ import annotations

"""Universe file reader for text-based ticker lists."""

from pathlib import Path


def load_tickers(tickers_file: str) -> list[str]:
    """Load tickers from text file, ignoring blanks and comment lines."""
    p = Path(tickers_file)
    tickers = []
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        tickers.append(s)
    return sorted(set(tickers))
