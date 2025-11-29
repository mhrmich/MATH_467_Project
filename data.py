"""Data helpers for retrieving option market data using yfinance."""

from __future__ import annotations

from typing import Dict, List

import yfinance as yf


class DataRetrievalError(RuntimeError):
    """Raised when market data cannot be retrieved."""


def get_spot_price(ticker: str) -> float:
    """Return the latest close or live price for a ticker."""

    try:
        asset = yf.Ticker(ticker)
        history = asset.history(period="1d")
        if not history.empty:
            return float(history["Close"].iloc[-1])

        fast_info = getattr(asset, "fast_info", None)
        if fast_info and "last_price" in fast_info:
            return float(fast_info.get("last_price"))
    except Exception as exc:  # noqa: BLE001
        raise DataRetrievalError(f"Failed to fetch spot price for {ticker}: {exc}") from exc

    raise DataRetrievalError(f"No price data available for {ticker}.")


def get_expirations(ticker: str) -> List[str]:
    """Return all available option expiration dates for a ticker."""

    try:
        asset = yf.Ticker(ticker)
        expirations = getattr(asset, "options", [])
    except Exception as exc:  # noqa: BLE001
        raise DataRetrievalError(f"Failed to fetch expirations for {ticker}: {exc}") from exc

    if not expirations:
        raise DataRetrievalError(f"No expirations available for {ticker}.")

    return list(expirations)


def get_option_chain(ticker: str, expiration: str) -> Dict[str, List[Dict]]:
    """Return the option chain for a ticker and expiration date."""

    try:
        chain = yf.Ticker(ticker).option_chain(expiration)
    except Exception as exc:  # noqa: BLE001
        raise DataRetrievalError(
            f"Failed to fetch option chain for {ticker} on {expiration}: {exc}"
        ) from exc

    calls = _extract_contracts(chain.calls)
    puts = _extract_contracts(chain.puts)
    if not calls and not puts:
        raise DataRetrievalError(
            f"Option chain for {ticker} on {expiration} did not contain data."
        )

    return {"calls": calls, "puts": puts}


def _extract_contracts(frame) -> List[Dict]:
    fields = ["strike", "lastPrice", "bid", "ask", "impliedVolatility"]
    if frame is None:
        return []
    try:
        filtered = frame[fields].fillna(0.0)
    except Exception:
        return []
    return [dict(row) for _, row in filtered.iterrows()]
