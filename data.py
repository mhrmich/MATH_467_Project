"""Helpers for retrieving market data via yfinance."""

from __future__ import annotations

from typing import Any, Dict, List

import yfinance as yf


def get_spot_price(ticker: str) -> float:
    """Return the latest spot price for the provided ticker.

    Uses fast_info when available, otherwise falls back to the latest close.
    Raises ValueError if the price cannot be determined.
    """

    if not ticker:
        raise ValueError("Ticker symbol must be provided.")

    ticker_obj = yf.Ticker(ticker)

    # Attempt to use fast_info for real-time-ish price data
    price = None
    try:
        price = float(ticker_obj.fast_info.get("last_price"))
    except Exception:
        price = None

    if price is None:
        history = ticker_obj.history(period="1d")
        if history.empty or "Close" not in history:
            raise ValueError(f"Unable to retrieve price for ticker {ticker}.")
        price = float(history["Close"].iloc[-1])

    return price


def get_expirations(ticker: str) -> List[str]:
    """Return all available option expiration dates for the ticker."""

    if not ticker:
        raise ValueError("Ticker symbol must be provided.")

    ticker_obj = yf.Ticker(ticker)
    expirations = list(ticker_obj.options)
    if not expirations:
        raise ValueError(f"No option expirations found for {ticker}.")
    return expirations


def _extract_contracts(dataframe) -> List[Dict[str, Any]]:
    desired_columns = ["strike", "lastPrice", "bid", "ask", "impliedVolatility"]
    contracts: List[Dict[str, Any]] = []
    for _, row in dataframe.iterrows():
        contract = {col: row.get(col) for col in desired_columns}
        contracts.append(contract)
    return contracts


def get_option_chain(ticker: str, expiration: str) -> Dict[str, List[Dict[str, Any]]]:
    """Return call and put option contracts for a ticker and expiration date."""

    if not ticker:
        raise ValueError("Ticker symbol must be provided.")
    if not expiration:
        raise ValueError("Expiration date must be provided.")

    ticker_obj = yf.Ticker(ticker)
    chain = ticker_obj.option_chain(expiration)

    calls = _extract_contracts(chain.calls)
    puts = _extract_contracts(chain.puts)

    return {"calls": calls, "puts": puts}

