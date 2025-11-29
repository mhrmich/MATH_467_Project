"""Command-line interface for fetching option data and computing Black-Scholes values."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import List

import pandas as pd

from black_scholes import BlackScholesResult, black_scholes
from data import get_expirations, get_option_chain, get_spot_price


DEFAULT_RISK_FREE_RATE = 0.04


def _prompt_ticker() -> str:
    ticker = input("Enter a ticker (e.g., AAPL): ").strip().upper()
    if not ticker:
        raise ValueError("Ticker cannot be empty.")
    return ticker


def _prompt_expiration(expirations: List[str]) -> str:
    print("\nAvailable expirations:")
    for idx, exp in enumerate(expirations, start=1):
        print(f"[{idx}] {exp}")

    while True:
        selection = input("Choose an expiration by number: ").strip()
        if not selection.isdigit():
            print("Please enter a valid number.")
            continue

        index = int(selection) - 1
        if 0 <= index < len(expirations):
            return expirations[index]
        print("Selection out of range. Try again.")


def _prompt_float(prompt: str, default: float | None = None) -> float:
    while True:
        user_input = input(prompt).strip()
        if user_input == "" and default is not None:
            return default
        try:
            return float(user_input)
        except ValueError:
            print("Please enter a valid number.")


def _calculate_time_to_expiry(expiration: str) -> float:
    expiration_date = datetime.strptime(expiration, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    days = (expiration_date - now).days
    if days <= 0:
        raise ValueError("Expiration date must be in the future.")
    return days / 365.0


def _compute_rows(option_data, option_type: str, S: float, T: float, r: float, sigma: float):
    rows = []
    for contract in option_data:
        strike = float(contract.get("strike", 0))
        if strike <= 0:
            continue

        market_price = contract.get("lastPrice")
        if market_price is None:
            bid = contract.get("bid")
            ask = contract.get("ask")
            if bid is not None and ask is not None:
                market_price = (bid + ask) / 2
            else:
                market_price = 0.0

        try:
            bs: BlackScholesResult = black_scholes(S, strike, T, r, sigma, option_type)  # type: ignore[arg-type]
        except ValueError as exc:
            print(f"Skipping strike {strike}: {exc}")
            continue

        rows.append(
            {
                "strike": strike,
                "option_type": option_type,
                "market_price": float(market_price),
                "bs_price": bs.price,
                "delta": bs.delta,
                "gamma": bs.gamma,
                "theta": bs.theta,
                "vega": bs.vega,
                "rho": bs.rho,
            }
        )
    return rows


def main() -> None:
    try:
        ticker = _prompt_ticker()
        spot_price = get_spot_price(ticker)
        expirations = get_expirations(ticker)
        expiration = _prompt_expiration(expirations)
        T = _calculate_time_to_expiry(expiration)

        print("\nUsing spot price:", spot_price)
        r = _prompt_float(f"Enter risk-free rate r (default {DEFAULT_RISK_FREE_RATE}): ", DEFAULT_RISK_FREE_RATE)
        sigma = _prompt_float("Enter volatility sigma (e.g., 0.25): ")

        print("\nFetching option chain...\n")
        chain = get_option_chain(ticker, expiration)

        rows = []
        rows.extend(_compute_rows(chain["calls"], "call", spot_price, T, r, sigma))
        rows.extend(_compute_rows(chain["puts"], "put", spot_price, T, r, sigma))

        if not rows:
            print("No option data available to display.")
            return

        df = pd.DataFrame(rows)
        df.sort_values(by=["strike", "option_type"], inplace=True)
        print(df.to_string(index=False, justify="center"))

        filename = f"{ticker}_{expiration}_bs_results.csv"
        df.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")
    except Exception as exc:  # Broad catch to provide friendly error messages for CLI usage
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

