"""Command-line tool to price options with Black-Scholes values."""

from __future__ import annotations

from datetime import datetime, date
from typing import Any, Dict, List

import pandas as pd

from black_scholes import OptionType, black_scholes
from data import DataRetrievalError, get_expirations, get_option_chain, get_spot_price


def _prompt_float(prompt: str, default: float | None = None) -> float:
    while True:
        raw = input(prompt).strip()
        if not raw and default is not None:
            return default
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a valid number.")
            continue
        return value


def _prompt_index(options: List[str]) -> int:
    while True:
        choice = input("Choose an expiration by index: ").strip()
        try:
            idx = int(choice)
            if 0 <= idx < len(options):
                return idx
        except ValueError:
            pass
        print("Invalid selection. Please enter a valid index from the list.")


def _parse_time_to_expiration(expiration: str) -> float:
    try:
        exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
    except ValueError:
        exp_date = datetime.fromisoformat(expiration).date()
    days = (exp_date - date.today()).days
    if days <= 0:
        raise ValueError("Expiration date must be in the future.")
    return days / 365.0


def _calculate_rows(
    ticker: str,
    expiration: str,
    spot: float,
    r: float,
    sigma: float,
) -> List[Dict[str, Any]]:
    T = _parse_time_to_expiration(expiration)
    option_chain = get_option_chain(ticker, expiration)
    rows: List[Dict[str, Any]] = []

    for option_type, contracts in option_chain.items():
        bs_type: OptionType = "call" if option_type == "calls" else "put"
        for contract in contracts:
            strike = float(contract.get("strike", 0))
            market_price = float(contract.get("lastPrice", 0))
            try:
                bs = black_scholes(spot, strike, T, r, sigma, bs_type)
            except Exception as exc:  # noqa: BLE001
                print(f"Skipping strike {strike} due to: {exc}")
                continue

            rows.append(
                {
                    "strike": strike,
                    "option_type": bs_type,
                    "market_price": market_price,
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
    ticker = input("Enter a ticker (e.g., AAPL): ").strip().upper()
    if not ticker:
        print("Ticker cannot be empty.")
        return

    try:
        spot = get_spot_price(ticker)
        expirations = get_expirations(ticker)
    except DataRetrievalError as exc:
        print(exc)
        return

    print(f"\nSpot price for {ticker}: {spot:.2f}\n")
    print("Available expirations:")
    for idx, exp in enumerate(expirations):
        print(f"[{idx}] {exp}")

    index = _prompt_index(expirations)
    expiration = expirations[index]

    r = _prompt_float("Enter risk-free rate r (default 0.04): ", default=0.04)
    sigma = _prompt_float("Enter volatility sigma (e.g., 0.2): ")

    try:
        rows = _calculate_rows(ticker, expiration, spot, r, sigma)
    except (DataRetrievalError, ValueError) as exc:
        print(exc)
        return

    if not rows:
        print("No option data available to display.")
        return

    df = pd.DataFrame(rows)
    df.sort_values(by=["strike", "option_type"], inplace=True)

    print("\nBlack-Scholes valuation:")
    print(df.to_string(index=False, float_format="{:.4f}".format))

    filename = f"{ticker}_{expiration}_bs_results.csv"
    df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    main()
