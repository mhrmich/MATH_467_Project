"""Black-Scholes pricing and Greeks for European options."""
from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, pi, sqrt
from typing import Literal

OptionType = Literal["call", "put"]


def _norm_cdf(x: float) -> float:
    """Cumulative distribution function for the standard normal distribution."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Probability density function for the standard normal distribution."""
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


@dataclass
class BlackScholesResult:
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


def black_scholes(
    option_type: OptionType,
    spot: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
) -> BlackScholesResult:
    """Calculate the Black-Scholes price and Greeks for a European option.

    Args:
        option_type: Either "call" or "put".
        spot: Current underlying price (S).
        strike: Strike price (K).
        time_to_maturity: Time to expiration in years (T).
        risk_free_rate: Continuously compounded risk-free rate (r).
        volatility: Implied volatility (sigma).

    Returns:
        BlackScholesResult containing the price and Greeks (Delta, Gamma, Theta,
        Vega, Rho).
    """
    option_type = option_type.lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if time_to_maturity <= 0:
        raise ValueError("time_to_maturity must be positive")
    if volatility <= 0:
        raise ValueError("volatility must be positive")

    sqrt_t = sqrt(time_to_maturity)
    d1 = (
        log(spot / strike)
        + (risk_free_rate + 0.5 * volatility**2) * time_to_maturity
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    nd1 = _norm_cdf(d1)
    nd2 = _norm_cdf(d2)
    pdf_d1 = _norm_pdf(d1)

    discount_factor = exp(-risk_free_rate * time_to_maturity)

    if option_type == "call":
        price = spot * nd1 - strike * discount_factor * nd2
        delta = nd1
        theta = (
            -(spot * pdf_d1 * volatility) / (2 * sqrt_t)
            - risk_free_rate * strike * discount_factor * nd2
        )
        rho = strike * time_to_maturity * discount_factor * nd2
    else:
        nd1_minus = _norm_cdf(-d1)
        nd2_minus = _norm_cdf(-d2)
        price = strike * discount_factor * nd2_minus - spot * nd1_minus
        delta = nd1 - 1
        theta = (
            -(spot * pdf_d1 * volatility) / (2 * sqrt_t)
            + risk_free_rate * strike * discount_factor * nd2_minus
        )
        rho = -strike * time_to_maturity * discount_factor * nd2_minus

    gamma = pdf_d1 / (spot * volatility * sqrt_t)
    vega = spot * pdf_d1 * sqrt_t

    return BlackScholesResult(
        price=price,
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega,
        rho=rho,
    )


if __name__ == "__main__":
    parameters = {
        "spot": 100.0,
        "strike": 100.0,
        "time_to_maturity": 1.0,
        "risk_free_rate": 0.05,
        "volatility": 0.2,
    }

    call_result = black_scholes("call", **parameters)
    put_result = black_scholes("put", **parameters)

    print("Sample parameters:", parameters)
    print("\nCall option:", call_result)
    print("Put option:", put_result)
