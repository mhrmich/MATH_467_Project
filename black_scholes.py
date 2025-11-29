"""Black-Scholes pricing and Greeks for European options.

This module contains pure mathematical computations without any side effects.
"""

from dataclasses import dataclass
from typing import Literal
from math import log, exp, sqrt

OptionType = Literal["call", "put"]

# Constant for normal distribution calculations
_INV_SQRT_2PI = 1.0 / sqrt(2.0 * 3.141592653589793)


def _norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return _INV_SQRT_2PI * exp(-0.5 * x * x)


def _norm_cdf(x: float) -> float:
    """Approximation of the standard normal cumulative distribution function.

    This uses the Abramowitz and Stegun approximation (formula 7.1.26), which
    offers sufficient accuracy for option pricing while only depending on
    elementary functions.
    """

    sign = 1.0 if x >= 0 else -1.0
    x_abs = abs(x)
    t = 1.0 / (1.0 + 0.2316419 * x_abs)
    polynomial = (
        t
        * (
            0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
        )
    )
    cdf = 1.0 - _norm_pdf(x_abs) * polynomial
    return cdf if sign > 0 else 1.0 - cdf


@dataclass
class BlackScholesResult:
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


def _validate_inputs(S: float, K: float, T: float, sigma: float) -> None:
    if S <= 0 or K <= 0:
        raise ValueError("Spot price and strike price must be positive.")
    if T <= 0:
        raise ValueError("Time to maturity must be positive.")
    if sigma <= 0:
        raise ValueError("Volatility must be positive.")


def black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType) -> BlackScholesResult:
    """Returns theoretical price and Greeks for a call or put option."""

    _validate_inputs(S, K, T, sigma)

    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type == "call":
        price = S * _norm_cdf(d1) - K * exp(-r * T) * _norm_cdf(d2)
        delta = _norm_cdf(d1)
        theta = (
            -(S * _norm_pdf(d1) * sigma) / (2 * sqrt(T))
            - r * K * exp(-r * T) * _norm_cdf(d2)
        )
        rho = K * T * exp(-r * T) * _norm_cdf(d2)
    elif option_type == "put":
        price = K * exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
        delta = _norm_cdf(d1) - 1
        theta = (
            -(S * _norm_pdf(d1) * sigma) / (2 * sqrt(T))
            + r * K * exp(-r * T) * _norm_cdf(-d2)
        )
        rho = -K * T * exp(-r * T) * _norm_cdf(-d2)
    else:
        raise ValueError("option_type must be either 'call' or 'put'.")

    gamma = _norm_pdf(d1) / (S * sigma * sqrt(T))
    vega = S * _norm_pdf(d1) * sqrt(T)

    return BlackScholesResult(price=price, delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)

