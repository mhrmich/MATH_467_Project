"""Black-Scholes pricing and Greeks for European options."""

from dataclasses import dataclass
from typing import Literal
from math import log, exp, sqrt

OptionType = Literal["call", "put"]


_INV_SQRT_2PI = 1.0 / sqrt(2.0 * 3.141592653589793)


def _norm_pdf(x: float) -> float:
    """Probability density function for the standard normal distribution."""
    return _INV_SQRT_2PI * exp(-0.5 * x * x)


def _norm_cdf(x: float) -> float:
    """Cumulative distribution function for the standard normal distribution."""
    k = 1.0 / (1.0 + 0.2316419 * abs(x))
    poly = (
        (((((1.330274429 * k) - 1.821255978) * k) + 1.781477937) * k)
        - 0.356563782
    ) * k + 0.319381530
    approximation = 1.0 - _norm_pdf(x) * poly
    return approximation if x >= 0 else 1.0 - approximation


@dataclass
class BlackScholesResult:
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


def black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType) -> BlackScholesResult:
    """Returns theoretical price and Greeks for a call or put option."""

    option_type = option_type.lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    if S <= 0 or K <= 0:
        raise ValueError("Underlying price and strike must be positive")
    if T <= 0:
        raise ValueError("Time to maturity must be positive")
    if sigma <= 0:
        raise ValueError("Volatility must be positive")

    sqrt_t = sqrt(T)
    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    nd1 = _norm_cdf(d1)
    nd2 = _norm_cdf(d2)
    pdf_d1 = _norm_pdf(d1)
    discount_factor = exp(-r * T)

    if option_type == "call":
        price = S * nd1 - K * discount_factor * nd2
        delta = nd1
        theta = (-(S * pdf_d1 * sigma) / (2.0 * sqrt_t)) - (r * K * discount_factor * nd2)
        rho = K * T * discount_factor * nd2
    else:
        nd1_minus = _norm_cdf(-d1)
        nd2_minus = _norm_cdf(-d2)
        price = K * discount_factor * nd2_minus - S * nd1_minus
        delta = nd1 - 1.0
        theta = (-(S * pdf_d1 * sigma) / (2.0 * sqrt_t)) + (r * K * discount_factor * nd2_minus)
        rho = -K * T * discount_factor * nd2_minus

    gamma = pdf_d1 / (S * sigma * sqrt_t)
    vega = S * pdf_d1 * sqrt_t

    return BlackScholesResult(price=price, delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)
