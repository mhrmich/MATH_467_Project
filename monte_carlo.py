import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from delta_hedge_simulator import (
    AmericanDeltaHedgingSimulator,
    OptionContract,
    HedgingParams,
    TransactionCostModel,
    generate_gbm_path
)

def run_monte_carlo_gbm(
        option: OptionContract,
        hedging_params: HedgingParams,
        transaction_cost_model: TransactionCostModel,
        simulator: AmericanDeltaHedgingSimulator,
        S0: float,
        T: float,
        N: int,
        mu: float,
        sigma: float,
        n_paths: int,
        lam: float = 0.0,
        random_state: int = None
    ) -> dict:

    """
    Run Monte Carlo simulation of delta hedging using Geometric Brownian Motion for the underlying asset price.
    
    Parameters:
    option: OptionContract - The option contract to be hedged
    hedging_params: HedgingParams - Parameters for the hedging strategy
    transaction_cost_model: TransactionCostModel - Model for transaction costs used in each rebalance
    simulator: AmericanDeltaHedgingSimulator - The delta hedging simulator
    S0: float - Initial stock price
    T: float - Time to maturity of the option (in years)
    N: int - Number of time steps in the simulation
    mu: float - Drift for the GBM model
    sigma: float - Volatility for the GBM model
    n_paths: int - Number of Monte Carlo paths to simulate
    lam: float - Parameter for the exponential utility function (default is 0.0, which corresponds to risk-neutral)
    random_state: int - Seed for random number generator (default is None)
    """

    rng = np.random.default_rng(random_state)

    pnls = np.empty(n_paths)
    costs = np.empty(n_paths)

    for m in range(n_paths):

        path_seed = rng.integers(0, 2**32 - 1)

        # Simulate a single GBM price path
        S_path = generate_gbm_path(
            S0=S0,
            T=T,
            N=N,
            mu=mu,
            sigma=sigma,
            random_state=path_seed
        )

        result = simulator.simulate_path(
            price_path=S_path,
            option=option,
            hedging_params=hedging_params,
            transaction_cost_model=transaction_cost_model,
            accrual_cash_at_r=True
        )

        pnls[m] = result["pnl"]
        costs[m] = result["total_transaction_costs"]

    mean_pnl = float(pnls.mean())  # Sample mean
    var_pnl = float(pnls.var())  # Sample variance
    mean_abs_pnl = float(np.mean(np.abs(pnls)))  # Mean absolute PnL
    mean_cost = float(costs.mean())

    return {
        "pnls": pnls,
        "transaction_costs": costs,
        "mean_pnl": mean_pnl,
        "var_pnl": var_pnl,
        "mean_abs_pnl": mean_abs_pnl,
        "mean_transaction_cost": mean_cost,
    }

def estimate_objective(
        k: int,
        epsilon: float,
        lam: float,  # lambda
        option: OptionContract,
        transaction_cost_model: TransactionCostModel,
        simulator: AmericanDeltaHedgingSimulator,
        S0: float,
        T: float,
        N: int,
        mu: float,
        sigma: float,
        n_paths: int,
        random_state: int = None
    ):

    hedging_params = HedgingParams(
        steps_per_rebalance=k,
        delta_threshold=epsilon
    )

    monte_carlo_stats = run_monte_carlo_gbm(
        option=option,
        hedging_params=hedging_params,
        transaction_cost_model=transaction_cost_model,
        simulator=simulator,
        S0=S0,
        T=T,
        N=N,
        mu=mu,
        sigma=sigma,
        n_paths=n_paths,
        random_state=random_state
    )

    pnls = monte_carlo_stats["pnls"]
    costs = monte_carlo_stats["transaction_costs"] 

    mse_pnl = float(np.mean(pnls**2))
    mean_cost = float(np.mean(costs))
    J_hat = mse_pnl + lam * mean_cost

    return J_hat, mse_pnl, mean_cost, monte_carlo_stats   

def main():
    S0 = 100.0  # Initial stock price
    K = 100.0   # Strike price
    T = 30 / 365.0  # Time to maturity (30 days)
    N = 30  # Number of time steps (rebalance daily)
    r = 0.01    # Risk-free interest rate
    sigma = 0.2  # Volatility of the underlying asset

    option = OptionContract(
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type='call',
        exercise_type='american',
        position_type=1  # Long position
    )

    hedging_params = HedgingParams(
        steps_per_rebalance=1,
        delta_threshold=0.0
    )

    transaction_cost_model = TransactionCostModel(rate=0.001)  # 0.1% transaction cost
    simulator = AmericanDeltaHedgingSimulator(tree_steps=100)

    mc_stats = run_monte_carlo_gbm(
        option=option,
        hedging_params=hedging_params,
        transaction_cost_model=transaction_cost_model,
        simulator=simulator,
        S0=S0,
        T=T,
        N=N,
        mu=0.05,
        sigma=sigma,
        n_paths=1000,
        random_state=42
    )
    pnls = mc_stats["pnls"]
    plt.hist(pnls, bins=50, edgecolor='black')
    plt.title('Histogram of PnL from Monte Carlo Simulation')
    plt.xlabel('PnL')
    plt.ylabel('Frequency')
    plt.show()

    print("Monte Carlo Simulation Results:")
    print(f"Mean PnL: {mc_stats['mean_pnl']}")
    print(f"Variance of PnL: {mc_stats['var_pnl']}")
    print(f"Mean Absolute PnL: {mc_stats['mean_abs_pnl']}")
    print(f"Mean Transaction Cost: {mc_stats['mean_transaction_cost']}")

if __name__ == "__main__":
    main()