import pandas as pd
import numpy as np
from CRR_option_pricing import CRROptionPricer as crr
from market_data_handler import MarketDataHandler as mdh
import matplotlib.pyplot as plt

class OptionContract:
    def __init__(self, K, T, r, sigma, option_type="call", exercise_type="american", position_type: int = -1):
        """
        Parameters:
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate
        sigma: Volatility of the underlying asset
        option_type: 'call' or 'put'
        exercise_type: 'european' or 'american'
        position_type: 1 for long, -1 for short
        """
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
        self.exercise_type = exercise_type
        self.position_type = position_type  # 1 for long, -1 for short

class HedgingParams:
    """
    Parameters describing the hedging strategy
    For now: rebalance to delta-neutral at a chosen frequency and/or interval
    """

    def __init__(self, steps_per_rebalance: int = 1, delta_threshold: float = 0.0):
        """
        Parameters:
        steps_per_rebalance: Number of time steps between each rebalance
        delta_threshold: Minimum change in delta to trigger a rebalance
        """
        self.steps_per_rebalance = steps_per_rebalance
        self.delta_threshold = delta_threshold

class TransactionCostModel:
    """
    Simple proportional transaction cost model
    """

    def __init__(self, rate: float = 0.0):
        self.rate = rate

    def cost(self, delta_hedge_change: float, price: float) -> float:
        """
        Calculate the transaction cost for a change in delta hedge position

        Parameters:
        delta_hedge_change: Change in delta hedge position (number of shares)
        price: Price S_t of the underlying asset at the time of the trade

        Returns:
        float - Transaction cost = |delta_hedge_change| * price * rate
        """
        return abs(delta_hedge_change) * price * self.rate

def generate_gbm_path(
        S0: float,
        T: float,
        N: int,
        mu: float,
        sigma: float,
        random_state: int | None = None
) -> np.ndarray:
    """
    Generate a Geometric Brownian Motion (GBM) path for the underlying asset price. Tested in main to be working correctly.
    dS_t = (mu * S_t * dt) + (sigma * S_t * dW_t) 

    S_t+1 = S_t * exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z_t)

    Parameters:
    S0: Initial stock price
    T: Time to maturity (in years)
    N: Number of time steps
    mu: Expected return (drift)
    sigma: Volatility of the underlying asset
    random_state: Seed for random number generator

    Returns:
    np.ndarray - Simulated GBM path of stock prices
    """
    # 
    rng = np.random.default_rng(random_state)
    dt = T / N

    S = np.empty(N+1)
    S[0] = S0

    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    for t in range(1, N+1):
        Z = rng.normal()
        S[t] = S[t-1] * np.exp(drift + diffusion * Z)

    return S

class AmericanDeltaHedgingSimulator:
    """
    Simulator for delta-hedging an American-style option along a single price path using a CRR binomial tree
    """
    def __init__(self, tree_steps: int = 100):
        self.tree_steps = tree_steps

    def simulate_path(
            self,
            price_path: np.ndarray,
            option: OptionContract,
            hedging_params: HedgingParams,
            transaction_cost_model: TransactionCostModel = TransactionCostModel(rate=0.0),
            accrual_cash_at_r: bool = True,
    ) -> dict:
        """
        Simulate a delta-hedging strategy along a single underlying price path

        Parameters: 
        price_path: np.ndarray - Simulated underlying asset price path
        option: OptionContract - The option contract to be hedged
        hedging_params: HedgingParams - Parameters for the hedging strategy
        transaction_cost_model: TransactionCostModel - Model for transaction costs
        accrual_cash_at_r: bool - If true, the cash amount accrues interest at rate r between steps
        

        Returns:
        dict with keys "S_path", "option_values", "deltas", "hedge_positions", "cash_positions", "portfolio_values", "pnl",
        "total_transaction_costs", "dt", "N"
        """

        S_path = np.asarray(price_path)
        N = len(S_path) - 1  # Number of time steps
        if N <= 0:
            raise ValueError("Price path must contain at least two prices (initial and final).")
        
        # Set CRR tree path
        tree_N = self.tree_steps if self.tree_steps is not None else N

        dt = option.T / N  # Time step size in years

        option_values = np.zeros(N+1)
        deltas = np.zeros(N+1)
        hedge_positions = np.zeros(N+1)
        cash_positions = np.zeros(N+1)
        portfolio_values = np.zeros(N+1)
        total_transaction_costs = 0.0

        # t = 0; initialize portfolio
        S0 = S_path[0]
        option_position = option.position_type  # 1 for long, -1 for short

        pricer_0 = crr(
            S0 = S0,
            K = option.K,
            T = option.T,
            r = option.r,
            sigma = option.sigma,
            option_type = option.option_type,
            exercise_type = option.exercise_type
        )

        # Option price and delta at t=0
        V_0, delta_0 = pricer_0.price_and_delta(N=tree_N)
        option_values[0] = V_0
        deltas[0] = delta_0

        # Initial hedge position. If you're short one option (option_position = -1), you buy delta shares to hedge
        # The hedge position indicates the number of shares held in the underlying asset
        hedge_positions[0] = -option_position * delta_0 

        # Initial cash position after setting up hedge. 
        # For a long call (option_position = 1), you pay V_0 to buy the option and buy delta_0 shares of stock to hedge
        cash_positions[0] = -option_position * V_0 - hedge_positions[0] * S0

        #  The value of the portfolio is the value of the option position plus the value of the hedge position plus cash
        portfolio_values[0] = option_position * option_values[0] + hedge_positions[0] * S0 + cash_positions[0]


        for k in range(1, N+1):
            S_t = S_path[k]  # Underlying price at time t
            t = k * dt  # Current time in years
            tau_remaining = max(option.T - t, 0.0)  # Time to maturity remaining

            if accrual_cash_at_r:
                # Accrue interest on cash position
                cash_positions[k] = cash_positions[k-1] * np.exp(option.r * dt)
            else:
                # No interest accrual
                cash_positions[k] = cash_positions[k-1]

            if tau_remaining > 0:
                pricer_t = crr(
                    S0 = S_t,
                    K = option.K,
                    T = tau_remaining,
                    r = option.r,
                    sigma = option.sigma,
                    option_type = option.option_type,
                    exercise_type = option.exercise_type
                )

                # Option price and delta at time t
                V_t, delta_t = pricer_t.price_and_delta(N=tree_N)

            else:
                if option.option_type == 'call':
                    V_t = max(0.0, S_t - option.K)
                else:
                    V_t = max(0.0, option.K - S_t)
                delta_t = 0.0  # Option has expired

            option_values[k] = V_t
            deltas[k] = delta_t

            # Determine if we need to rebalance hedge
            hedge_prev = hedge_positions[k-1]
            hedge_target = -option_position * delta_t  # delta-neutral target hedge position

            do_rebalance = False

            # Check if we need to rebalance based on time or delta change
            if (k % hedging_params.steps_per_rebalance) == 0 and tau_remaining != 0.0:  # Rebalance at specified frequency
                if abs(hedge_target - hedge_prev) >= hedging_params.delta_threshold:
                    do_rebalance = True

            hedge_positions[k] = hedge_prev
            if do_rebalance:
                delta_h = hedge_target - hedge_prev
                transaction_cost = transaction_cost_model.cost(delta_h, S_t)
                total_transaction_costs += transaction_cost

                # We rebalance by buying/selling delta_h shares at price S_t and paying transaction costs
                cash_positions[k] -= delta_h * S_t + transaction_cost
                hedge_positions[k] = hedge_target

            # At the final step, close the hedge and realize the payoff
            if k == N:
                if option.option_type == 'call':
                    payoff = max(0.0, S_t - option.K)
                else:
                    payoff = max(0.0, option.K - S_t)
                
                option_values[k] = payoff
                deltas[k] = 0.0

                # Close hedge position by selling/buying back the shares and paying transaction costs
                remaining_hedge = hedge_positions[k]
                if remaining_hedge != 0.0:
                    transaction_cost_close = transaction_cost_model.cost(remaining_hedge, S_t)
                    total_transaction_costs += transaction_cost_close
                    cash_positions[k] += remaining_hedge * S_t - transaction_cost_close
                    hedge_positions[k] = 0.0
                
            # Update portfolio value
            portfolio_values[k] = option_position * option_values[k] + hedge_positions[k] * S_t + cash_positions[k]

        pnl = portfolio_values[-1]

        results  = {
            "S_path": S_path,
            "option_values": option_values,
            "deltas": deltas,
            "hedge_positions": hedge_positions,
            "cash_positions": cash_positions,
            "portfolio_values": portfolio_values,
            "pnl": pnl,
            "total_transaction_costs": total_transaction_costs,
        }

        return results
    
    
                

def main():
    
    # Get market data for underlying asset
    handler = mdh(ticker='GOOGL')
    history = handler.get_underlying_history(period='1y', interval='1d')
    S_path = history['close'].values
    S0 = S_path[0]
    K = S0  # At-the-money strike
    N = len(S_path) - 1  # Number of time steps
    

    #K = 100.0  # Strike price of the option (price at which the option can be exercised)
    T = N / 252  # Time to maturity in years (60 days)
    r = 0.03  # Risk-free interest rate (annualized, 3%)
    sigma = 0.25  # Volatility of the underlying asset (annualized, 25%)
    S0 = 100.0  # Initial stock price
    K = 100.0   # Strike price
    S_path = generate_gbm_path(S0=S0, T=T, N=N, mu=0.05, sigma=sigma, random_state=42)
    S_2 = generate_gbm_path(S0=S0, T=T, N=N, mu=0.05, sigma=sigma)
    S_3 = generate_gbm_path(S0=S0, T=T, N=N, mu=0.05, sigma=sigma)
    S_4 = generate_gbm_path(S0=S0, T=T, N=N, mu=0.05, sigma=sigma)
    S_5 = generate_gbm_path(S0=S0, T=T, N=N, mu=0.05, sigma=sigma)
    
    

    # Create an option contract for a long American call option with the given parameters
    option = OptionContract(K=K, T=T, r=r, sigma=sigma, option_type='call', position_type=1)
    hedging_params = HedgingParams(steps_per_rebalance=1, delta_threshold=0.0)
    transaction_cost_model = TransactionCostModel(rate=0.001)

    simulator = AmericanDeltaHedgingSimulator(tree_steps=N)
    results = simulator.simulate_path(
        price_path=S_path,
        option=option,
        hedging_params=hedging_params,
        transaction_cost_model=transaction_cost_model,
        accrual_cash_at_r=True
    )

    print(results)

    
    plt.figure(figsize=(10, 7))
    plt.plot(results['portfolio_values'], label='Portfolio Value')
    plt.plot(results['option_values'], label='Option Value')
    plt.plot(results['cash_positions'], label='Cash Position')
    plt.plot(results['hedge_positions'] * S_path, label='Hedge Position Value')
    plt.plot(S_path, label='Underlying Price', alpha=0.5)
    #plt.plot(S_2, label='Underlying Price 2', alpha=0.5)
    #plt.plot(S_3, label='Underlying Price 3', alpha=0.5)
    #plt.plot(S_4, label='Underlying Price 4', alpha=0.5)
    #plt.plot(S_5, label='Underlying Price 5', alpha=0.5)
    plt.title('Delta Hedging Simulation')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    print(f"Final P&L from delta hedging: {results['pnl']:.2f}")
    

if __name__ == '__main__':
    main()