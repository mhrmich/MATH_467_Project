import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm # normal distribution, which we will use for Black-Scholes
import time
import statistics as stats


tickers = ["AAPL", "MSFT", "GOOGL"]

def get_all_calls(ticker):
    '''
    Fetches all call options for a given ticker across all expiration dates and returns them as a single DataFrame.
    '''
    
    ticker_obj = yf.Ticker(ticker)

    # DataFrame to hold all call options
    all_calls = pd.DataFrame()

    # Loop through each expiration date and concatenate call options
    for exp in ticker_obj.options:
        option_chain = ticker_obj.option_chain(exp)
        calls = option_chain.calls
        calls['expirationDate'] = exp
        all_calls = pd.concat([all_calls, calls], ignore_index=True)

    return all_calls


all_calls_df = get_all_calls("AAPL")
print("All AAPL Call Options DataFrame:")
print(all_calls_df.columns)

def binomial_tree_slow(S_0=100, K=100, r=0.06, T=1.0, N=3, u=1.1, d=1/1.1, opttype='call'):
        '''
        Binomial tree option pricing model (slow implementation (O(N^2))).
        Calculates a call option price using the CRR Binomial Tree model.

        :param S_0: Initial stock price
        :param K: Strike price 
        :param r: Risk-free interest rate (annualized)
        :param T: Time to maturity (in years)
        :param N: Number of time steps
        :param u: Up factor
        :param d: Down factor (reciprocal of up factor)
        :param opttype: Option type: 'call' or 'put'
        :return: Option value
        '''
        
        #Precompute constants
        dt = T/N # Length of each time step
        q = (np.exp(r*dt) - d) / (u - d) # Risk-neutral probability of upward move
        disc = np.exp(-r*dt) # Discount factor per time step

        # initialize asset prices at maturity (time step N)
        S = np.zeros(N+1) # S is a vector of possible stock prices at maturity
        S[0] = S_0 * (d**N) # Lowest possible price at maturity
        for j in range(1, N+1):
            S[j] = S[j-1] * (u/d) # S[j] is the outcome of j up moves and (N-j) down moves

        # initialize option values at maturity
        C = np.zeros(N+1) # C is a vector of option values at maturity
        for j in range(N+1):
            C[j] = max(0, S[j] - K) # Call option payoff

        # step backwards through tree
        for i in np.arange(N, 0, -1): # For each time step
            for j in range(i): # For each node at this time step
                C[j] = disc * (q * C[j+1] + (1-q) * C[j]) # Risk-neutral valuation

        return C[0]

print("Binomial Tree Slow Call Price:", binomial_tree_slow())

def binomial_tree_fast(S_0=100, K=100, r=0.06, T=1.0, N=3, u=1.1, d=1/1.1, opttype='call'):
        '''
        Binomial tree option pricing model.

        :param S_0: Initial stock price
        :param K: Strike price 
        :param r: Risk-free interest rate (annualized)
        :param T: Time to maturity (in years)
        :param N: Number of time steps
        :param u: Up factor
        :param d: Down factor (reciprocal of up factor)
        :param opttype: Option type: 'call' or 'put'
        :return: Option value
        '''
        #Precompute constants
        dt = T/N # Length of each time step
        q = (np.exp(r*dt) - d) / (u - d) # Risk-neutral probability of upward move
        disc = np.exp(-r*dt) # Discount factor per time step

        # initialize asset prices at maturity (time step N); utilize the formula that S(i, j) = S_0 * u^j * d^(i-j)
        S = S_0 * d**(np.arange(N, -1, -1)) * u**(np.arange(0, N+1, 1)) # Vectorized computation of stock prices at maturity
        
        # initialize option values at maturity
        C = np.maximum(np.zeros(N+1), S - K) # Vectorized computation of call option payoffs at maturity

        # step backwards through tree
        for i in np.arange(N, 0, -1): # For each time step
            C = disc * (q * C[1:i+1] + (1-q) * C[0:i]) # Vectorized risk-neutral valuation

        return C[0]

print("Binomial Tree Fast Call Price:", binomial_tree_fast())

#Def Cox_Rubinstein_Tree(S, K, T, r, sigma, N)



def bench(func, *args, warmup=1, repeat=7, number=1, **kwargs):
    # warmup
    for _ in range(warmup):
        func(*args, **kwargs)
    # timed runs
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(number):
            func(*args, **kwargs)
        t1 = time.perf_counter()
        times.append((t1 - t0) / number)
    return {
        "median": stats.median(times),
        "mean": stats.mean(times),
        "stdev": stats.pstdev(times),
        "runs": times,
    }

print("Benchmarking Binomial Tree Slow Implementation:")
print(bench(binomial_tree_slow, N=5000, warmup=3, repeat=5, number=3))
print("Benchmarking Binomial Tree Fast Implementation:")
print(bench(binomial_tree_fast, N=5000, warmup=3, repeat=5, number=3))