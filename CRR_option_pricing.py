import numpy as np

class CRROptionPricer:
    "Cox-Ross-Rubinstein binomial option pricing model"

    def __init__(self, S0, K, T, r, sigma, option_type='call', exercise_type='american'):
        self.S0 = S0  # Initial stock price
        self.K = K      # Strike price
        self.T = T      # Time to maturity (in years)
        self.r = r      # Risk-free interest rate
        self.sigma = sigma  # Volatility of the underlying asset
        self.option_type = option_type  # 'call' or 'put'
        self.exercise_type = exercise_type  # 'european' or 'american'

    def _payoff(self, S: np.ndarray) -> np.ndarray:
        """
        Calculate the payoff of the option at a given stock price.
        Parameters:
        S: np.ndarray - The stock prices at a given time step
        Returns:
        np.ndarray - The payoffs of the option
        """
        if self.option_type == 'call':
            return np.maximum(S - self.K, 0)
        else:
            return np.maximum(self.K - S, 0)

    def price_and_delta(self, N: int = 100) -> tuple[float, float]:
        '''
        Calculate the option pricing using the Cox-Ross-Rubinstein binomial tree model.
        '''

        dt = self.T / N  # Length of each time step
        u = np.exp(self.sigma * np.sqrt(dt)) # Up factor
        d = 1/u  # Down factor
        discount = np.exp(-self.r * dt) # Discount factor per time step
        p = (np.exp(self.r * dt) - d) / (u - d)  # Risk-neutral probability of upward move
        if not 0 <= p <= 1:
            raise ValueError(f"Risk-neutral probability p is not between 0 and 1: p = {p}")

        #  Initialize the vector of asset prices at maturity, S_N is a vector of dimension N+1
        S_N = self.S0 * d ** np.arange(N, -1, -1) * u**np.arange(0, N+1, 1)  # Stock prices at maturity

        # Initialize the vector of option values at maturity
        if self.option_type == 'call':
            option_values = np.maximum(S_N - self.K, 0)
        else:
            option_values = np.maximum(self.K - S_N, 0)

        #  Value of option and stock values at the next time step, used to calculate delta at t = 0
        V_d = V_u = None
        S_d = S_u = None

        #  Initialize the vector of asset prices at the current time step, S is a vector of dimension N+1
        S = S_N

        # Backward induction from time step N-1 to time step 0
        for i in range(N - 1, -1, -1):

            # Update the stock prices, so S starts at time step i, for i from N-1 to 0, at each iteration
            S = S[:i+1]/d  #  Update the stock prices for the next time step

            #  The option value at each node is the discounted expected value of the option at the next time step
            #  option_Values is a vector of dimension i+1, option_values[1:i+2] is for upward steps and option_values[0:i+1] is for downward steps
            option_values = discount * (p * option_values[1:i+2] + (1-p) * option_values[0:i+1])  

            if self.exercise_type == 'american':
                intrinsic_values = self._payoff(S)  # The intrinsic value is the payoff of exercising the option immediately

                #  If the intrinsic value is greater than the expected value, exercise the option. 
                #  This line is fundamentally why the binomial model overcomes the limitations of the Black-Scholes model for American options.
                # At each node, the option value is the maximum of the 
                option_values = np.maximum(option_values, intrinsic_values)  
        
            #  Store the stock values and option values at the next time step for the calculation of delta at t = 0
            if i == 1:
                S_d, S_u = S[0], S[1]
                V_d, V_u = option_values[0], option_values[1]

        option_price_0 = option_values[0]
        delta_0 = (V_u - V_d) / (S_u - S_d)
        return option_price_0, delta_0

        
def main():
    pricer = CRROptionPricer(S0=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type='call', exercise_type='american')
    option_price, delta = pricer.price_and_delta(N=100)
    print(f"Call Option Price: {option_price}")
    print(f"Call Option Delta: {delta}")

if __name__ == "__main__":
    main()