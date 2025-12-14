import yfinance as yf
import numpy as np
import pandas as pd

class MarketDataHandler:
    def __init__(self, ticker: str):
        self.ticker = ticker


    def get_underlying_history(self, period: str = '1y', interval: str = '1d') -> pd.DataFrame:
        """
        Gets the underlying history for a given ticker and period

        Parameters:
        period: str - The period for which to get the history (e.g., '1y' for 1 year)
        interval: str - The data interval (e.g., '1d' for daily data

        Returns:
        pd.DataFrame - Daily historical data for the underlying asset
        The DataFrame contains columns: ['open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits']

        """

        try:
            # obtains the ticker data from yfinance, drops nulls (which there shouldn't be) and renames the columns to lowercase
            yf_ticker = yf.Ticker(self.ticker)  
            history = yf_ticker.history(period=period, interval=interval) 

            history = history.dropna(subset=['Close'])
            history = history.rename(columns=str.lower)

        except Exception as e:
            raise RuntimeError(f"Error getting underlying history for {self.ticker}: {e}")

        return history

    def get_options_chain(self) -> pd.DataFrame:
        """
        Gets the options chain containing all options for a given ticker
        """

        try:
            yf_ticker = yf.Ticker(self.ticker)
            expiration_dates = yf_ticker.options
            all_rows = []  # list of dictionaries to store all option chain data

            for expiration_date in expiration_dates:
                options_chain = yf_ticker.option_chain(expiration_date)
                for option_type, df in [('call', options_chain.calls), ('put', options_chain.puts)]:
                    if df is None or df.empty:
                        continue
                    df = df.copy()
                    df['expiration_date'] = pd.to_datetime(expiration_date)
                    df['option_type'] = option_type
                    all_rows.append(df)

            chain_df = pd.concat(all_rows, ignore_index=True)
            return chain_df
        
        except Exception as e:
            raise ValueError(f"Error getting options chain for {self.ticker}: {e}")


    def build_option_universe(
            self,
            chain_df: pd.DataFrame,
            spot_price: float,
            as_of: pd.Timestamp = None,
            min_tau_days: int = 30,
            max_tau_days: int = 120,
            max_moneyness_deviation: float = 0.1,
            min_open_interest: int = 10
        ) -> pd.DataFrame:
    
        """
        Generates  the option universe, a filtered list of useful options for delta hedging for a given ticker

        Parameters:
        chain_df: pd.DataFrame - The options chain dataframe as returned by get_options_chain
        spot_price: float - The current price of the underlying asset
        as_of: pd.Timestamp - The date for which the universe is being built
        min_tau_days: int - The minimum time to maturity in days
        max_tau_days: int - The maximum time to maturity in days
        max_moneyness_deviation: float - The maximum deviation from the at-the-money price
        min_open_interest: int - The minimum open interest for the option

        Returns:
        pd.DataFrame - The option universe dataframe
        """

        # If as_of is not provided, use today's date
        if as_of is None:
            as_of = pd.Timestamp.now().normalize()

        df = chain_df.copy()
        
        # tau days (days to expiration) filtering
        df['tau_days'] = (df['expiration_date'] - as_of).dt.days
        df = df[df['tau_days'].between(min_tau_days, max_tau_days)]
        if df.empty:
            raise ValueError("No options found within the specified tau_days range.")
        df['tau_years'] = df['tau_days'] / 365.0  # Number of years to expiration (more standard format)

        # moneyness filtering
        df['moneyness'] = spot_price / df['strike']
        df = df[df['moneyness'].between(1 - max_moneyness_deviation, 1 + max_moneyness_deviation)]
        if df.empty:
            raise ValueError("No options found within the specified moneyness range.")

        # open interest filtering
        if "openInterest" in df.columns:
            df = df[df['openInterest'] >= min_open_interest]  # Filter by minimum open interest
        else:
            pass  # If openInterest column is not present, skip this filter
        if df.empty:
            raise ValueError("No options found after open interest filter.")
        
        has_bid_ask = ("bid" in df.columns) and ("ask" in df.columns)
        if has_bid_ask:
            df['mid_price'] = (df['bid'] + df['ask']) / 2.0
            mask_bad_mid = (df['mid_price'] <= 0) | (df['mid_price'].isna())
            if "lastPrice" in df.columns:
                df.loc[mask_bad_mid, 'mid_price'] = df.loc[mask_bad_mid, 'lastPrice']
        elif "lastPrice" in df.columns:
            df['mid_price'] = df['lastPrice']
        else:
            df['mid_price'] = np.nan  # If no price information is available, prevents errors

        def intrinsic_row(row):
            if row['option_type'] == 'call':
                return max(0.0, spot_price - row['strike'])
            elif row['option_type'] == 'put':
                return max(0.0, row['strike'] - spot_price)
            
        df['intrinsic_value'] = df.apply(intrinsic_row, axis=1)
        df['time_value'] = df['mid_price'] - df['intrinsic_value']

        return df


    def select_near_atm_option(
            self,
            universe_df: pd.DataFrame,
            target_tau_days: int = 60,
            option_type: str = 'call'
    ) -> pd.Series:
        """
        Selects the option from the universe that is closest to at-the-money and has time to maturity closest to target_tau_days

        Parameters:
        universe_df: pd.DataFrame - The option universe dataframe as returned by build_option_universe
        target_tau_days: int - The target time to maturity in days
        option_type: str - The type of option to select ('call' or 'put')

        Returns:
        pd.Series - The selected option row
        """

        df = universe_df[universe_df['option_type'] == option_type].copy()
        if df.empty:
            raise ValueError(f"No options of type {option_type} found in the universe.")

        # Find the option with tau_days closest to target_tau_days
        df['tau_diff'] = (df['tau_days'] - target_tau_days).abs()
        min_tau = df['tau_diff'].min()
        df_tau = df[df['tau_diff'] == min_tau]

        # From the options with the closest tau, find the one closest to at-the-money (moneyness closest to 1)
        df_tau["moneyness_diff"] = (df_tau['moneyness'] - 1.0).abs()
        idx = df_tau['moneyness_diff'].idxmin()

        return df_tau.loc[idx]


def main():

    # Example usage of MarketDataHandler
    pd.set_option('display.max_columns', None)
    handler = MarketDataHandler(ticker='AAPL')  # Market data handler for Apple
    history = handler.get_underlying_history(period='1y') # 1 year of daily data for Apple
    spot = history['close'].iloc[-1]  # Current spot price is the last closing price
    print(f"Current spot price for {handler.ticker}: {spot}")
    
    options_chain = handler.get_options_chain()
    universe = handler.build_option_universe(options_chain, spot_price=spot)
    selected_option = handler.select_near_atm_option(universe, target_tau_days=60, option_type='call')

    print()
    print("Selected Option:")
    print(selected_option)
    

if __name__ == '__main__':
    main()