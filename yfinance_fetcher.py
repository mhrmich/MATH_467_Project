import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

class YFinanceFetcher:
    """
    Fetches option data from YFinance for a given ticker and expiration date.
    """

    def __init__(self):
        self.cache = {}

    def get_stock_data(self, ticker: str, period: str = '1y') -> Dict:
        """
        Gets comprehensive stock data from YFinance for a given ticker and period.
        """

        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')



            hist = stock.history(period=period)
            if hist.empty:
                print(f"No historical data found for {ticker}")
                return None

            return {
                'ticker': ticker,
                'current_price': current_price,
                'history': hist,
            }
        
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None

    def get_options_chain(self, ticker: str, expiration_date: Optional[str] = None) -> Dict:
        """
        Gets the option chains for a given ticker

        Parameters:
        ticker: str - The ticker symbol of the stock to fetch options for
        expiration_date: Optional[str] - The expiration date of the option to fetch (YYYY-MM-DD)

        Returns:
        Dict - A dictionary containing the option chain data
            ticker: str - The ticker symbol of the stock
            current_price: float - The current price of the stock
            expiration_date: datetime - The expiration date of the option
            expiration_string: str - The expiration date of the option in string format
            days_to_expiration: int - The number of days to expiration
            calls: DataFrame - The call options
            puts: DataFrame - The put options
            all_expirations: List[str] - The list of all available expiration dates
            timestamp: datetime - The timestamp of the data retrieval
        """

        try:
            stock = yf.Ticker(ticker)
            
            # Retrieve a list of all available expiration dates
            expirations = stock.options

            if not expirations:
                print(f"No options available for {ticker}")
                return None

            # If no expiration date is provided, use the first available expiration date
            if expiration_date is None:
                expiration_date = expirations[0]
                print(f"Using first available expiration date: {expiration_date}")

            elif expiration_date not in expirations:
                print(f"Invalid expiration date: {expiration_date}")
                print(f"Available expiration dates: {expirations[:5]}")
                print(f"Using first available expiration date: {expirations[0]}")
                expiration_date = expirations[0]

            # Get the current stock price info
            option_chain = stock.option_chain(expiration_date)
            info = stock.info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')

            # Parse expiration date 
            exp_date = datetime.strptime(expiration_date, "%Y-%m-%d")
            days_to_exp = (exp_date - datetime.now()).days

            return {
                'ticker': ticker,
                'current_price': current_price,
                'expiration_date': exp_date,
                'expiration_string': expiration_date,
                'days_to_expiration': days_to_exp,
                'calls': option_chain.calls,
                'puts': option_chain.puts,
                'all_expirations': expirations,
                'timestamp': datetime.now()
            }

        except Exception as e:
            print(f"Error fetching options chain for {ticker}: {e}")
            return None

def main():
    fetcher = YFinanceFetcher()
    stock_data = fetcher.get_stock_data("AAPL", period="1y")

    options_data = fetcher.get_options_chain("AAPL")
    print(options_data['calls'].head())

if __name__ == "__main__":
    main()