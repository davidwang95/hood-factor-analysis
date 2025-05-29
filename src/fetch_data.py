import yfinance as yf
import pandas_datareader as pdr
import pandas as pd
from datetime import datetime, date
import argparse
from pathlib import Path

def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch daily adjusted close prices for a given ticker.
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        pd.DataFrame: DataFrame with daily adjusted close prices
    """
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    print('DEBUG: yfinance DataFrame columns:', df.columns)
    print('DEBUG: yfinance DataFrame head:')
    print(df.head())
    if 'Adj Close' in df.columns:
        return df[['Adj Close']].rename(columns={'Adj Close': 'price'})
    elif 'Close' in df.columns:
        return df[['Close']].rename(columns={'Close': 'price'})
    else:
        raise ValueError("Neither 'Adj Close' nor 'Close' found in yfinance data.")

def fetch_ff_factors(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch Fama-French 5 factors + Momentum + Risk-free rate.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        pd.DataFrame: DataFrame with factor data
    """
    # Convert dates to datetime for pandas_datareader
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Fetch Fama-French 5 factors + Momentum
    ff5 = pdr.get_data_famafrench('F-F_Research_Data_5_Factors_2x3_daily', start=start, end=end)[0]
    
    # Fetch Risk-free rate
    rf = pdr.get_data_famafrench('F-F_Research_Data_Factors_daily', start=start, end=end)[0]['RF']
    
    # Combine factors
    factors = ff5.copy()
    factors['RF'] = rf
    
    return factors

def save_data(df: pd.DataFrame, filename: str):
    """
    Save DataFrame to CSV in data directory.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        filename (str): Name of the output file
    """
    # Create data directory if it doesn't exist
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Save to CSV
    df.to_csv(data_dir / filename)

def main():
    parser = argparse.ArgumentParser(description='Fetch stock prices and Fama-French factors')
    parser.add_argument('--ticker', type=str, default='HOOD', help='Stock ticker symbol')
    parser.add_argument('--start_date', type=str, default='2019-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=date.today().strftime('%Y-%m-%d'), 
                        help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Fetch and save stock prices
    prices = fetch_stock_data(args.ticker, args.start_date, args.end_date)
    save_data(prices, f'{args.ticker.lower()}_prices.csv')
    
    # Fetch and save Fama-French factors
    factors = fetch_ff_factors(args.start_date, args.end_date)
    save_data(factors, 'ff_factors.csv')
    
    print(f"Data saved to data/{args.ticker.lower()}_prices.csv and data/ff_factors.csv")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        print('ERROR:', e)
        traceback.print_exc() 