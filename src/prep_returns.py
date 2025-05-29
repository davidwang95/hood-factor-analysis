import pandas as pd
from pathlib import Path

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load stock prices and factor data from CSV files.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Stock prices and factor data
    """
    data_dir = Path('data')
    
    # Load stock prices
    prices = pd.read_csv(data_dir / 'hood_prices.csv', index_col=0, parse_dates=True)
    
    # Load factor data
    factors = pd.read_csv(data_dir / 'ff_factors.csv', index_col=0, parse_dates=True)
    
    return prices, factors

def compute_excess_returns(prices: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """
    Compute excess returns by subtracting risk-free rate from stock returns.
    
    Args:
        prices (pd.DataFrame): Stock prices
        factors (pd.DataFrame): Factor data including risk-free rate
        
    Returns:
        pd.DataFrame: Excess returns and factors
    """
    # Compute daily returns
    returns = prices['price'].pct_change()
    
    # Create DataFrame with returns
    returns_df = pd.DataFrame({'returns': returns})
    
    # Align indices to date only
    if hasattr(returns_df.index, 'tz') and returns_df.index.tz is not None:
        returns_df.index = returns_df.index.tz_convert(None)
    returns_df.index = [d.date() for d in returns_df.index]
    factors.index = [pd.to_datetime(d).date() for d in factors.index]

    print('DEBUG: returns_df date range:', returns_df.index.min(), 'to', returns_df.index.max())
    print('DEBUG: factors date range:', factors.index.min(), 'to', factors.index.max())

    # Join with factors
    combined = returns_df.join(factors, how='inner')
    
    # Compute excess returns
    combined['excess_returns'] = combined['returns'] - combined['RF']
    
    # Drop the raw returns column
    combined = combined.drop('returns', axis=1)
    
    return combined

def save_data(df: pd.DataFrame, filename: str):
    """
    Save DataFrame to CSV in data directory.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        filename (str): Name of the output file
    """
    data_dir = Path('data')
    df.to_csv(data_dir / filename)

def main():
    # Load data
    prices, factors = load_data()
    
    # Compute excess returns and join with factors
    excess_returns = compute_excess_returns(prices, factors)
    
    # Save to CSV
    save_data(excess_returns, 'hood_excess_and_factors.csv')
    
    print("Data saved to data/hood_excess_and_factors.csv")
    print("\nFirst few rows of processed data:")
    print(excess_returns.head())

if __name__ == '__main__':
    main() 