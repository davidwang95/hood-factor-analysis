import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import List

def fetch_company_metrics(ticker: str) -> dict:
    """
    Fetch key metrics for a company using yfinance.
    
    Args:
        ticker (str): Company ticker symbol
        
    Returns:
        dict: Dictionary of company metrics
    """
    try:
        # Get company info
        company = yf.Ticker(ticker)
        info = company.info
        
        # Calculate trailing 12-month revenue CAGR
        # Get quarterly revenue data
        quarterly_revenue = company.quarterly_financials.loc['Total Revenue']
        if len(quarterly_revenue) >= 5:  # Need 5 quarters to calculate 1-year CAGR
            current_revenue = quarterly_revenue.iloc[0]
            year_ago_revenue = quarterly_revenue.iloc[4]
            revenue_cagr = (current_revenue / year_ago_revenue) ** (1/1) - 1
        else:
            revenue_cagr = np.nan
        
        # Get other metrics
        metrics = {
            'Ticker': ticker,
            'Revenue CAGR': revenue_cagr,
            'P/S Ratio': info.get('priceToSalesTrailing12Months', np.nan),
            'ROE': info.get('returnOnEquity', np.nan),
            'Beta': info.get('beta', np.nan)
        }
        
        return metrics
    
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return {
            'Ticker': ticker,
            'Revenue CAGR': np.nan,
            'P/S Ratio': np.nan,
            'ROE': np.nan,
            'Beta': np.nan
        }

def standardize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize metrics to z-scores.
    
    Args:
        df (pd.DataFrame): DataFrame with raw metrics
        
    Returns:
        pd.DataFrame: DataFrame with standardized metrics
    """
    # Create copy of DataFrame
    standardized = df.copy()
    
    # Standardize each metric (except Ticker)
    for col in df.columns:
        if col != 'Ticker':
            mean = df[col].mean()
            std = df[col].std()
            if std != 0:  # Avoid division by zero
                standardized[col] = (df[col] - mean) / std
            else:
                standardized[col] = 0
    
    return standardized

def save_data(df: pd.DataFrame, filename: str):
    """
    Save DataFrame to CSV in data directory.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        filename (str): Name of the output file
    """
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    df.to_csv(data_dir / filename, index=False)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Compare peer companies using standardized metrics')
    parser.add_argument('--tickers', nargs='+', 
                       default=['HOOD', 'SQ', 'COIN', 'IBKR', 'SOFI'],
                       help='List of ticker symbols to compare')
    
    args = parser.parse_args()
    
    # Fetch metrics for each company
    metrics_list = [fetch_company_metrics(ticker) for ticker in args.tickers]
    metrics_df = pd.DataFrame(metrics_list)
    
    # Standardize metrics
    standardized_df = standardize_metrics(metrics_df)
    
    # Save results
    save_data(standardized_df, 'peer_style_table.csv')
    
    print("\nRaw metrics:")
    print(metrics_df.to_string(index=False))
    print("\nStandardized metrics (z-scores):")
    print(standardized_df.to_string(index=False))
    print(f"\nResults saved to data/peer_style_table.csv")

if __name__ == '__main__':
    main() 