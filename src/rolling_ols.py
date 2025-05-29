import pandas as pd
import numpy as np
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
from pathlib import Path

def load_data() -> pd.DataFrame:
    """
    Load excess returns and factor data.
    
    Returns:
        pd.DataFrame: Combined data with excess returns and factors
    """
    data_dir = Path('data')
    return pd.read_csv(data_dir / 'hood_excess_and_factors.csv', index_col=0, parse_dates=True)

def run_rolling_regression(data: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Run rolling OLS regression of excess returns on factors.
    
    Args:
        data (pd.DataFrame): Data with excess returns and factors
        window (int): Rolling window size in days
        
    Returns:
        pd.DataFrame: Rolling regression coefficients
    """
    # Prepare X (factors) and y (excess returns)
    X = data.drop('excess_returns', axis=1)
    y = data['excess_returns']
    
    # Add constant to X for intercept
    X = sm.add_constant(X)
    
    # Run rolling regression
    model = RollingOLS(y, X, window=window)
    results = model.fit()
    
    # Extract coefficients
    betas = results.params
    
    # Rename columns to be more descriptive
    betas.columns = ['alpha'] + [f'beta_{col}' for col in X.columns[1:]]
    
    return betas

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
    data = load_data()
    
    # Run rolling regression
    betas = run_rolling_regression(data)
    
    # Save results
    save_data(betas, 'rolling_betas.csv')
    
    print("Rolling regression results saved to data/rolling_betas.csv")
    print("\nFirst few rows of rolling betas:")
    print(betas.head())
    
    # Print summary statistics
    print("\nSummary statistics of rolling betas:")
    print(betas.describe())

if __name__ == '__main__':
    main() 