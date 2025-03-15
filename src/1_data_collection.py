"""
Data collection script for the SP100-Index-Tracker project.

This script downloads S&P 100 constituent data and their historical prices,
calculates returns, and splits the data into training and testing periods.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from utils import (
    ensure_dir, 
    get_sp100_tickers, 
    download_stock_data, 
    calculate_returns, 
    split_data_into_periods, 
    save_covariance_matrix,
    plot_cumulative_returns,
    plot_correlation_heatmap
)

def main():
    print("Starting S&P 100 data collection...")
    
    # Create directories
    ensure_dir('data')
    ensure_dir('results/visualizations')
    
    # Set parameters
    start_date = '2022-03-14'  # 3 years of data
    end_date = '2025-03-14'    # Current date in the prompt
    
    # Get S&P 100 tickers
    sp100_tickers = get_sp100_tickers()
    print(f"Found {len(sp100_tickers)} S&P 100 stocks.")
    
    # Save tickers to file
    pd.DataFrame({'ticker': sp100_tickers}).to_csv('data/sp100_tickers.csv', index=False)
    
    # Download stock data
    prices_df = download_stock_data(sp100_tickers, start_date, end_date, save_dir='data')
    print(f"Downloaded price data with shape: {prices_df.shape}")
    
    # Calculate returns
    returns_df = calculate_returns(prices_df, save_dir='data')
    print(f"Calculated returns with shape: {returns_df.shape}")
    
    # Split data into periods
    data_periods = split_data_into_periods(returns_df, save_dir='data')
    print(f"Split data into training and testing periods:")
    print(f"  - Training period: {data_periods['train_returns'].index[0]} to {data_periods['train_returns'].index[-1]}")
    print(f"  - Testing period: {data_periods['test_returns'].index[0]} to {data_periods['test_returns'].index[-1]}")
    
    # Calculate and save covariance matrix
    cov_matrix = save_covariance_matrix(data_periods['train_returns'], save_dir='data')
    print(f"Calculated covariance matrix with shape: {cov_matrix.shape}")
    
    # Analyze data
    print("\nAnalyzing data...")
    
    # Basic statistics of returns
    stats = returns_df.describe()
    print("\nBasic statistics of daily returns:")
    print(stats.loc[['mean', 'std', 'min', 'max']].T.sort_values('mean', ascending=False).head(10))
    
    # Plot cumulative returns
    print("\nPlotting cumulative returns...")
    plot_cumulative_returns(returns_df, 
                           title='Cumulative Returns of S&P 100 Stocks',
                           save_path='results/visualizations/cumulative_returns.png')
    
    # Plot correlation heatmap
    print("\nPlotting correlation heatmap...")
    plot_correlation_heatmap(returns_df, n_stocks=30,
                            save_path='results/visualizations/correlation_heatmap.png')
    
    # Save summary statistics to file
    summary_stats = pd.DataFrame({
        'mean': returns_df.mean() * 252,  # Annualized return
        'std': returns_df.std() * np.sqrt(252),  # Annualized volatility
        'sharpe': (returns_df.mean() * 252) / (returns_df.std() * np.sqrt(252)),  # Sharpe ratio
        'correlation': returns_df.corrwith(returns_df['^OEX'])  # Correlation with S&P 100
    }).sort_values('correlation', ascending=False)
    
    summary_stats.to_csv('data/summary_statistics.csv')
    print(f"\nSaved summary statistics to data/summary_statistics.csv")
    
    print("\nData collection and analysis complete!")

if __name__ == "__main__":
    main()
