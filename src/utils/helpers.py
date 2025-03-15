"""
Helper functions for the SP100-Index-Tracker project.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Add this import for correlation heatmaps
from datetime import datetime
import yfinance as yf
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

def ensure_dir(directory):
    """
    Create directory if it doesn't exist.
    
    Parameters:
    -----------
    directory : str
        Path to directory to create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_sp100_tickers():
    """
    Get the list of S&P 100 tickers.
    For a real project, you would get this from an official source,
    but for this example we'll use a hardcoded recent list.
    """
    # This is a sample of S&P 100 constituents as of March 2025
    # In a real implementation, you should get this from an official source
    sp100_tickers = [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA', 'BRK-B', 'UNH',
        'JPM', 'V', 'XOM', 'JNJ', 'PG', 'MA', 'HD', 'CVX', 'AVGO', 'LLY',
        'MRK', 'PEP', 'COST', 'ABBV', 'KO', 'PFE', 'TMO', 'ORCL', 'ACN', 'MCD',
        'BAC', 'CSCO', 'CRM', 'ABT', 'DHR', 'WMT', 'DIS', 'ADBE', 'TXN', 'AMD',
        'CMCSA', 'VZ', 'NFLX', 'INTC', 'PM', 'NEE', 'RTX', 'T', 'LIN', 'WFC',
        'BMY', 'UPS', 'HON', 'QCOM', 'COP', 'AMGN', 'INTU', 'LOW', 'DE', 'IBM',
        'CAT', 'AMAT', 'GE', 'SPGI', 'SBUX', 'MS', 'BA', 'GS', 'MDLZ', 'CVS',
        'BLK', 'GILD', 'ADI', 'MMC', 'BKNG', 'C', 'PGR', 'AMT', 'AXP', 'TJX',
        'ISRG', 'SYK', 'MO', 'EOG', 'CI', 'VRTX', 'TMUS', 'PLD', 'NOW', 'ADP',
        'DUK', 'BDX', 'ETN', 'CB', 'CME', 'ITW', 'ZTS', 'SO', 'CSX', 'APD'
    ]
    return sp100_tickers


def download_stock_data(tickers, start_date, end_date, save_dir='data'):
    """
    Download historical stock data for the given tickers.

    Parameters:
    -----------
    tickers : list
        List of stock tickers.
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.
    save_dir : str
        Directory to save the data.

    Returns:
    --------
    prices_df : pandas.DataFrame
        DataFrame with adjusted closing prices.
    """
    ensure_dir(save_dir)

    try:
        # Download S&P 100 index data
        print(f"Downloading S&P 100 index data...")
        sp100 = yf.download('^OEX', start=start_date, end=end_date, progress=False)

        if sp100.empty:
            raise ValueError(
                "Failed to download S&P 100 index data. Please check your internet connection and ticker symbol.")

        # Print columns for debugging
        print(f"S&P 100 data columns: {sp100.columns.tolist()}")

        # Download individual stock data
        print(f"Downloading data for {len(tickers)} S&P 100 stocks...")
        data = yf.download(tickers, start=start_date, end=end_date, progress=True)

        if data.empty:
            raise ValueError("Failed to download stock data.")

        # Print structure for debugging
        print(f"Data shape: {data.shape}")
        if isinstance(data.columns, pd.MultiIndex):
            print(f"Data column levels: {[list(level) for level in data.columns.levels]}")
        else:
            print(f"Data columns: {data.columns.tolist()}")

        # Extract price data based on the structure
        if isinstance(data.columns, pd.MultiIndex):
            # Handle multi-index columns (multiple stocks)
            price_col = 'Close'  # Default to Close
            if 'Adj Close' in data.columns.levels[0]:
                price_col = 'Adj Close'
                print("Using 'Adj Close' prices")
            else:
                print("Using 'Close' prices")

            prices_df = data[price_col]
        else:
            # Handle single stock case
            price_col = 'Close'  # Default to Close
            if 'Adj Close' in data.columns:
                price_col = 'Adj Close'
                print("Using 'Adj Close' prices")
            else:
                print("Using 'Close' prices")

            prices_df = pd.DataFrame(data[price_col])
            prices_df.columns = tickers

        # Add S&P 100 index to the DataFrame
        if 'Adj Close' in sp100.columns:
            prices_df['^OEX'] = sp100['Adj Close']
        else:
            prices_df['^OEX'] = sp100['Close']
            print("Using 'Close' for S&P 100 index")

        # Save data to CSV
        prices_df.to_csv(os.path.join(save_dir, 'sp100_prices.csv'))
        print(f"Price data saved to {os.path.join(save_dir, 'sp100_prices.csv')}")

        return prices_df

    except Exception as e:
        print(f"Error in download_stock_data: {str(e)}")
        if 'data' in locals():
            print(f"Data columns type: {type(data.columns)}")
            if isinstance(data.columns, pd.MultiIndex):
                print(f"MultiIndex levels: {[list(level) for level in data.columns.levels]}")
        raise

def calculate_returns(prices_df, save_dir='data'):
    """
    Calculate daily returns from price data.
    
    Parameters:
    -----------
    prices_df : pandas.DataFrame
        DataFrame with adjusted closing prices.
    save_dir : str
        Directory to save the calculated returns.
    
    Returns:
    --------
    returns_df : pandas.DataFrame
        DataFrame with daily returns.
    """
    ensure_dir(save_dir)
    
    # Calculate daily returns
    returns_df = prices_df.pct_change().dropna()
    
    # Save returns to CSV
    returns_df.to_csv(os.path.join(save_dir, 'sp100_returns.csv'))
    
    return returns_df

def split_data_into_periods(returns_df, save_dir='data'):
    """
    Split data into training and testing periods.
    
    Parameters:
    -----------
    returns_df : pandas.DataFrame
        DataFrame with daily returns.
    save_dir : str
        Directory to save the split data.
    
    Returns:
    --------
    data_periods : dict
        Dictionary with data for different periods.
    """
    ensure_dir(save_dir)
    
    # Calculate split date (use most recent year for testing)
    split_date = returns_df.index[-252]  # Approximately 1 year of trading days
    
    # Split data
    train_returns = returns_df.loc[:split_date]
    test_returns = returns_df.loc[split_date:]
    
    # Save train and test data to CSV
    train_returns.to_csv(os.path.join(save_dir, 'train_returns.csv'))
    test_returns.to_csv(os.path.join(save_dir, 'test_returns.csv'))
    
    # Split test data into quarterly periods
    dates = test_returns.index
    total_days = len(dates)
    q1_end_idx = total_days // 4
    q2_end_idx = total_days // 2
    q3_end_idx = 3 * total_days // 4
    q4_end_idx = total_days
    
    q1_returns = test_returns.iloc[:q1_end_idx]
    q2_returns = test_returns.iloc[:q2_end_idx]
    q3_returns = test_returns.iloc[:q3_end_idx]
    q4_returns = test_returns
    
    # Save quarterly data to CSV
    q1_returns.to_csv(os.path.join(save_dir, 'q1_returns.csv'))
    q2_returns.to_csv(os.path.join(save_dir, 'q2_returns.csv'))
    q3_returns.to_csv(os.path.join(save_dir, 'q3_returns.csv'))
    q4_returns.to_csv(os.path.join(save_dir, 'q4_returns.csv'))
    
    # Create a dictionary with all data
    data_periods = {
        'train_returns': train_returns,
        'test_returns': test_returns,
        'q1_returns': q1_returns,
        'q2_returns': q2_returns,
        'q3_returns': q3_returns,
        'q4_returns': q4_returns
    }
    
    return data_periods

def save_covariance_matrix(returns_df, save_dir='data'):
    """
    Calculate and save covariance matrix.
    
    Parameters:
    -----------
    returns_df : pandas.DataFrame
        DataFrame with daily returns.
    save_dir : str
        Directory to save the covariance matrix.
    
    Returns:
    --------
    cov_matrix : pandas.DataFrame
        Covariance matrix.
    """
    ensure_dir(save_dir)
    
    # Calculate covariance matrix
    cov_matrix = returns_df.cov()
    
    # Save covariance matrix to CSV
    cov_matrix.to_csv(os.path.join(save_dir, 'covariance_matrix.csv'))
    
    return cov_matrix

def plot_cumulative_returns(returns_df, title='Cumulative Returns', 
                            save_path='results/visualizations/cumulative_returns.png'):
    """
    Plot cumulative returns.
    
    Parameters:
    -----------
    returns_df : pandas.DataFrame
        DataFrame with daily returns.
    title : str
        Title of the plot.
    save_path : str
        Path to save the plot.
    """
    ensure_dir(os.path.dirname(save_path))
    
    # Calculate cumulative returns
    cumulative_returns = (1 + returns_df).cumprod() - 1
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot S&P 100 index
    if '^OEX' in cumulative_returns.columns:
        cumulative_returns['^OEX'].plot(label='S&P 100', linewidth=2, color='black')
        
        # Plot other stocks with lighter color
        other_columns = [col for col in cumulative_returns.columns if col != '^OEX']
        if other_columns:
            cumulative_returns[other_columns].plot(alpha=0.3)
    else:
        cumulative_returns.plot()
    
    # Add labels and title
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add legend if not too many stocks
    if len(returns_df.columns) <= 10:
        plt.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_correlation_heatmap(returns_df, n_stocks=30,
                             save_path='results/visualizations/correlation_heatmap.png'):
    """
    Plot correlation heatmap.

    Parameters:
    -----------
    returns_df : pandas.DataFrame
        DataFrame with daily returns.
    n_stocks : int
        Number of stocks to include in the heatmap.
    save_path : str
        Path to save the plot.
    """
    ensure_dir(os.path.dirname(save_path))

    try:
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()

        # Select top stocks by correlation with S&P 100
        if '^OEX' in corr_matrix.columns:
            top_stocks = corr_matrix['^OEX'].sort_values(ascending=False).index[:n_stocks].tolist()
            if '^OEX' not in top_stocks:
                top_stocks.append('^OEX')
        else:
            print("Warning: S&P 100 index (^OEX) not found in correlation matrix.")
            # Just use the first n_stocks
            top_stocks = corr_matrix.columns[:n_stocks].tolist()

        # Create figure
        plt.figure(figsize=(12, 10))

        # Get the subset of the correlation matrix for the selected stocks
        heatmap_data = corr_matrix.loc[top_stocks, top_stocks]

        # Plot heatmap
        sns.heatmap(heatmap_data, annot=True, cmap='coolwarm',
                    center=0, fmt='.2f', linewidths=0.5)

        # Add title
        plt.title(f'Correlation Heatmap of Top {n_stocks} Stocks and S&P 100')

        # Save figure
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        print(f"Correlation heatmap saved to: {save_path}")

    except Exception as e:
        print(f"Error in plot_correlation_heatmap: {str(e)}")
        print("Skipping correlation heatmap generation.")

def calculate_portfolio_statistics(portfolio, returns_df):
    """
    Calculate portfolio statistics.
    
    Parameters:
    -----------
    portfolio : dict
        Dictionary with stock tickers as keys and weights as values.
    returns_df : pandas.DataFrame
        DataFrame with daily returns.
    
    Returns:
    --------
    stats : dict
        Dictionary with portfolio statistics.
    """
    # Calculate portfolio returns
    portfolio_returns = pd.Series(0, index=returns_df.index)
    
    for stock, weight in portfolio.items():
        if stock in returns_df.columns:
            portfolio_returns += returns_df[stock] * weight
    
    # Extract S&P 100 returns
    if '^OEX' in returns_df.columns:
        sp100_returns = returns_df['^OEX']
    else:
        sp100_returns = None
    
    # Calculate statistics
    stats = {
        'mean_return': portfolio_returns.mean() * 252,  # Annualized return
        'std_return': portfolio_returns.std() * np.sqrt(252),  # Annualized volatility
    }
    
    if sp100_returns is not None:
        # Calculate correlation with S&P 100
        stats['correlation'] = portfolio_returns.corr(sp100_returns)
        
        # Calculate tracking error
        stats['tracking_error'] = np.sqrt(np.mean((portfolio_returns - sp100_returns) ** 2)) * np.sqrt(252)
        
        # Calculate beta
        stats['beta'] = portfolio_returns.cov(sp100_returns) / sp100_returns.var()
        
        # Calculate information ratio
        excess_return = portfolio_returns - sp100_returns
        stats['information_ratio'] = excess_return.mean() / excess_return.std() * np.sqrt(252)
    
    return stats
