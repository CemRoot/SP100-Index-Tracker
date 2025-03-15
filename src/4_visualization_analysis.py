"""
Visualization and analysis script for the SP100-Index-Tracker project.

This script compares the AMPL and Genetic Algorithm approaches, 
generating visualizations and analysis for the report.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import warnings
from utils import ensure_dir, calculate_portfolio_statistics
import shutil

# Ignore warnings
warnings.filterwarnings('ignore')

def plot_stock_selection_overlap(ampl_results_file='results/ampl/portfolio_results.csv', 
                                ga_results_file='results/ga/ga_portfolio_results.csv',
                                output_dir='results/visualizations'):
    """
    Analyze and visualize the overlap in stock selection between AMPL and GA approaches.
    
    Parameters:
    -----------
    ampl_results_file : str
        Path to the CSV file with AMPL results.
    ga_results_file : str
        Path to the CSV file with GA results.
    output_dir : str
        Directory to save visualizations.
    
    Returns:
    --------
    overlap_data : list
        List of dictionaries with overlap data for each q value.
    """
    ensure_dir(output_dir)
    
    print("Analyzing stock selection overlap between AMPL and GA approaches...")
    
    # Load results
    try:
        ampl_results = pd.read_csv(ampl_results_file)
        ga_results = pd.read_csv(ga_results_file)
    except FileNotFoundError:
        print(f"Error: Could not find results files. Make sure both approaches have been run.")
        return []
    
    # Get stocks selected for each q value
    q_values = sorted(ampl_results['q'].unique())
    overlap_data = []
    all_common_stocks = set()
    always_common_stocks = set()
    first_q = True
    
    for q in q_values:
        ampl_stocks = set(ampl_results[ampl_results['q'] == q]['stock'])
        ga_stocks = set(ga_results[ga_results['q'] == q]['stock'])
        
        common_stocks = ampl_stocks.intersection(ga_stocks)
        all_common_stocks.update(common_stocks)
        
        if first_q:
            always_common_stocks = common_stocks
            first_q = False
        else:
            always_common_stocks = always_common_stocks.intersection(common_stocks)
        
        overlap_pct = len(common_stocks) / q * 100
        
        overlap_data.append({
            'q': q,
            'overlap_count': len(common_stocks),
            'overlap_percentage': overlap_pct,
            'common_stocks': ', '.join(sorted(common_stocks))
        })
        
        print(f"\nq = {q}: {len(common_stocks)}/{q} stocks overlap ({overlap_pct:.1f}%)")
        print(f"Common stocks: {', '.join(sorted(common_stocks))}")
    
    # Save overlap data
    pd.DataFrame(overlap_data).to_csv(f"{output_dir}/stock_selection_overlap.csv", index=False)
    
    # Plot overlap percentage
    plt.figure(figsize=(10, 6))
    q_values = [data['q'] for data in overlap_data]
    overlap_pcts = [data['overlap_percentage'] for data in overlap_data]
    
    plt.bar(q_values, overlap_pcts, color='cornflowerblue')
    plt.xlabel('Number of stocks (q)')
    plt.ylabel('Overlap Percentage (%)')
    plt.title('Stock Selection Overlap Between AMPL and GA Approaches')
    plt.xticks(q_values)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, pct in enumerate(overlap_pcts):
        plt.text(q_values[i], pct + 1, f"{pct:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/stock_selection_overlap.png", dpi=300)
    
    print(f"\nNumber of unique stocks in the overlap across all q values: {len(all_common_stocks)}")
    if always_common_stocks:
        print(f"Stocks selected by both approaches for all q values: {', '.join(sorted(always_common_stocks))}")
    
    return overlap_data

def analyze_weights_distribution(ampl_results_file='results/ampl/portfolio_results.csv',
                                ga_results_file='results/ga/ga_portfolio_results.csv',
                                q_value=20,
                                output_dir='results/visualizations'):
    """
    Analyze and visualize the weight distribution in portfolios.
    
    Parameters:
    -----------
    ampl_results_file : str
        Path to the CSV file with AMPL results.
    ga_results_file : str
        Path to the CSV file with GA results.
    q_value : int
        The specific q value to analyze.
    output_dir : str
        Directory to save visualizations.
    
    Returns:
    --------
    weight_data : dict
        Dictionary with weight concentration data.
    """
    ensure_dir(output_dir)
    
    print("\nAnalyzing weight distribution in portfolios...")
    
    # Load results
    try:
        ampl_results = pd.read_csv(ampl_results_file)
        ga_results = pd.read_csv(ga_results_file)
    except FileNotFoundError:
        print(f"Error: Could not find results files. Make sure both approaches have been run.")
        return {}
    
    # Get all q values
    q_values = sorted(ampl_results['q'].unique())
    weight_data = []
    
    # Calculate weight concentration for all q values
    for q in q_values:
        ampl_q_data = ampl_results[ampl_results['q'] == q].sort_values('weight', ascending=False)
        ga_q_data = ga_results[ga_results['q'] == q].sort_values('weight', ascending=False)
        
        ampl_top5_concentration = ampl_q_data.head(5)['weight'].sum() * 100
        ga_top5_concentration = ga_q_data.head(5)['weight'].sum() * 100
        
        weight_data.append({
            'q': q,
            'ampl_top5_concentration': ampl_top5_concentration,
            'ga_top5_concentration': ga_top5_concentration
        })
    
    # Save weight concentration data
    weight_df = pd.DataFrame(weight_data)
    weight_df.to_csv(f"{output_dir}/weight_concentration.csv", index=False)
    
    # Plot weight concentration
    plt.figure(figsize=(10, 6))
    plt.plot(weight_df['q'], weight_df['ampl_top5_concentration'], 'o-', label='AMPL', color='indianred')
    plt.plot(weight_df['q'], weight_df['ga_top5_concentration'], 'o-', label='GA', color='teal')
    plt.xlabel('Number of stocks (q)')
    plt.ylabel('Top 5 Stocks Weight Concentration (%)')
    plt.title('Portfolio Concentration: Top 5 Stocks')
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(q_values)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/weight_concentration.png", dpi=300)
    
    # Print weight concentration
    print("\nWeight Concentration Analysis:")
    print("==============================")
    print("Top 5 stocks weight concentration (%):")
    print(" q  ampl_top5_concentration  ga_top5_concentration")
    for row in weight_data:
        print(f"{row['q']:<3} {row['ampl_top5_concentration']:<24} {row['ga_top5_concentration']:<20}")
    
    # Plot weight distribution for the specific q value
    ampl_q_data = ampl_results[ampl_results['q'] == q_value].sort_values('weight', ascending=False)
    ga_q_data = ga_results[ga_results['q'] == q_value].sort_values('weight', ascending=False)
    
    plt.figure(figsize=(12, 6))
    
    # AMPL weights
    plt.subplot(1, 2, 1)
    plt.bar(range(len(ampl_q_data)), ampl_q_data['weight'], color='indianred')
    plt.title(f'AMPL Portfolio Weights (q={q_value})')
    plt.xlabel('Stock Rank (by weight)')
    plt.ylabel('Weight')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # GA weights
    plt.subplot(1, 2, 2)
    plt.bar(range(len(ga_q_data)), ga_q_data['weight'], color='teal')
    plt.title(f'GA Portfolio Weights (q={q_value})')
    plt.xlabel('Stock Rank (by weight)')
    plt.ylabel('Weight')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/weight_distribution_q{q_value}.png", dpi=300)
    
    return weight_df.to_dict('records')

def plot_performance_across_time(ampl_results_file='results/ampl/portfolio_results.csv',
                                ga_results_file='results/ga/ga_portfolio_results.csv',
                                q_value=20,
                                test_returns_file='data/test_returns.csv',
                                output_dir='results/visualizations'):
    """
    Analyze and visualize portfolio performance across time.
    
    Parameters:
    -----------
    ampl_results_file : str
        Path to the CSV file with AMPL results.
    ga_results_file : str
        Path to the CSV file with GA results.
    q_value : int
        The specific q value to analyze.
    test_returns_file : str
        Path to the test returns file.
    output_dir : str
        Directory to save visualizations.
    
    Returns:
    --------
    performance_metrics : dict
        Dictionary with performance metrics.
    """
    ensure_dir(output_dir)
    
    print("\nAnalyzing portfolio performance across time...")
    
    # Load results
    try:
        ampl_results = pd.read_csv(ampl_results_file)
        ga_results = pd.read_csv(ga_results_file)
        test_returns = pd.read_csv(test_returns_file, index_col=0, parse_dates=True)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure both approaches have been run and test returns file exists.")
        return {}
    
    # Get portfolio weights for the specified q
    ampl_portfolio = ampl_results[ampl_results['q'] == q_value][['stock', 'weight']]
    ga_portfolio = ga_results[ga_results['q'] == q_value][['stock', 'weight']]
    
    # Convert to dictionary for easier calculation
    ampl_weights = dict(zip(ampl_portfolio['stock'], ampl_portfolio['weight']))
    ga_weights = dict(zip(ga_portfolio['stock'], ga_portfolio['weight']))
    
    # Calculate portfolio returns
    ampl_returns = pd.Series(0, index=test_returns.index)
    ga_returns = pd.Series(0, index=test_returns.index)
    
    for stock, weight in ampl_weights.items():
        if stock in test_returns.columns:
            ampl_returns += test_returns[stock] * weight
    
    for stock, weight in ga_weights.items():
        if stock in test_returns.columns:
            ga_returns += test_returns[stock] * weight
    
    # Calculate S&P 100 returns
    sp100_returns = test_returns['^OEX']
    
    # Calculate cumulative returns
    ampl_cum_returns = (1 + ampl_returns).cumprod() - 1
    ga_cum_returns = (1 + ga_returns).cumprod() - 1
    sp100_cum_returns = (1 + sp100_returns).cumprod() - 1
    
    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    plt.plot(ampl_cum_returns.index, ampl_cum_returns * 100, label='AMPL Portfolio', color='indianred')
    plt.plot(ga_cum_returns.index, ga_cum_returns * 100, label='GA Portfolio', color='teal')
    plt.plot(sp100_cum_returns.index, sp100_cum_returns * 100, label='S&P 100', color='navy')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.title(f'Portfolio Performance Comparison (q={q_value})')
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend()
    
    # Format x-axis date ticks
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cumulative_returns_over_time.png", dpi=300)
    
    # Calculate performance metrics
    trading_days = 252
    
    ampl_metrics = {
        'final_return': ampl_cum_returns.iloc[-1] * 100,
        'annualized_return': ampl_returns.mean() * trading_days * 100,
        'annualized_volatility': ampl_returns.std() * np.sqrt(trading_days) * 100,
        'sharpe_ratio': (ampl_returns.mean() / ampl_returns.std()) * np.sqrt(trading_days),
        'correlation_with_sp100': ampl_returns.corr(sp100_returns),
        'tracking_error': (ampl_returns - sp100_returns).std() * np.sqrt(trading_days),
        'relative_return': (ampl_cum_returns.iloc[-1] - sp100_cum_returns.iloc[-1]) * 100
    }
    
    ga_metrics = {
        'final_return': ga_cum_returns.iloc[-1] * 100,
        'annualized_return': ga_returns.mean() * trading_days * 100,
        'annualized_volatility': ga_returns.std() * np.sqrt(trading_days) * 100,
        'sharpe_ratio': (ga_returns.mean() / ga_returns.std()) * np.sqrt(trading_days),
        'correlation_with_sp100': ga_returns.corr(sp100_returns),
        'tracking_error': (ga_returns - sp100_returns).std() * np.sqrt(trading_days),
        'relative_return': (ga_cum_returns.iloc[-1] - sp100_cum_returns.iloc[-1]) * 100
    }
    
    sp100_metrics = {
        'final_return': sp100_cum_returns.iloc[-1] * 100,
        'annualized_return': sp100_returns.mean() * trading_days * 100,
        'annualized_volatility': sp100_returns.std() * np.sqrt(trading_days) * 100,
        'sharpe_ratio': (sp100_returns.mean() / sp100_returns.std()) * np.sqrt(trading_days)
    }
    
    # Print performance metrics
    print(f"\nPerformance Analysis for q={q_value}:")
    print("====================================")
    print("AMPL Portfolio:")
    print(f"  Final Cumulative Return: {ampl_metrics['final_return']:.2f}%")
    print(f"  Correlation with S&P 100: {ampl_metrics['correlation_with_sp100']:.4f}")
    print(f"  Tracking Error (Annualized): {ampl_metrics['tracking_error']:.4f}")
    print(f"  Return Relative to S&P 100: {ampl_metrics['relative_return']:.2f}%")
    print()
    print("GA Portfolio:")
    print(f"  Final Cumulative Return: {ga_metrics['final_return']:.2f}%")
    print(f"  Correlation with S&P 100: {ga_metrics['correlation_with_sp100']:.4f}")
    print(f"  Tracking Error (Annualized): {ga_metrics['tracking_error']:.4f}")
    print(f"  Return Relative to S&P 100: {ga_metrics['relative_return']:.2f}%")
    print()
    print("S&P 100:")
    print(f"  Final Cumulative Return: {sp100_metrics['final_return']:.2f}%")
    
    # Plot rolling correlation
    window = 30  # 30-day rolling window
    rolling_corr_ampl = ampl_returns.rolling(window=window).corr(sp100_returns)
    rolling_corr_ga = ga_returns.rolling(window=window).corr(sp100_returns)
    
    plt.figure(figsize=(12, 6))
    plt.plot(rolling_corr_ampl.index, rolling_corr_ampl, label='AMPL Portfolio', color='indianred')
    plt.plot(rolling_corr_ga.index, rolling_corr_ga, label='GA Portfolio', color='teal')
    plt.xlabel('Date')
    plt.ylabel('Correlation with S&P 100')
    plt.title(f'Rolling {window}-Day Correlation with S&P 100 (q={q_value})')
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend()
    
    # Format x-axis date ticks
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rolling_correlation.png", dpi=300)
    
    # Save performance metrics
    performance_metrics = {
        'ampl': ampl_metrics,
        'ga': ga_metrics,
        'sp100': sp100_metrics
    }
    
    pd.DataFrame(performance_metrics).to_csv(f"{output_dir}/performance_metrics_q{q_value}.csv")
    
    return performance_metrics

def compare_performance_across_q(ampl_results_file='results/ampl/portfolio_results.csv',
                                 ga_results_file='results/ga/ga_portfolio_results.csv',
                                 test_returns_files={
                                     'q1': 'data/q1_returns.csv',
                                     'q2': 'data/q2_returns.csv',
                                     'q3': 'data/q3_returns.csv',
                                     'q4': 'data/q4_returns.csv'
                                 },
                                 output_dir='results/visualizations'):
    """
    Compare performance across different q values and time periods.
    
    Parameters:
    -----------
    ampl_results_file : str
        Path to the CSV file with AMPL results.
    ga_results_file : str
        Path to the CSV file with GA results.
    test_returns_files : dict
        Dictionary mapping period name to returns file path.
    output_dir : str
        Directory to save visualizations.
    
    Returns:
    --------
    comparison_data : pd.DataFrame
        DataFrame with performance comparison data.
    """
    ensure_dir(output_dir)
    
    print("\nComparing performance across different q values and time periods...")
    
    # Load results
    try:
        ampl_results = pd.read_csv(ampl_results_file)
        ga_results = pd.read_csv(ga_results_file)
    except FileNotFoundError:
        print(f"Error: Could not find portfolio results files.")
        return None
    
    # Get q values
    q_values = sorted(ampl_results['q'].unique())
    
    # Initialize results dictionary
    comparison_data = []
    
    # Loop through each period
    best_ampl_by_period = {}
    best_ga_by_period = {}
    
    for period, returns_file in test_returns_files.items():
        try:
            test_returns = pd.read_csv(returns_file, index_col=0, parse_dates=True)
            print(f"\n{period.upper()} Period:")
        except FileNotFoundError:
            print(f"Warning: Could not find returns file for {period} period: {returns_file}")
            continue
        
        # Get S&P 100 returns
        sp100_returns = test_returns['^OEX']
        
        # For each q value
        best_ampl_te = float('inf')
        best_ga_te = float('inf')
        best_ampl_q = None
        best_ga_q = None
        
        for q in q_values:
            # Get AMPL portfolio
            ampl_portfolio = ampl_results[ampl_results['q'] == q][['stock', 'weight']]
            ampl_weights = dict(zip(ampl_portfolio['stock'], ampl_portfolio['weight']))
            
            # Get GA portfolio
            ga_portfolio = ga_results[ga_results['q'] == q][['stock', 'weight']]
            ga_weights = dict(zip(ga_portfolio['stock'], ga_portfolio['weight']))
            
            # Calculate portfolio returns
            ampl_returns = pd.Series(0, index=test_returns.index)
            ga_returns = pd.Series(0, index=test_returns.index)
            
            for stock, weight in ampl_weights.items():
                if stock in test_returns.columns:
                    ampl_returns += test_returns[stock] * weight
            
            for stock, weight in ga_weights.items():
                if stock in test_returns.columns:
                    ga_returns += test_returns[stock] * weight
            
            # Calculate performance metrics
            trading_days = 252
            
            ampl_cum_return = (1 + ampl_returns).cumprod().iloc[-1] - 1
            ga_cum_return = (1 + ga_returns).cumprod().iloc[-1] - 1
            sp100_cum_return = (1 + sp100_returns).cumprod().iloc[-1] - 1
            
            ampl_corr = ampl_returns.corr(sp100_returns)
            ga_corr = ga_returns.corr(sp100_returns)
            
            ampl_te = (ampl_returns - sp100_returns).std() * np.sqrt(trading_days)
            ga_te = (ga_returns - sp100_returns).std() * np.sqrt(trading_days)
            
            # Track best q value for each approach
            if ampl_te < best_ampl_te:
                best_ampl_te = ampl_te
                best_ampl_q = q
            
            if ga_te < best_ga_te:
                best_ga_te = ga_te
                best_ga_q = q
            
            # Add to comparison data
            comparison_data.append({
                'period': period,
                'q': q,
                'approach': 'AMPL',
                'cum_return': ampl_cum_return * 100,
                'correlation': ampl_corr,
                'tracking_error': ampl_te
            })
            
            comparison_data.append({
                'period': period,
                'q': q,
                'approach': 'GA',
                'cum_return': ga_cum_return * 100,
                'correlation': ga_corr,
                'tracking_error': ga_te
            })
        
        # Store best q values for the period
        best_ampl_by_period[period] = (best_ampl_q, best_ampl_te)
        best_ga_by_period[period] = (best_ga_q, best_ga_te)
        
        print(f"  Best AMPL q value: {best_ampl_q} (Tracking Error: {best_ampl_te:.4f})")
        print(f"  Best GA q value: {best_ga_q} (Tracking Error: {best_ga_te:.4f})")
    
    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(f"{output_dir}/performance_comparison.csv", index=False)
    
    # Plot tracking error by period
    plt.figure(figsize=(14, 10))
    periods = sorted(comparison_df['period'].unique())
    num_periods = len(periods)
    
    for i, period in enumerate(periods):
        plt.subplot(2, 2, i+1)
        period_data = comparison_df[comparison_df['period'] == period]
        
        for approach in ['AMPL', 'GA']:
            approach_data = period_data[period_data['approach'] == approach]
            plt.plot(approach_data['q'], approach_data['tracking_error'], 'o-', 
                     label=approach, 
                     color='indianred' if approach == 'AMPL' else 'teal')
        
        plt.xlabel('Number of stocks (q)')
        plt.ylabel('Tracking Error')
        plt.title(f'{period.upper()} Period: Tracking Error by Portfolio Size')
        plt.grid(linestyle='--', alpha=0.7)
        plt.legend()
        plt.xticks(q_values)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tracking_error_by_period.png", dpi=300)
    
    # Plot correlation by period
    plt.figure(figsize=(14, 10))
    
    for i, period in enumerate(periods):
        plt.subplot(2, 2, i+1)
        period_data = comparison_df[comparison_df['period'] == period]
        
        for approach in ['AMPL', 'GA']:
            approach_data = period_data[period_data['approach'] == approach]
            plt.plot(approach_data['q'], approach_data['correlation'], 'o-', 
                     label=approach, 
                     color='indianred' if approach == 'AMPL' else 'teal')
        
        plt.xlabel('Number of stocks (q)')
        plt.ylabel('Correlation with S&P 100')
        plt.title(f'{period.upper()} Period: Correlation by Portfolio Size')
        plt.grid(linestyle='--', alpha=0.7)
        plt.legend()
        plt.xticks(q_values)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_by_period.png", dpi=300)
    
    # Print summary
    print("\nPerformance Comparison by Period:")
    print("================================")
    for period in periods:
        print(f"\n{period.upper()} Period:")
        print(f"  Best AMPL q value: {best_ampl_by_period[period][0]} (Tracking Error: {best_ampl_by_period[period][1]:.4f})")
        print(f"  Best GA q value: {best_ga_by_period[period][0]} (Tracking Error: {best_ga_by_period[period][1]:.4f})")
    
    return comparison_df

def main():
    """Main execution function."""
    print("Starting visualization and analysis...")
    
    # Projenin kök dizinini güvenilir bir şekilde belirle
    try:
        # Önce __file__ kullanarak deneyelim
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        print(f"Project root: {project_root}")
    except:
        # Alternatif olarak geçerli çalışma dizinini kullan
        project_root = os.getcwd()
        # Eğer src dizinindeyiz, bir üst dizine çık
        if os.path.basename(project_root) == 'src':
            project_root = os.path.dirname(project_root)
        print(f"Project root (from cwd): {project_root}")
    
    # Gerekli dizinlerin oluşturulması
    results_dir = os.path.join(project_root, 'results')
    visualizations_dir = os.path.join(results_dir, 'visualizations')
    ensure_dir(visualizations_dir)
    
    # Mutlak dosya yollarını belirle (Windows ve Unix uyumlu)
    ampl_dir = os.path.join(project_root, 'results', 'ampl')
    ga_dir = os.path.join(project_root, 'results', 'ga')
    ensure_dir(ampl_dir)
    ensure_dir(ga_dir)
    
    ampl_results_file = os.path.join(ampl_dir, 'portfolio_results.csv')
    ga_results_file = os.path.join(ga_dir, 'ga_portfolio_results.csv')
    
    # Sonuç dosyalarını alternatif konumlarda ara
    if not os.path.exists(ampl_results_file):
        print(f"Looking for AMPL results in alternative locations...")
        
        # Ana dizinde ara
        root_ampl = os.path.join(project_root, 'portfolio_results.csv')
        if os.path.exists(root_ampl):
            print(f"Found at root: {root_ampl}")
            # Dosyayı beklenen konuma kopyala
            shutil.copy(root_ampl, ampl_results_file)
            print(f"Copied to expected location: {ampl_results_file}")
        # AMPL dizininde ara
        elif os.path.exists(os.path.join(project_root, 'ampl', 'portfolio_results.csv')):
            ampl_alt = os.path.join(project_root, 'ampl', 'portfolio_results.csv')
            print(f"Found in ampl dir: {ampl_alt}")
            # Dosyayı beklenen konuma kopyala
            shutil.copy(ampl_alt, ampl_results_file)
            print(f"Copied to expected location: {ampl_results_file}")
        else:
            print("No AMPL results found. Some analysis will be skipped.")
    else:
        print(f"Found AMPL results: {ampl_results_file}")
    
    if not os.path.exists(ga_results_file):
        print(f"Looking for GA results in alternative locations...")
        
        # Ana dizinde ara
        root_ga = os.path.join(project_root, 'ga_portfolio_results.csv')
        if os.path.exists(root_ga):
            print(f"Found at root: {root_ga}")
            # Dosyayı beklenen konuma kopyala
            shutil.copy(root_ga, ga_results_file)
            print(f"Copied to expected location: {ga_results_file}")
        else:
            print("No GA results found. Some analysis will be skipped.")
    else:
        print(f"Found GA results: {ga_results_file}")
    
    # Data dizini oluştur (eğer yoksa)
    data_dir = os.path.join(project_root, 'data')
    ensure_dir(data_dir)
    
    # Test dosyalarını mutlak yollar ile tanımla
    test_returns_files = {
        'q1': os.path.join(data_dir, 'q1_returns.csv'),
        'q2': os.path.join(data_dir, 'q2_returns.csv'),
        'q3': os.path.join(data_dir, 'q3_returns.csv'),
        'q4': os.path.join(data_dir, 'q4_returns.csv')
    }
    test_returns_file = os.path.join(data_dir, 'test_returns.csv')
    
    # Dosyalar varsa analizi çalıştır
    if os.path.exists(ampl_results_file) and os.path.exists(ga_results_file):
        # Analiz fonksiyonlarını çalıştır
        plot_stock_selection_overlap(ampl_results_file, ga_results_file, visualizations_dir)
        analyze_weights_distribution(ampl_results_file, ga_results_file, q_value=20, output_dir=visualizations_dir)
        
        if os.path.exists(test_returns_file):
            plot_performance_across_time(
                ampl_results_file, 
                ga_results_file,
                q_value=20, 
                test_returns_file=test_returns_file,
                output_dir=visualizations_dir
            )
        
        # Dönem bazlı dönüş dosyalarının var olup olmadığını kontrol et
        missing_files = []
        for period, file_path in test_returns_files.items():
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            print(f"Warning: The following returns files are missing: {', '.join(missing_files)}")
            print("Some period-specific analysis will be skipped.")
        else:
            compare_performance_across_q(
                ampl_results_file,
                ga_results_file,
                test_returns_files,
                output_dir=visualizations_dir
            )
        
        print("\nAnalysis complete. All visualizations have been saved to results/visualizations directory.")
    else:
        print("\nSkipping analysis steps due to missing files.")
        print("Run both AMPL optimization and alternative approach first to generate full analysis.")

if __name__ == "__main__":
    main() 