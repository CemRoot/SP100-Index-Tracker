"""
AMPL data preparation script for the SP100-Index-Tracker project.

This script prepares data for the AMPL optimization model, creating
necessary data files in the format required by AMPL.
"""

import os
import pandas as pd
import numpy as np
import warnings
from utils import ensure_dir

# Ignore warnings
warnings.filterwarnings('ignore')

def prepare_ampl_data(train_returns_file='data/train_returns.csv', output_dir='ampl'):
    """
    Prepare data for AMPL optimization.
    
    Parameters:
    -----------
    train_returns_file : str
        Path to the CSV file with training returns.
    output_dir : str
        Directory to save AMPL data files.
    """
    ensure_dir(output_dir)
    
    print(f"Loading training returns from {train_returns_file}...")
    # Load training returns
    train_returns = pd.read_csv(train_returns_file, index_col=0, parse_dates=True)
    
    # Extract S&P 100 returns
    sp100_returns = train_returns['^OEX']
    stock_returns = train_returns.drop('^OEX', axis=1)
    
    print(f"Preparing AMPL data with {stock_returns.shape[1]} stocks and {len(stock_returns)} days...")
    
    # Calculate mean returns
    mean_returns = stock_returns.mean()
    mean_sp100 = sp100_returns.mean()
    
    # Calculate covariance matrix
    cov_matrix = stock_returns.cov()
    
    # Calculate covariance with S&P 100
    cov_with_sp100 = pd.Series(
        {stock: stock_returns[stock].cov(sp100_returns) for stock in stock_returns.columns}
    )
    
    # Calculate variance of S&P 100
    var_sp100 = sp100_returns.var()
    
    # Prepare AMPL data file
    ampl_data_file = os.path.join(output_dir, 'index_fund.dat')
    print(f"Creating AMPL data file: {ampl_data_file}...")
    
    with open(ampl_data_file, 'w') as f:
        # Write STOCKS set
        f.write('set STOCKS := ')
        for stock in stock_returns.columns:
            # Replace any dots, hyphens, etc. with underscores for AMPL compatibility
            ampl_stock = stock.replace('.', '_').replace('-', '_')
            f.write(f'"{ampl_stock}" ')
        f.write(';\n\n')
        
        # Write DAYS set
        f.write('set DAYS := ')
        for i, day in enumerate(stock_returns.index):
            f.write(f'"Day{i+1}" ')
        f.write(';\n\n')
        
        # Write q parameter
        f.write('param q := 20;  # Default value, will be changed in run file\n\n')
        
        # Write R parameter (stock returns)
        f.write('param R :\n')
        # Write header
        f.write('    ')
        for i in range(len(stock_returns.index)):
            f.write(f'"Day{i+1}" ')
        f.write(':=\n')
        
        # Write data
        for stock in stock_returns.columns:
            ampl_stock = stock.replace('.', '_').replace('-', '_')
            f.write(f'    "{ampl_stock}" ')
            for i, day in enumerate(stock_returns.index):
                f.write(f'{stock_returns.loc[day, stock]:.6f} ')
            f.write('\n')
        f.write(';\n\n')
        
        # Write R_SP100 parameter
        f.write('param R_SP100 :=\n')
        for i, day in enumerate(sp100_returns.index):
            f.write(f'    "Day{i+1}" {sp100_returns.loc[day]:.6f}\n')
        f.write(';\n\n')
        
        # Write mean_R parameter
        f.write('param mean_R :=\n')
        for stock in stock_returns.columns:
            ampl_stock = stock.replace('.', '_').replace('-', '_')
            f.write(f'    "{ampl_stock}" {mean_returns[stock]:.6f}\n')
        f.write(';\n\n')
        
        # Write mean_SP100 parameter
        f.write(f'param mean_SP100 := {mean_sp100:.6f};\n\n')
        
        # Write cov parameter (covariance matrix)
        f.write('param cov : ')
        for stock in stock_returns.columns:
            ampl_stock = stock.replace('.', '_').replace('-', '_')
            f.write(f'"{ampl_stock}" ')
        f.write(':=\n')
        
        for i, stock_i in enumerate(stock_returns.columns):
            ampl_stock_i = stock_i.replace('.', '_').replace('-', '_')
            f.write(f'    "{ampl_stock_i}" ')
            for stock_j in stock_returns.columns:
                f.write(f'{cov_matrix.loc[stock_i, stock_j]:.6f} ')
            f.write('\n')
        f.write(';\n\n')
        
        # Write cov_with_sp100 parameter
        f.write('param cov_with_sp100 :=\n')
        for stock in stock_returns.columns:
            ampl_stock = stock.replace('.', '_').replace('-', '_')
            f.write(f'    "{ampl_stock}" {cov_with_sp100[stock]:.6f}\n')
        f.write(';\n\n')
        
        # Write var_sp100 parameter
        f.write(f'param var_sp100 := {var_sp100:.6f};\n')
    
    print(f"AMPL data file created: {ampl_data_file}")
    
    # Copy the model and run files from the provided templates
    create_ampl_model_file(output_dir)
    create_ampl_run_file(output_dir)
    
    print("AMPL files preparation complete!")

def create_ampl_model_file(output_dir='ampl'):
    """
    Create the AMPL model file.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the AMPL model file.
    """
    model_file = os.path.join(output_dir, 'index_fund.mod')
    print(f"Creating AMPL model file: {model_file}...")
    
    with open(model_file, 'w') as f:
        f.write("""# AMPL Model File for S&P 100 Index Tracking

# Sets
set STOCKS;  # Set of all S&P 100 stocks
set DAYS;    # Set of days in the training period

# Parameters
param q;                        # Number of stocks to select
param R{STOCKS, DAYS};          # Daily returns for each stock
param R_SP100{DAYS};            # Daily returns for S&P 100 index
param mean_R{STOCKS};           # Mean return for each stock
param mean_SP100;               # Mean return for S&P 100
param cov{STOCKS, STOCKS};      # Covariance matrix of stock returns
param cov_with_sp100{STOCKS};   # Covariance of each stock with S&P 100
param var_sp100;                # Variance of S&P 100 returns

# Decision Variables
var x{STOCKS} binary;           # Stock selection variables (1 if selected, 0 otherwise)
var w{STOCKS} >= 0;             # Weight allocation for each stock

# Variance of the tracking error (to be minimized)
var tracking_error_var;

# Objective Function
# Minimize tracking error variance
minimize obj: tracking_error_var;

# Constraint: Total number of stocks selected equals q
subject to num_stocks: sum{i in STOCKS} x[i] = q;

# Constraint: Sum of weights equals 1
subject to sum_weights: sum{i in STOCKS} w[i] = 1;

# Constraint: Weight can be positive only if stock is selected
subject to weight_selection{i in STOCKS}: w[i] <= x[i];

# Constraint: Define tracking error variance
subject to tracking_def:
    tracking_error_var = 
    sum{i in STOCKS, j in STOCKS} w[i] * w[j] * cov[i,j] -
    2 * sum{i in STOCKS} w[i] * cov_with_sp100[i] +
    var_sp100;
""")
    
    print(f"AMPL model file created: {model_file}")

def create_ampl_run_file(output_dir='ampl'):
    """
    Create the AMPL run file.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the AMPL run file.
    """
    run_file = os.path.join(output_dir, 'index_fund.run')
    print(f"Creating AMPL run file: {run_file}...")
    
    with open(run_file, 'w') as f:
        f.write("""# AMPL Run File for S&P 100 Index Tracking

# Reset AMPL
reset;

# Load the model
model index_fund.mod;

# Load the data
data index_fund.dat;

# Set the solver (change to your available solver)
option solver cplex;

# Create a parameter for different q values to test
param q_values{1..5};
let q_values[1] := 10;
let q_values[2] := 15;
let q_values[3] := 20;
let q_values[4] := 25;
let q_values[5] := 30;

# Create parameters to store results
param tracking_error{1..5};
param selected_stocks{1..5, 1..100} default 0;  # Stores which stocks are selected for each q
param weights{1..5, 1..100} default 0;          # Stores weights for each q

# Solve for each value of q
for {k in 1..5} {
    let q := q_values[k];
    solve;
    
    # Store tracking error
    let tracking_error[k] := sqrt(tracking_error_var);
    
    # Store selected stocks and weights
    for {i in STOCKS} {
        if x[i] > 0.5 then {
            let selected_stocks[k, ord(i)] := 1;
            let weights[k, ord(i)] := w[i];
        }
    }
    
    # Display results
    printf "\\n----- Results for q = %d -----\\n", q;
    printf "Tracking Error: %f\\n", sqrt(tracking_error_var);
    printf "Selected Stocks and Weights:\\n";
    for {i in STOCKS: x[i] > 0.5} {
        printf "%s: %f\\n", i, w[i];
    }
}

# Save results to a file
printf "q,tracking_error\\n" > "tracking_error_results.csv";
for {k in 1..5} {
    printf "%d,%f\\n", q_values[k], tracking_error[k] >> "tracking_error_results.csv";
}

# Save selected stocks and weights
printf "q,stock,weight\\n" > "portfolio_results.csv";
for {k in 1..5} {
    for {i in STOCKS: selected_stocks[k,ord(i)] > 0} {
        printf "%d,%s,%f\\n", q_values[k], i, weights[k,ord(i)] >> "portfolio_results.csv";
    }
}

# Display summary
printf "\\n----- Summary -----\\n";
printf "q\\tTracking Error\\n";
for {k in 1..5} {
    printf "%d\\t%f\\n", q_values[k], tracking_error[k];
}

# End of AMPL script
""")
    
    print(f"AMPL run file created: {run_file}")

def main():
    print("Starting AMPL data preparation...")
    prepare_ampl_data(train_returns_file='data/train_returns.csv', output_dir='ampl')
    print("\nAMPL data preparation complete!")
    print("Next steps:")
    print("1. Open AMPL")
    print("2. Navigate to the 'ampl' directory")
    print("3. Run the model with: include index_fund.run")
    print("4. Check the results in tracking_error_results.csv and portfolio_results.csv")

if __name__ == "__main__":
    main()
