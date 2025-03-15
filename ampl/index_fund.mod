# AMPL Model File for S&P 100 Index Tracking

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
