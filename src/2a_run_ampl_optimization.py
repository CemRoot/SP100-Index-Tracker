"""
Faster AMPL Optimization script using the amplpy Python API.

This script runs the AMPL optimization model with time limits and other
performance options to get results more quickly.
"""

import os
import pandas as pd
import numpy as np
import time
from amplpy import AMPL, Environment
from utils import ensure_dir

def run_ampl_optimization(model_file='ampl/index_fund.mod',
                         data_file='ampl/index_fund.dat',
                         output_dir='results/ampl',
                         time_limit=300):  # 5 minute time limit per solve
    """
    Run AMPL optimization using the amplpy Python API with performance options.

    Parameters:
    -----------
    model_file : str
        Path to the AMPL model file.
    data_file : str
        Path to the AMPL data file.
    output_dir : str
        Directory to save optimization results.
    time_limit : int
        Time limit in seconds for each optimization.
    """
    # Get absolute paths relative to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_file_abs = os.path.join(project_root, model_file)
    data_file_abs = os.path.join(project_root, data_file)
    output_dir_abs = os.path.join(project_root, output_dir)

    # Check if files exist
    if not os.path.exists(model_file_abs):
        raise FileNotFoundError(f"Model file not found: {model_file_abs}")
    if not os.path.exists(data_file_abs):
        raise FileNotFoundError(f"Data file not found: {data_file_abs}")

    # Ensure output directory exists
    ensure_dir(output_dir_abs)

    print(f"Starting faster AMPL optimization using amplpy...")
    print(f"Project root: {project_root}")
    print(f"Model file: {model_file_abs}")
    print(f"Data file: {data_file_abs}")
    print(f"Output directory: {output_dir_abs}")
    print(f"Time limit per solve: {time_limit} seconds")

    try:
        # Initialize AMPL environment
        print("Initializing AMPL environment...")
        ampl = AMPL()

        # Load model and data
        print("Loading model and data...")
        ampl.read(model_file_abs)
        ampl.readData(data_file_abs)

        # Set solver and options for faster execution
        ampl.setOption('solver', 'highs')
        ampl.setOption('highs_options', f'time_limit={time_limit} mip_rel_gap=0.05')

        # Create arrays to store results
        q_values = [10, 15, 20, 25, 30]
        tracking_errors = []
        all_portfolios = []

        # Run optimization for each q value
        for q in q_values:
            start_time = time.time()
            print(f"\nSolving for q = {q}...")

            # Set q parameter
            ampl.getParameter('q').set(q)

            # Solve the model
            ampl.solve()

            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            print(f"Solution time: {elapsed_time:.2f} seconds")

            # Check solution status
            solve_result = ampl.getValue("solve_result")
            print(f"Solution status: {solve_result}")

            # Get objective value (tracking error variance)
            obj_value = ampl.getObjective('obj').value()
            tracking_error = np.sqrt(obj_value)
            tracking_errors.append(tracking_error)

            print(f"Tracking error: {tracking_error:.6f}")

            # Get selected stocks and weights
            x = ampl.getVariable('x')  # Selection variables
            w = ampl.getVariable('w')  # Weight variables

            # Get the set of stocks
            stocks = list(ampl.getSet('STOCKS'))

            # Create portfolio for this q value
            portfolio = []

            selected_count = 0
            print("Selected stocks and weights:")
            for stock in stocks:
                x_val = x[stock].value()
                w_val = w[stock].value()

                if x_val > 0.5:  # Stock is selected
                    selected_count += 1
                    print(f"  {stock}: {w_val:.6f}")

                    # Convert stock name back to original format (replace underscores)
                    original_stock = stock.replace('_', '.').replace('\"', '')

                    # Add to portfolio
                    portfolio.append({
                        'q': q,
                        'stock': original_stock,
                        'weight': w_val
                    })

            print(f"Number of selected stocks: {selected_count}")

            # Add portfolios to results
            all_portfolios.extend(portfolio)

        # Save tracking error results
        tracking_error_df = pd.DataFrame({
            'q': q_values,
            'tracking_error': tracking_errors
        })
        tracking_error_df.to_csv(os.path.join(output_dir_abs, 'tracking_error_results.csv'), index=False)
        print(f"\nTracking error results saved to {os.path.join(output_dir_abs, 'tracking_error_results.csv')}")

        # Save portfolio results
        portfolio_df = pd.DataFrame(all_portfolios)
        portfolio_df.to_csv(os.path.join(output_dir_abs, 'portfolio_results.csv'), index=False)
        print(f"Portfolio results saved to {os.path.join(output_dir_abs, 'portfolio_results.csv')}")

        # Print summary
        print("\n----- Summary -----")
        print("q\tTracking Error")
        for q, te in zip(q_values, tracking_errors):
            print(f"{q}\t{te:.6f}")

        return {
            'q_values': q_values,
            'tracking_errors': tracking_errors,
            'portfolios': all_portfolios
        }

    except Exception as e:
        print(f"Error in AMPL optimization: {str(e)}")
        print("Make sure amplpy is installed and your license is active.")

        # List directory contents for debugging
        print("\nProject directory contents:")
        for root, dirs, files in os.walk(project_root, topdown=True, maxdepth=2):
            level = root.replace(project_root, '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            sub_indent = ' ' * 4 * (level + 1)
            for f in files:
                print(f"{sub_indent}{f}")

        raise

def main():
    print("Starting faster AMPL optimization via Python API...")

    # Run AMPL optimization with a time limit
    results = run_ampl_optimization(
        model_file='ampl/index_fund.mod',
        data_file='ampl/index_fund.dat',
        output_dir='results/ampl',
        time_limit=180  # 3 minutes per solve
    )

    print("\nAMPL optimization complete!")
    print("Next, run the alternative approach with:")
    print("  python src/3_alternative_approach.py")
    print("Then run the visualization and analysis with:")
    print("  python src/4_visualization_analysis.py")

if __name__ == "__main__":
    main()