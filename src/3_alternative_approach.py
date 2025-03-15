"""
Alternative approach script for the SP100-Index-Tracker project.

This script implements a Clustering and Genetic Algorithm approach to
construct an index fund that tracks the S&P 100 index.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from deap import base, creator, tools, algorithms
import random
import time
import warnings
from tqdm import tqdm
from utils import ensure_dir, calculate_portfolio_statistics

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ignore warnings
warnings.filterwarnings('ignore')

class ClusteringGeneticAlgorithmOptimizer:
    """
    A class that combines clustering and genetic algorithm to optimize
    an index fund that tracks the S&P 100.
    """

    def __init__(self, stock_returns, sp100_returns, q, output_dir='results/ga'):
        """
        Initialize the optimizer.

        Parameters:
        -----------
        stock_returns : pandas.DataFrame
            DataFrame with daily returns for each stock.
        sp100_returns : pandas.Series
            Series with daily returns for S&P 100 index.
        q : int
            Number of stocks to select.
        output_dir : str
            Directory to save results.
        """
        self.stock_returns = stock_returns
        self.sp100_returns = sp100_returns
        self.q = q
        self.n_stocks = stock_returns.shape[1]
        self.tickers = stock_returns.columns.tolist()
        self.output_dir = output_dir

        # Ensure output directory exists
        ensure_dir(output_dir)

        # Calculate some statistics for clustering
        self.mean_returns = stock_returns.mean()
        self.std_returns = stock_returns.std()
        self.corr_with_sp100 = pd.Series(
            {stock: stock_returns[stock].corr(sp100_returns) for stock in self.tickers}
        )
        self.beta = pd.Series(
            {stock: stock_returns[stock].cov(sp100_returns) / sp100_returns.var()
             for stock in self.tickers}
        )

    def cluster_stocks(self, n_clusters=10):
        """
        Cluster stocks based on their characteristics.

        Parameters:
        -----------
        n_clusters : int
            Number of clusters to create.

        Returns:
        --------
        clusters : dict
            Dictionary mapping cluster labels to lists of stocks.
        """
        print(f"Clustering stocks into {n_clusters} clusters...")

        # Create features for clustering
        features = pd.DataFrame({
            'mean_return': self.mean_returns,
            'std_return': self.std_returns,
            'corr_with_sp100': self.corr_with_sp100,
            'beta': self.beta
        })

        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)

        # Group stocks by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(self.tickers[i])

        # Print cluster information
        for label, stocks in clusters.items():
            print(f"Cluster {label}: {len(stocks)} stocks")

        # Save cluster information
        cluster_df = pd.DataFrame({
            'stock': self.tickers,
            'cluster': cluster_labels
        })
        cluster_df.to_csv(f"{self.output_dir}/clusters_q{self.q}.csv", index=False)

        self.clusters = clusters
        return clusters

    def _calculate_portfolio_returns(self, weights):
        """
        Calculate portfolio returns given weights.

        Parameters:
        -----------
        weights : numpy.ndarray
            Array of weights for each stock.

        Returns:
        --------
        portfolio_returns : pandas.Series
            Series with daily returns for the portfolio.
        """
        return np.dot(self.stock_returns, weights)

    def _calculate_tracking_error(self, weights):
        """
        Calculate the tracking error between portfolio and S&P 100.

        Parameters:
        -----------
        weights : numpy.ndarray
            Array of weights for each stock.

        Returns:
        --------
        tracking_error : float
            Tracking error (root mean squared error).
        """
        portfolio_returns = self._calculate_portfolio_returns(weights)
        return np.sqrt(np.mean((portfolio_returns - self.sp100_returns) ** 2))

    def _calculate_correlation(self, weights):
        """
        Calculate the correlation between portfolio and S&P 100.

        Parameters:
        -----------
        weights : numpy.ndarray
            Array of weights for each stock.

        Returns:
        --------
        correlation : float
            Correlation coefficient.
        """
        portfolio_returns = self._calculate_portfolio_returns(weights)
        return np.corrcoef(portfolio_returns, self.sp100_returns)[0, 1]

    def setup_genetic_algorithm(self):
        """
        Set up the genetic algorithm for portfolio optimization.
        """
        # Create fitness class (minimizing tracking error)
        # Check if creator already has FitnessMin class to avoid error on re-runs
        if "FitnessMin" not in creator.__dict__:
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if "Individual" not in creator.__dict__:
            creator.create("Individual", list, fitness=creator.FitnessMin)

        # Initialize toolbox
        self.toolbox = base.Toolbox()

        # Register attribute generator
        self.toolbox.register("attr_float", random.uniform, 0, 1)

        # Register individual and population creation
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                             self.toolbox.attr_float, n=self.n_stocks)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Register evaluation function
        self.toolbox.register("evaluate", self._evaluate_individual)

        # Register genetic operators
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def _evaluate_individual(self, individual):
        """
        Evaluate an individual in the genetic algorithm.

        Parameters:
        -----------
        individual : list
            List of weights for each stock.

        Returns:
        --------
        tracking_error : tuple
            Tuple with tracking error (as required by DEAP).
        """
        # Convert individual to numpy array
        weights = np.array(individual)

        # Apply sparsity constraint (only q stocks have non-zero weights)
        # Find indices of q largest weights
        top_q_indices = np.argsort(weights)[-self.q:]

        # Create sparse weights (only q non-zero elements)
        sparse_weights = np.zeros_like(weights)
        sparse_weights[top_q_indices] = weights[top_q_indices]

        # Normalize weights to sum to 1
        if np.sum(sparse_weights) > 0:
            sparse_weights = sparse_weights / np.sum(sparse_weights)

        # Calculate tracking error
        tracking_error = self._calculate_tracking_error(sparse_weights)

        return (tracking_error,)

    def run_genetic_algorithm(self, n_generations=50, population_size=100):
        """
        Run the genetic algorithm to optimize the portfolio.

        Parameters:
        -----------
        n_generations : int
            Number of generations to run.
        population_size : int
            Size of the population.

        Returns:
        --------
        best_individual : list
            List of weights for each stock in the best solution.
        stats : dict
            Statistics of the optimization process.
        """
        print(f"Running genetic algorithm with population={population_size}, generations={n_generations}...")

        # Create initial population
        pop = self.toolbox.population(n=population_size)

        # Statistics to track
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Create a progress bar
        pbar = tqdm(total=n_generations)

        # Modified version without callback
        # Initialize the hall of fame
        hof = tools.HallOfFame(1)

        # Initialize logbook
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + stats.fields

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update hall of fame
        hof.update(pop)

        # Record stats
        record = stats.compile(pop)
        logbook.record(gen=0, nevals=len(invalid_ind), **record)

        # Update progress bar
        pbar.update(1)

        # Begin the evolution
        for gen in range(1, n_generations):
            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))

            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation
            for i in range(1, len(offspring), 2):
                if random.random() < 0.7:  # crossover probability
                    offspring[i-1], offspring[i] = self.toolbox.mate(offspring[i-1], offspring[i])
                    del offspring[i-1].fitness.values
                    del offspring[i].fitness.values

            for i in range(len(offspring)):
                if random.random() < 0.2:  # mutation probability
                    offspring[i], = self.toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update the population
            pop[:] = offspring

            # Update hall of fame
            hof.update(pop)

            # Record stats
            record = stats.compile(pop)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)

            # Update progress bar
            pbar.update(1)

        # Close progress bar
        pbar.close()

        # Get the best individual
        best_individual = hof[0]

        # Extract information from logbook
        gen = logbook.select("gen")
        fit_mins = logbook.select("min")
        fit_avgs = logbook.select("avg")

        # Plot evolution
        plt.figure(figsize=(10, 6))
        plt.plot(gen, fit_mins, 'b-', label='Minimum Tracking Error')
        plt.plot(gen, fit_avgs, 'r-', label='Average Tracking Error')
        plt.xlabel('Generation')
        plt.ylabel('Tracking Error')
        plt.title(f'Evolution of Tracking Error (q={self.q})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, f"ga_evolution_q{self.q}.png"))
        plt.close()

        return best_individual, {"gen": gen, "min": fit_mins, "avg": fit_avgs}

    def get_portfolio(self, individual):
        """
        Get the portfolio composition from an individual.

        Parameters:
        -----------
        individual : list
            List of weights for each stock.

        Returns:
        --------
        portfolio : dict
            Dictionary mapping stock tickers to weights.
        """
        # Convert individual to numpy array
        weights = np.array(individual)

        # Apply sparsity constraint (only q stocks have non-zero weights)
        # Find indices of q largest weights
        top_q_indices = np.argsort(weights)[-self.q:]

        # Create sparse weights (only q non-zero elements)
        sparse_weights = np.zeros_like(weights)
        sparse_weights[top_q_indices] = weights[top_q_indices]

        # Normalize weights to sum to 1
        if np.sum(sparse_weights) > 0:
            sparse_weights = sparse_weights / np.sum(sparse_weights)

        # Create portfolio dictionary
        portfolio = {}
        for i, ticker in enumerate(self.tickers):
            if sparse_weights[i] > 0:
                portfolio[ticker] = sparse_weights[i]

        return portfolio

    def optimize(self, n_clusters=10, n_generations=50, population_size=100):
        """
        Run the complete optimization process.

        Parameters:
        -----------
        n_clusters : int
            Number of clusters for stock clustering.
        n_generations : int
            Number of generations for genetic algorithm.
        population_size : int
            Size of the population for genetic algorithm.

        Returns:
        --------
        portfolio : dict
            Dictionary mapping stock tickers to weights.
        stats : dict
            Statistics of the optimization process.
        """
        # Step 1: Cluster stocks
        self.cluster_stocks(n_clusters)

        # Step 2: Set up genetic algorithm
        print("Setting up genetic algorithm...")
        self.setup_genetic_algorithm()

        # Step 3: Run genetic algorithm
        best_individual, stats = self.run_genetic_algorithm(n_generations, population_size)

        # Step 4: Get portfolio
        portfolio = self.get_portfolio(best_individual)

        # Step 5: Calculate performance metrics
        weights = np.zeros(self.n_stocks)
        for i, ticker in enumerate(self.tickers):
            if ticker in portfolio:
                weights[i] = portfolio[ticker]

        tracking_error = self._calculate_tracking_error(weights)
        correlation = self._calculate_correlation(weights)

        print(f"\nOptimization complete for q={self.q}:")
        print(f"Tracking Error: {tracking_error:.4f}")
        print(f"Correlation with S&P 100: {correlation:.4f}")
        print(f"Number of stocks selected: {len(portfolio)}")

        return portfolio, stats

def run_alternative_approach(train_returns_file='data/train_returns.csv',
                            q_values=[10, 15, 20, 25, 30],
                            output_dir='results/ga'):
    """
    Run the alternative approach for different q values.

    Parameters:
    -----------
    train_returns_file : str
        Path to the CSV file with training returns.
    q_values : list
        List of q values to try.
    output_dir : str
        Directory to save results.

    Returns:
    --------
    portfolios : dict
        Dictionary mapping q values to portfolios.
    """
    # Convert to absolute path
    train_returns_file = os.path.join(PROJECT_ROOT, train_returns_file)
    output_dir = os.path.join(PROJECT_ROOT, output_dir)

    # Ensure output directory exists
    ensure_dir(output_dir)

    print(f"Loading training returns from {train_returns_file}...")

    # Check if file exists
    if not os.path.exists(train_returns_file):
        raise FileNotFoundError(f"File not found: {train_returns_file}")

    # Load training returns
    train_returns = pd.read_csv(train_returns_file, index_col=0, parse_dates=True)

    # Extract S&P 100 returns
    sp100_returns = train_returns['^OEX']
    stock_returns = train_returns.drop('^OEX', axis=1)

    # Initialize portfolios dictionary
    portfolios = {}

    # Run optimization for each q value
    for q in q_values:
        print(f"\n{'='*50}")
        print(f"Optimizing portfolio with q={q}")
        print(f"{'='*50}")

        # Create optimizer
        optimizer = ClusteringGeneticAlgorithmOptimizer(stock_returns, sp100_returns, q, output_dir)

        # Run optimization
        portfolio, stats = optimizer.optimize(
            n_clusters=min(20, q),
            n_generations=50,
            population_size=100
        )

        # Store portfolio
        portfolios[q] = portfolio

        # Save portfolio to file
        portfolio_df = pd.DataFrame({
            'stock': list(portfolio.keys()),
            'weight': list(portfolio.values())
        })
        portfolio_df['q'] = q
        portfolio_df.to_csv(os.path.join(output_dir, f"ga_portfolio_q{q}.csv"), index=False)

    # Combine all portfolios into a single file
    all_portfolios = []
    for q, portfolio in portfolios.items():
        for stock, weight in portfolio.items():
            all_portfolios.append({
                'q': q,
                'stock': stock,
                'weight': weight
            })
    pd.DataFrame(all_portfolios).to_csv(os.path.join(output_dir, "ga_portfolio_results.csv"), index=False)

    return portfolios

def main():
    print("Starting alternative approach optimization...")
    # Ensure directories exist
    ensure_dir(os.path.join(PROJECT_ROOT, 'results'))
    ensure_dir(os.path.join(PROJECT_ROOT, 'results/ga'))

    # Run alternative approach for different q values
    q_values = [10, 15, 20, 25, 30]
    portfolios = run_alternative_approach('data/train_returns.csv', q_values, 'results/ga')

    print("\nAlternative approach optimization complete!")
    print("Results saved to results/ga directory.")
    print(f"Next, run the visualization and analysis script to compare the AMPL and GA approaches.")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")