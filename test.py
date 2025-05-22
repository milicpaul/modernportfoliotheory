import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate random data for returns
def generate_random_data(num_assets, num_days):
    np.random.seed(42)  # For reproducibility
    # Generate daily random returns for each asset
    returns = np.random.randn(num_days, num_assets) * 0.01  # Mean close to 0 and std dev of 1% per day
    return pd.DataFrame(returns, columns=[f'Asset_{i+1}' for i in range(num_assets)])

# Calculate mean returns and covariance matrix
def calculate_mean_return_and_covariance(returns):
    mean_returns = returns.mean()
    covariance_matrix = returns.cov()
    return mean_returns, covariance_matrix

# Function to calculate the Markowitz portfolio performance
def calculate_portfolio_performance(weights, mean_returns, covariance_matrix):
    portfolio_return = np.sum(weights * mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    return portfolio_return, portfolio_volatility

# Function to generate the efficient frontier
def efficient_frontier(mean_returns, covariance_matrix, num_portfolios=10000):
    results = np.zeros((3, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)  # Normalize weights
        portfolio_return, portfolio_volatility = calculate_portfolio_performance(weights, mean_returns, covariance_matrix)
        results[0,i] = portfolio_return
        results[1,i] = portfolio_volatility
        results[2,i] = portfolio_return / portfolio_volatility  # Sharpe ratio
    return results

# Example usage with random data
num_assets = 5  # Number of assets
num_days = 252  # Number of trading days in a year

# Generate random data for returns
returns = generate_random_data(num_assets, num_days)

# Calculate mean returns and covariance matrix
mean_returns, covariance_matrix = calculate_mean_return_and_covariance(returns)

# Generate the efficient frontier
results = efficient_frontier(mean_returns, covariance_matrix)

# Plot the efficient frontier
plt.figure(figsize=(10, 6))
plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis', marker='o')
plt.title('Markowitz Efficient Frontier')
plt.xlabel('Volatility (Risk)')
plt.ylabel('Expected Return')
plt.colorbar(label='Sharpe Ratio')
plt.show()
