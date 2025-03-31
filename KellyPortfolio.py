import numpy as np
from scipy.optimize import minimize

# Simulons des rendements moyens et une matrice de covariance
np.random.seed(42)
n_assets = 4
mean_returns = np.random.uniform(0.05, 0.15, n_assets)  # Rendement moyen
cov_matrix = np.random.rand(n_assets, n_assets)
cov_matrix = np.dot(cov_matrix, cov_matrix.T)  # Matrice symétrique positive définie

# Taux sans risque
risk_free_rate = 0.02

# Fonction Kelly pour plusieurs actifs (résout la version matricielle)
def kelly_criterion(weights, mean_returns, cov_matrix, risk_free_rate):
    portfolio_return = np.dot(weights, mean_returns) - risk_free_rate
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    return - portfolio_return / portfolio_variance  # On minimise l'inverse

# Contraintes : somme des poids = 1
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = [(0, 1) for _ in range(n_assets)]  # Pas de ventes à découvert

# Optimisation
initial_weights = np.array([1/n_assets] * n_assets)
opt_result = minimize(kelly_criterion, initial_weights, args=(mean_returns, cov_matrix, risk_free_rate),
                      method='SLSQP', bounds=bounds, constraints=constraints)

optimal_weights_kelly = opt_result.x
print("Poids optimaux (Kelly) :", optimal_weights_kelly)
