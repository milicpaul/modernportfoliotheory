import numpy as np
from scipy.optimize import minimize

class KellyCriterion():
    # Taux sans risque
    risk_free_rate = 0.02
    returns = 0
    variance = 0
    # Fonction Kelly pour plusieurs actifs (résout la version matricielle)
    def kelly_criterion(self, weights, mean_returns, cov_matrix, risk_free_rate):
        portfolio_return = np.dot(weights, mean_returns) - risk_free_rate
        portfolio_variance = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        self.returns = portfolio_return
        self.variance = portfolio_variance
        return - portfolio_return / portfolio_variance # On minimise l'inverse
    def SolveKellyCriterion(self, returns, n_assets):
        # Contraintes : somme des poids = 1
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(n_assets)]  # Pas de ventes à découvert
        cov_matrix = returns.cov() * 252
        mean_returns = returns.mean() * 252
        # Optimisation
        initial_weights = np.array([1/n_assets] * n_assets)
        opt_result = minimize(self.kelly_criterion, initial_weights, args=(mean_returns, cov_matrix, self.risk_free_rate),
                              method='SLSQP', bounds=bounds, constraints=constraints)
        return opt_result.x
