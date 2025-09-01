import numpy as np
import pandas as pd

class MonteCarloMarketSimulator:
    def __init__(self, mu, cov, start_prices, n_scenarios=1000, horizon=252):
        """
        mu : vecteur des rendements moyens (taille = nb actifs)
        cov : matrice de covariance (nb actifs x nb actifs)
        start_prices : prix initiaux des actifs
        n_scenarios : nombre de scénarios Monte Carlo
        horizon : nombre de pas de temps (ex: 252 jours de trading)
        """
        self.mu = mu
        self.cov = cov
        self.start_prices = np.array(start_prices)
        self.n_assets = len(start_prices)
        self.n_scenarios = n_scenarios
        self.horizon = horizon

    def simulate(self):
        # Simule les rendements journaliers multivariés
        shocks = np.random.multivariate_normal(self.mu, self.cov,
                                               (self.n_scenarios, self.horizon))
        # Convertit en trajectoires de prix
        prices = np.zeros((self.n_scenarios, self.horizon + 1, self.n_assets))
        prices[:, 0, :] = self.start_prices

        for t in range(1, self.horizon + 1):
            prices[:, t, :] = prices[:, t-1, :] * np.exp(shocks[:, t-1, :])

        return prices


# Exemple avec 3 actifs
mu = np.array([0.0005, 0.0003, 0.0004])   # rendements moyens quotidiens
cov = np.array([
    [0.0001, 0.00008, 0.00005],
    [0.00008, 0.0002, 0.00007],
    [0.00005, 0.00007, 0.00015]
])  # matrice de covariance
start_prices = [100, 50, 200]

sim = MonteCarloMarketSimulator(mu, cov, start_prices, n_scenarios=500, horizon=252)
prices = sim.simulate()

print("Forme des données simulées:", prices.shape)
# → (500 scénarios, 253 jours, 3 actifs)
