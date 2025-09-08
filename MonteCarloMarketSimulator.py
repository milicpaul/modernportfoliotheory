import pandas as pd
import numpy as np

class MonteCarloMarketSimulator:
    def __init__(self, mainList):
        """
        mu : vecteur des rendements moyens (taille = nb actifs)
        cov : matrice de covariance (nb actifs x nb actifs)
        start_prices : prix initiaux des actifs
        n_scenarios : nombre de scénarios Monte Carlo
        horizon : nombre de pas de temps (ex: 252 jours de trading)
        """
        initialPrices = mainList[6]
        self.weights = mainList[1]
        self.mu = initialPrices.pct_change().mean()
        self.cov = initialPrices.pct_change().cov()
        self.start_prices = np.array(initialPrices.iloc[-1])
        self.n_assets = len(self.start_prices)
        self.n_scenarios = 1000
        self.horizon = 252
        self.prices = self.simulate()
        self.ComputeReturns(self.start_prices)

    def ComputeReturns(self, start_prices):
        initialPrice = self.start_prices @ self.weights
        print(initialPrice)
        bestPrice = []
        for p in self.prices:
            bestPrice.append(p[-1] @ self.weights)
            for b in bestPrice:
                print(b)

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

if __name__ == "__main__":
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
