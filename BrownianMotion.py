import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class BrownianMotion():

    def extend_series_with_brownian(self, S, n_days=30):
        log_returns = np.diff(np.log(S))
        mu = np.mean(log_returns)
        sigma = np.std(log_returns)
        Z = np.random.normal(0, 1, n_days)
        drift = (mu - 0.5 * sigma ** 2)
        simulated_log_returns = drift + sigma * Z
        S_last = S[-1]
        simulated_prices = S_last * np.exp(np.cumsum(simulated_log_returns))
        return np.concatenate([S, simulated_prices])

# Exemple : série factice
    def PrintBrownianMotion(self):
        S = np.linspace(100, 110, 60) + np.random.normal(0, 1, 60)  # 60 jours de données
        extended = self.extend_series_with_brownian(S, n_days=30)

        plt.plot(extended)
        plt.axvline(len(S), color='r', linestyle='--')
        plt.title("Extension par mouvement brownien")
        plt.show()