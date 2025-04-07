import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class BrownianMotion():

    # 🔹 Paramètres des 3 actifs
    prix_initials = [100, 100, 100]  # Prix initiaux des actifs
    rendements_moyens = [0.05, 0.07, 0.06]  # Rendements moyens annuels
    volatilites = [0.2, 0.25, 0.18]  # Volatilités annuelles
    correlations = np.array([
        [1.0, 0.3, 0.5],  # Corrélation Actif 1 avec 1, 2, 3
        [0.3, 1.0, 0.4],  # Corrélation Actif 2 avec 1, 2, 3
        [0.5, 0.4, 1.0]  # Corrélation Actif 3 avec 1, 2, 3
    ])

    def __init__(self, prix_initials):
        self.prix_initials = prix_initials.apply(
            lambda col: pd.Series(col[col.first_valid_index()] if col.first_valid_index() is not None else np.nan)
        )
        self.prix_initials = self.prix_initials.to_numpy().tolist()[0]
        self.rendements_moyens = prix_initials.pct_change(fill_method=None).mean() * 252
        self.correlations = prix_initials.pct_change(fill_method=None).corr()
        self.volatilites = prix_initials.pct_change(fill_method=None).std() * 252
        #tt

# 🔹 Simulation des trajectoires corrélées des actifs
    def simulate_geometric_brownian_motion_corr(self, S0, mu, L, T, N, n_simulations):
        dt = T / N  # Intervalle de temps
        dW = np.random.normal(size=(n_simulations, N, len(S0))) * np.sqrt(dt)  # Bruit brownien
        # ✅ Correction : appliquer la matrice de Cholesky L.T correctement
        dW_corr = dW @ L.T
        S = np.zeros((n_simulations, N, len(S0)))
        S[:, 0, :] = S0  # Initialiser les prix
        for t in range(1, N):
            S[:, t, :] = S[:, t - 1, :] * np.exp((np.array(mu) - 0.5 * np.diag(L @ L.T)) * dt + dW_corr[:, t, :])
        return S

    def simulate_portfolios(self, n_portfolios, trajectoires):
        portefeuilles = []
        for _ in range(n_portfolios):
            # Générer des poids aléatoires
            poids = np.random.random(len(self.prix_initials))
            poids /= np.sum(poids)  # Normaliser à 1

            # Calculer les rendements pondérés des portefeuilles
            rendements_portefeuille = np.dot(trajectoires, poids)  # Application des poids sur les trajectoires

            # Calcul du rendement et de la volatilité du portefeuille
            rendement_portefeuille = np.mean(rendements_portefeuille, axis=1)
            volatilite_portefeuille = np.std(rendements_portefeuille, axis=1)

            portefeuilles.append((poids, rendement_portefeuille, volatilite_portefeuille))
        return portefeuilles

    def Simulate(self, n_jours, n_trajectoires):
        # 🔹 Matrice de covariance entre les actifs
        covariance = np.outer(self.volatilites, self.volatilites) * self.correlations
        # 🔹 Décomposition de Cholesky pour corréler les actifs
        L = np.linalg.cholesky(covariance)

        # 🔹 Simuler les trajectoires des actifs (corrélées)
        trajectoires = self.simulate_geometric_brownian_motion_corr(self.prix_initials, self.rendements_moyens, L, 1, n_jours, n_trajectoires)
        # 🔹 Simulation des portefeuilles
        # 🔹 Simuler les portefeuilles
        n_portefeuilles = 1000
        portefeuilles = self.simulate_portfolios(n_portefeuilles, trajectoires)

        # 🔹 Extraire les rendements et les volatilités
        rendements_portefeuille = np.array([x[1] for x in portefeuilles])
        volatilite_portefeuille = np.array([x[2] for x in portefeuilles])
        print("max sharpe:", np.max(rendements_portefeuille/volatilite_portefeuille))
        self.PlotEfficientLimit(volatilite_portefeuille, rendements_portefeuille)

        # 🔹 Tracer la frontière de risque
    def PlotEfficientLimit(self, volatilite_portefeuille, rendements_portefeuille):
        plt.figure(figsize=(10, 6))
        plt.scatter(volatilite_portefeuille, rendements_portefeuille, c=rendements_portefeuille / volatilite_portefeuille,
                    cmap='viridis')
        plt.colorbar(label='Ratio rendement/risque')
        plt.title('Frontière de Risque des Portefeuilles (3 actifs)')
        plt.xlabel('Volatilité (Risque)')
        plt.ylabel('Rendement')
        plt.grid(True)
        plt.show()
