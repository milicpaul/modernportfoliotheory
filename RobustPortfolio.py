import numpy as np
import cvxpy as cp
import pandas as pd

class RobustPortfolio():

    def RobustPortfolio(self, data, variation=True):
        mu  = data.mean() * 252
        Sigma = data.cov() * 252
        n_assets = len(data.columns)
        # Définition des incertitudes (on suppose une erreur de ±10% sur mu et Sigma)
        if variation:
            delta_mu = 0.10 * mu  # 10% d'incertitude sur les rendements
            delta_Sigma = 0.10 * Sigma  # 10% d'incertitude sur la covariance

        # Variables d'optimisation : poids du portefeuille
        w = cp.Variable(len(data.columns))

        # Définition de l'ensemble d'incertitude pour mu
        mu_robust = cp.Parameter(n_assets)
        if variation:
            mu_robust.value = (mu - delta_mu).values # Cas pessimiste
        else:
            mu_robust.value = mu.values
        # Définition de l'ensemble d'incertitude pour Sigma
        Sigma_robust = cp.Parameter((n_assets, n_assets))
        if variation:
            Sigma_robust.value = (Sigma + delta_Sigma).values  # Cas pessimiste
        else:
            Sigma_robust.value = Sigma.values
        #Sigma_robust.value = 0.5 * (Sigma_robust.value + Sigma_robust.value.T)  # Rendre symétrique

        # Formulation du problème robuste
        risk = cp.quad_form(w, Sigma_robust.value)  # Risque du portefeuille
        expected_return = mu_robust @ w  # Rendement espéré

        # Contraintes : somme des poids = 1 et pas de vente à découvert
        constraints = [cp.sum(w) == 1, w >= 0.01]

        # Résolution du problème robuste
        prob = cp.Problem(cp.Minimize(risk), constraints)
        prob.solve()
        return np.round(w.value, 2), np.sqrt(risk.value)
