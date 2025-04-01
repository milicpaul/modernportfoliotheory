import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Données de rendements des actifs (exemple avec 4 actifs)
# Supposons que nous avons les rendements mensuels des 4 actifs pour 100 mois
np.random.seed(42)
n_assets = 4
n_obs = 100
returns = np.random.randn(n_obs, n_assets) / 100  # rendements mensuels simulés

# Calcul des rendements moyens et de la matrice de covariance
mean_returns = np.mean(returns, axis=0)  # Moyenne des rendements pour chaque actif
cov_matrix = np.cov(returns, rowvar=False)  # Matrice de covariance des rendements


# Fonction pour calculer le rendement du portefeuille
def portfolio_return(weights, mean_returns):
    return np.sum(weights * mean_returns)


# Fonction pour calculer le risque (volatilité) du portefeuille
def portfolio_risk(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


# Fonction objectif à minimiser (on veut maximiser le rendement pour un niveau de risque donné)
def objective_function(weights, mean_returns, cov_matrix, target_risk):
    # On maximise le rendement pour un niveau de risque donné
    risk = portfolio_risk(weights, cov_matrix)
    ret = portfolio_return(weights, mean_returns)

    # Si le risque est trop élevé, pénaliser
    penalty = 1000 * (risk - target_risk) ** 2 if risk > target_risk else 0

    # Retourner le rendement négatif (minimisation), on ajoute la pénalité
    return -ret + penalty


# Contrainte que la somme des poids doit être égale à 1 (portefeuille complet)
def constraint(weights):
    return np.sum(weights) - 1


# Initialisation des poids des actifs (poids égaux au début)
initial_weights = np.ones(n_assets) / n_assets

# Définition des contraintes (somme des poids = 1, chaque poids >= 0)
constraints = ({'type': 'eq', 'fun': constraint})

# Définition des bornes pour les poids (0 <= poids <= 1)
bounds = tuple((0, 1) for asset in range(n_assets))

# Objectif : maximiser le rendement pour un risque cible de 1%
target_risk = 0.01

# Optimisation
result = minimize(objective_function, initial_weights, args=(mean_returns, cov_matrix, target_risk),
                  method='SLSQP', bounds=bounds, constraints=constraints)

# Affichage des résultats
optimal_weights = result.x
optimal_return = portfolio_return(optimal_weights, mean_returns)
optimal_risk = portfolio_risk(optimal_weights, cov_matrix)

print("Poids optimaux des actifs :", optimal_weights)
print("Rendement attendu du portefeuille :", optimal_return)
print("Risque du portefeuille :", optimal_risk)

# Affichage des résultats graphiquement (optionnel)
labels = [f'Asset {i + 1}' for i in range(n_assets)]
plt.bar(labels, optimal_weights)
plt.title('Poids optimaux du portefeuille')
plt.show()
