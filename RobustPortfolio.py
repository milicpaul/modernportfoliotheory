import cvxpy as cp
import numpy as np

# Estimations (ex: historiques)
mu_hat = np.array([0.05, 0.07])           # rendements espérés
Sigma = np.array([[0.1, 0.02], [0.02, 0.08]])  # matrice covariance
delta = 0.02                              # incertitude max sur mu

# Variables
w = cp.Variable(2)

# Objectif: rendement minimum dans la pire hypothèse
mu_uncertain = mu_hat - delta * np.ones(2)
objective = cp.Minimize(-mu_uncertain @ w + cp.quad_form(w, Sigma))

constraints = [cp.sum(w) == 1, w >= 0]
problem = cp.Problem(objective, constraints)
problem.solve()

print("Poids robustes :", w.value)
