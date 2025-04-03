import numpy as np
import matplotlib.pyplot as plt

# Paramètres des actifs
prix_initials = [100, 100]  # Prix initiaux de deux actifs
rendements_moyens = [0.05, 0.07]  # Rendements moyens annuels des actifs
volatilites = [0.2, 0.25]  # Volatilités annuelles des actifs
corrélation = 0.3  # Corrélation entre les actifs
n_trajectoires = 1000  # Nombre de trajectoires à simuler
n_jours = 252  # Nombre de jours de trading dans une année
periods = n_jours  # Nombre de périodes par simulation (1 par jour)

# Matrice de covariance entre les actifs
covariance = np.array([[volatilites[0] ** 2, corrélation * volatilites[0] * volatilites[1]],
                       [corrélation * volatilites[0] * volatilites[1], volatilites[1] ** 2]])

# Temps (en jours)
time = np.linspace(0, 1, periods)


# Fonction pour simuler un mouvement brownien géométrique pour un actif
def simulate_geometric_brownian_motion(S0, mu, sigma, T, N):
    dt = T / N  # intervalle de temps
    dW = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=(N, 1))  # rendements aléatoires
    W = np.cumsum(dW, axis=0)  # Cumsum pour obtenir la trajectoire
    t = np.linspace(0, T, N)
    S = S0 * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * W.flatten())
    return S


# Simuler les trajectoires pour chaque actif
trajectoires = []
for i in range(len(prix_initials)):
    trajectoires_actif = []
    for _ in range(n_trajectoires):
        trajectoire = simulate_geometric_brownian_motion(prix_initials[i], rendements_moyens[i], volatilites[i], 1,
                                                         periods)
        trajectoires_actif.append(trajectoire)
    trajectoires.append(np.array(trajectoires_actif))

# Convertir les trajectoires en numpy array
trajectoires = np.array(trajectoires)


# Fonction pour simuler des portefeuilles avec des poids aléatoires
# Fonction pour simuler des portefeuilles avec des poids aléatoires
def simulate_portfolios(n_portfolios, trajectoires, covariance):
    portefeuilles = []

    for _ in range(n_portfolios):
        # Générer des poids aléatoires pour les deux actifs
        poids = np.random.random(len(prix_initials))
        poids /= np.sum(poids)  # Normaliser pour que la somme des poids soit égale à 1

        # Calculer les rendements simulés du portefeuille
        # On prend les trajectoires des actifs et on applique les poids correspondants
        rendements_portefeuille = np.dot(trajectoires.T, poids)  # Poids des actifs sur les trajectoires

        # Calculer le rendement moyen et l'écart-type (volatilité) du portefeuille
        rendement_portefeuille = np.mean(rendements_portefeuille, axis=1)  # Rendement moyen
        volatilite_portefeuille = np.std(rendements_portefeuille, axis=1)  # Volatilité des rendements

        portefeuilles.append((poids, rendement_portefeuille, volatilite_portefeuille))

    return portefeuilles


# Simuler les portefeuilles
n_portefeuilles = 1000
portefeuilles = simulate_portfolios(n_portefeuilles, trajectoires, covariance)

# Extraire les rendements et les volatilités
rendements_portefeuille = np.array([x[1] for x in portefeuilles])
volatilite_portefeuille = np.array([x[2] for x in portefeuilles])

# Tracer la frontière de risque
plt.figure(figsize=(10, 6))
plt.scatter(volatilite_portefeuille, rendements_portefeuille, c=rendements_portefeuille / volatilite_portefeuille,
            cmap='viridis')
plt.colorbar(label='Ratio rendement/risque')
plt.title('Frontière de Risque des Portefeuilles')
plt.xlabel('Volatilité (Risque)')
plt.ylabel('Rendement')
plt.grid(True)
plt.show()
