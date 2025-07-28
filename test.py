import numpy as np
import matplotlib.pyplot as plt

# ---------- Données financières simulées ----------
mu = np.array([0.08, 0.03, 0.01])  # Rendements espérés
Sigma = np.array([
    [0.15**2, 0.01, 0.005],
    [0.01, 0.07**2, 0.002],
    [0.005, 0.002, 0.01**2],
])
rf = 0.0  # Taux sans risque

# ---------- Sommets du triangle ----------
A = np.array([0, 0])
B = np.array([1, 0])
C = np.array([0.5, np.sqrt(3)/2])
T = np.array([A, B, C])

# ---------- Fonctions utiles ----------
def random_portfolios(n_points=1000):
    w = np.random.rand(n_points, 3)
    return w / w.sum(axis=1, keepdims=True)

def bary_to_cartesian(w):
    return w[:, 0][:, None]*A + w[:, 1][:, None]*B + w[:, 2][:, None]*C

def cartesian_to_barycentric(P, A, B, C):
    v0 = B - A
    v1 = C - A
    v2 = P - A
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w
    return np.array([u, v, w])

# ---------- Simulation des portefeuilles ----------
portfolios = random_portfolios(1000)
xy = bary_to_cartesian(portfolios)
expected_returns = portfolios @ mu
variances = np.einsum('ij,ij->i', portfolios @ Sigma, portfolios)
std_devs = np.sqrt(variances)
sharpe_ratios = (expected_returns - rf) / std_devs

# ---------- Tracé du triangle ----------
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
ax.axis('off')
plt.title("Portefeuilles dans le triangle\n(Couleur = Sharpe ratio)")

# Triangle
triangle = plt.Polygon(T, edgecolor='black', fill=False, linewidth=1.5)
ax.add_patch(triangle)

# Noms des sommets
ax.text(*A - 0.05, "Actions", fontsize=12)
ax.text(*B + [0.02, -0.05], "Oblig.", fontsize=12)
ax.text(*C + [0, 0.03], "Cash", fontsize=12)

# Points colorés
sc = ax.scatter(xy[:, 0], xy[:, 1], c=sharpe_ratios, cmap='viridis', s=15)
cbar = plt.colorbar(sc, ax=ax, orientation='vertical', label='Sharpe Ratio')

# ---------- Interaction au clic ----------
def on_click(event):
    if event.inaxes != ax:
        return
    P = np.array([event.xdata, event.ydata])
    w = cartesian_to_barycentric(P, A, B, C)
    if np.all(w >= -1e-6):  # dans le triangle
        w = np.clip(w, 0, 1)
        w /= w.sum()  # pour éviter les erreurs numériques
        print(f"\n📍 Point cliqué : {P}")
        print(f"📊 Poids du portefeuille : Actions = {w[0]:.2%}, Oblig = {w[1]:.2%}, Cash = {w[2]:.2%}")
        ax.plot(P[0], P[1], 'ro', markersize=6)
        fig.canvas.draw()
    else:
        print("⚠️ Hors du triangle.")

fig.canvas.mpl_connect('button_press_event', on_click)

plt.tight_layout()
plt.show()
