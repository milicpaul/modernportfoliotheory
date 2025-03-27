import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import assets
import os
import seaborn as sns
import requests
import numpy as np

class PortfolioUtilities():
    def __init__(self):
        if os.name == "nt":
            self.path = "C:/Users/paul.milic/Modern Portfolio/"
        else:
            self.path = "/Users/paul/Documents/Modern Portfolio Theory Data/"

    def FindIsin(self, isin, fileNames):
        found = [False for _ in range(len(isin))]
        for i in isin:
            for f in fileNames:
                df = pd.read_pickle(self.path + f[0])
                if i in df.columns.tolist():
                    found[isin.index(i)] = True
                    break
        return found

        return False
    def GetAssetsTimeSeries(self, assetComponents, fileName):
        isinDf = yf.download(assetComponents, start="2015-01-01", end="2025-03-01", interval="1d")
        isinDf = isinDf['Close']
        isinDf.to_pickle(self.path + fileName)

    def GetTicker(self, component):
        spy = yf.Ticker("SPY")
        print(spy.holdings)
        dow = yf.Ticker("^DJI")
        dow_tickers = dow.history_metadata.get("symbols", [])

        print("Tickers du Dow Jones récupérés:", dow_tickers)

        tickers = yf.Ticker(component)
        dfTickers = tickers.mutualfund_holders
        return yf.Ticker(component).history(period="1d").index

    def GetTimeSeries(self, fileName, toTicker):
        df = pd.read_csv(self.path + fileName, on_bad_lines='skip', encoding_errors='ignore', sep=";")
        isin = []
        for d in df.itertuples(index=False):
            if toTicker:
                d = self.isin_to_ticker(d.ISIN, False)
            isin.append(d.ISIN)
        isinDf = yf.download(isin, start="2015-01-01", end="2025-03-01", interval="1d")
        isinDf = isinDf.loc[:, isinDf.isna().mean() * 100 < 100]['Close']
        isinDf.to_pickle(self.path + fileName.replace(".csv", ".pkl"))

    def ReturnAssetDescription(self, isin):
        found = []
        for i in isin:
            ticker = self.isin_to_ticker(i, True)
            if ticker != "Ticker introuvable":
                found.append(ticker)
            else:
                found.append(i)
        return found

    def isin_to_ticker(self, isin, name = True):
        HEADERS = {"Content-Type": "application/json", "X-OPENFIGI-APIKEY": "ffc41f35-e136-4c95-866e-82783229c4cd"}
        url = "https://api.openfigi.com/v3/mapping"
        data = [{"idType": "ID_ISIN", "idValue": isin}]
        response = requests.post(url, json=data, headers=HEADERS)
        if response.status_code == 200:
            result = response.json()
            if result and "data" in result[0]:
                if name:
                    return result[0]["data"][0]["name"]
                else:
                    return result[0]["data"][0]["ticker"]
        return "Ticker introuvable"

    def plot_series_temporelles(self, df, sharpe_ratio=None, rendement=None, volatilite=None, performance=None,
                                title="Évolution des Séries Économiques", xlabel="Date", ylabel="Valeur",
                                log_scale=True):
        """
        Affiche un graphe de séries temporelles avec labels à gauche et à droite des courbes.

        :param df: DataFrame Pandas avec une colonne de dates en index et plusieurs séries économiques.
        :param sharpe_ratio: Ratio de Sharpe du portefeuille (float).
        :param rendement: Rendement du portefeuille en % (float).
        :param volatilite: Volatilité du portefeuille en % (float).
        :param title: Titre du graphique.
        :param xlabel: Étiquette de l'axe des x.
        :param ylabel: Étiquette de l'axe des y.
        :param log_scale: Booléen pour activer l'échelle logarithmique sur l'axe Y.
        """

        # Vérifier que l'index est bien en datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("L'index du DataFrame doit être de type datetime.")

        # Style et figure
        sns.set(style="darkgrid")
        fig, ax = plt.subplots(figsize=(12, 6))

        # Tracer chaque série et ajouter un label à gauche et à droite
        for col in df.columns:
            ax.plot(df.index, df[col], label=col)

            # Ajouter un label à la fin de chaque courbe
            last_value = df[col].iloc[-1]  # Dernière valeur de la série
            last_date = df.index[-1]  # Dernière date
            ax.text(last_date, last_value, col, fontsize=12, verticalalignment='center',
                    horizontalalignment='left',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle="round,pad=0.3"))

            # Ajouter un label au début de chaque courbe
            first_value = df[col].iloc[0]  # Première valeur de la série
            first_date = df.index[0]  # Première date
            ax.text(first_date, first_value, col, fontsize=12, verticalalignment='center',
                    horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle="round,pad=0.3"))

        # Personnalisation du graphique
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        plt.xticks(rotation=45)

        # Activer l'échelle logarithmique si demandé
        if log_scale:
            ax.set_yscale("log")

        # Ajouter des annotations pour les indicateurs financiers
        info_text = ""
        if sharpe_ratio is not None:
            info_text += f"Ratio de Sharpe: {sharpe_ratio:.2f}\n\n"
        if rendement is not None:
            info_text += f"Rendement: {rendement*100:.2f}%\n\n"
        if volatilite is not None:
            info_text += f"Volatilité: {volatilite*100:.2f}%\n\n"
        if performance is not None:
            info_text += f"Performance: {performance*100:.2f}%"

        if info_text:
            # Ajouter un encadré de texte sur le graphique
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightgray'))
        try:
            plt.tight_layout()
        except Exception as a:
            print(a)

        plt.show()

    @staticmethod
    def ShowDensity(data):
        data = np.array(data)
        num_columns = data.shape[1]

        # Définir les positions verticales pour chaque colonne
        y_positions = np.arange(1, num_columns + 1)  # 1, 2, 3, ..., num_columns

        plt.figure(figsize=(10, 6))

        # Tracer chaque colonne comme une ligne horizontale et placer les points
        for i in range(num_columns):
            plt.scatter(data[:, i], [y_positions[i]] * len(data[:, i]), label=f'Colonne {i + 1}')
            plt.axhline(y=y_positions[i], color="gray", linestyle="--", alpha=0.5)  # Ligne horizontale

        # Ajustement des axes
        plt.xlim(0, 1)  # Valeurs en abscisse entre 0 et 1
        plt.ylim(0, num_columns + 1)  # Ajuster pour voir toutes les lignes

        # Labels et titre
        plt.xlabel("Valeurs normalisées entre 0 et 1")
        plt.ylabel("Colonnes")
        plt.title("Projection des colonnes sur des droites horizontales")

        # Activer la légende et le quadrillage
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.3)

        # Affichage du graphique
        plt.show()
