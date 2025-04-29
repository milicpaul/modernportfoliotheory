import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import os
import seaborn as sns
import requests
import numpy as np
import threading
import random
from datetime import datetime, timedelta
import time

class PortfolioUtilities():
    cvsLock = threading.Lock()
    def __init__(self):
        if os.name == "nt":
            self.path = "C:/Users/paul.milic/Modern Portfolio/"
        else:
            self.path = "/Users/paul/Documents/Modern Portfolio Theory Data/"
        self.df = pd.read_csv(self.path + "Assets Description.csv", sep=",", index_col=False)

    def CheckPklFile(self, filename):
        df = pd.read_pickle(self.path + filename)
        pass

    def RefreshPkl(self):
        files = [f for f in os.listdir(self.path) if f.endswith('.pkl')]
        for f in files:
            df = pd.read_pickle(self.path + f)
            isinDf = yf.download(df.columns.tolist(), start=(df.index[-1] + timedelta(days=1)).date().isoformat(), end=datetime.today(), interval="1d")
            isinDf = isinDf['Close']
            fullData = pd.concat([df, isinDf], axis=0)
            #fullData.to_pickle(self.path + f)

    def GetAllAssets(self):
        assets = []
        files = [f for f in os.listdir(self.path) if f.endswith('.pkl')]
        assetNames = pd.read_csv(self.path + "Assets Description.csv", sep=",")
        for f in files:
            df = round(pd.read_pickle(self.path + f).pct_change(fill_method=None), 2)
            for c in df.columns:
                description = assetNames.loc[assetNames['ISIN'] == c, 'Description'].values
                if description.size == 0:
                    desc = self.ReturnAssetDescription([c])
                    if len(desc[0]) > 0:
                        description = desc[0]
                        self.df.loc[len(self.df)] = [c, description]
                    else:
                        description = ''
                if isinstance(description, np.ndarray): description = description[0]
                assets.append([description, c, round(df[c].var(), 4), df[df[c]>0][c].sum(), df[df[c]<0][c].sum(), f])
        self.df.to_csv(self.path + "Assets Description.csv")
        return assets

    def DisplayIsin(self, portfolioStructure):
        df = pd.DataFrame()
        for p in portfolioStructure:
            df = pd.read_pickle(self.path + p[0])
            for c in df.columns:
                print(c, self.ReturnAssetDescription([c])[0])
        self.df.to_csv(self.path + "Assets Description.csv", sep=";")

    def FindIsin(self, isin, fileNames):
        found = [False for _ in range(len(isin))]
        for i in isin:
            for f in fileNames:
                df = pd.read_pickle(self.path + f[0])
                if i in df.columns.tolist():
                    found[isin.index(i)] = True
                    break
        return found

    def GetTicker(self, component):
        spy = yf.Ticker("SPY")
        dow = yf.Ticker("^DJI")
        dow_tickers = dow.history_metadata.get("symbols", [])

        print("Tickers du Dow Jones récupérés:", dow_tickers)

        tickers = yf.Ticker(component)
        dfTickers = tickers.mutualfund_holders
        return yf.Ticker(component).history(period="1d").index

    def GetTimeSeries(self, fileName, toTicker):
        df = pd.read_csv(self.path + fileName, on_bad_lines='skip', encoding_errors='ignore', sep=";")
        isin = df.iloc[:,df.columns.get_loc('ISIN')].tolist()
        isinDf = yf.download(isin, start="2015-01-01", end="2025-03-01", interval="1d")
        isinDf = isinDf.loc[:, isinDf.isna().mean() * 100 < 100]['Close']
        isinDf.to_pickle(self.path + fileName.replace(".csv", ".pkl"))

    def ReturnAssetDescription(self, isin):
        found = []
        for i in isin:
            try:
                description = next(iter(self.df.loc[self.df["ISIN"] == i, "Description"].values), None)
            except Exception as e:
                print("ReturnAssetDescription:", e)
            if not description is None:
                found.append(description)
            else:
                ticker = self.isin_to_ticker(i, True)
                if ticker != "Ticker introuvable":
                    self.df.loc[len(self.df)] = [i, ticker]
                    found.append(ticker)
                else:
                    found.append(i)
        return found

    def isin_to_ticker(self, isin, name = True):
        HEADERS = {"Content-Type": "application/json", "X-OPENFIGI-APIKEY": "ffc41f35-e136-4c95-866e-82783229c4cd"}
        url = "https://api.openfigi.com/v3/mapping"
        data = [{"idType": "ID_ISIN", "idValue": isin}]
        try:
            response = requests.post(url, json=data, headers=HEADERS)
            if response.status_code == 200:
                result = response.json()
                if result and "data" in result[0]:
                    if name:
                        return result[0]["data"][0]["name"]
                    else:
                        return result[0]["data"][0]["ticker"]
        except Exception as e:
            a = 1
        return "Ticker introuvable"

    def plot_series_temporelles(self, assetsDescription, df, sharpe_ratio=None, rendement=None, volatilite=None, performance=None,
                                title="Évolution des Séries Économiques", xlabel="Date", ylabel="Valeur",
                                log_scale=True):
        """
        Affiche un graphe de séries temporelles avec labels à gauche et à droite des courbes.

        :param df: DataFrame Pandas avec une colonne de dates en index et plusieurs séries économiques.
        :param sharpe_ratio: Ratio de Sharpe du portefeuille (float).
        :param rendement: Rendement du portefeuille en % (float).
        :param volatilite: Volatilité du portefeuille en % (float).
        :param performance: Performance du portefeuille en % (float).
        :param title: Titre du graphique.
        :param xlabel: Étiquette de l'axe des x.
        :param ylabel: Étiquette de l'axe des y.
        :param log_scale: Booléen pour activer l'échelle logarithmique sur l'axe Y.
        """
        try:
            # Filtrer les dates à partir de 2015
            df = df[df.index >= pd.to_datetime('2015-01-01')]

            # Assurer que l'index est bien en datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("L'index du DataFrame doit être de type datetime.")

            # Conversion de toutes les colonnes en numérique (les erreurs deviennent NaN)
            df = df.apply(pd.to_numeric, errors='coerce')

            # Style du graphique
            sns.set(style="darkgrid")
            fig, ax = plt.subplots(figsize=(12, 6))
            names = pd.DataFrame(columns=assetsDescription)
            j = 0
            #for col in df.columns:
            for col in names.columns:
                #serie = df[col].dropna()
                serie = df.iloc[:, j].dropna()
                j += 1
                if serie.empty:
                    continue  # On saute les colonnes vides

                ax.plot(serie.index, serie.values, label=col)

                # Labels de début et de fin de courbe (avec cast en float)
                try:
                    first_value = float(serie.iloc[0])
                    first_date = serie.index[0]
                    ax.text(first_date, first_value, col, fontsize=12, verticalalignment='center',
                            horizontalalignment='right',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle="round,pad=0.3"))
                except Exception as e:
                    print(f"Erreur sur la première valeur de {col}: {e}")

                try:
                    last_value = float(serie.iloc[-1])
                    last_date = serie.index[-1]
                    ax.text(last_date, last_value, col, fontsize=12, verticalalignment='center',
                            horizontalalignment='left',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle="round,pad=0.3"))
                except Exception as e:
                    print(f"Erreur sur la dernière valeur de {col}: {e}")

            # Titre et axes
            ax.set_title(title, fontsize=14)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            plt.xticks(rotation=45)

            # Échelle logarithmique
            if log_scale:
                ax.set_yscale("log")

            # Encadré d'infos financières
            info_text = ""
            if sharpe_ratio is not None:
                info_text += f"Ratio de Sharpe: {sharpe_ratio:.2f}\n\n"
            if rendement is not None:
                info_text += f"Rendement: {rendement * 100:.2f}%\n\n"
            if volatilite is not None:
                info_text += f"Volatilité: {volatilite * 100:.2f}%\n\n"
            if performance is not None:
                info_text += f"Performance: {performance * 100:.2f}%"

            if info_text:
                ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=12,
                        verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightgray'))

            try:
                plt.tight_layout()
            except Exception as e:
                print(f"Erreur dans tight_layout : {e}")

            plt.show()

        except Exception as e:
            print(f"Erreur dans plot_series_temporelles : {e}")

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

    def ReturnDataset(self, portfolio, fullDataset):
        return fullDataset[list(portfolio)[0]]

    def TransformToPickle(self, fileName):
        assets = pd.read_csv(fileName, on_bad_lines="skip", encoding_errors="ignore", sep=";")
        assets.to_pickle("C:/Users/paul.milic/Modern Portfolio/ETF Swiss Equity Themes.pkl")

    def ReturnRandomPortfolio(self, percentage, isin):
        k = 0
        portfolio = []  # random portfolio
        localRandom = random.Random(time.time_ns() + id(threading.current_thread()))
        for p in percentage:
            if p[1] > 0:
                if len(isin[k]) < percentage[k][1]:
                    indices = localRandom.sample(range(len(isin[k])), range(len(isin[k])))
                else:
                    indices = localRandom.sample(range(len(isin[k])), percentage[k][1])
                #indices = [random.choices(range(len(isin[k])), k=percentage[k][1]) for k in range(len(percentage))]
                portfolio += [isin[k][i] for i in indices]
            k += 1
        k = 0
        return portfolio

class ColumnManager:
    def __init__(self, columns: list[dict]):
        self.columns = columns.copy()  # copie défensive

    def has(self, field_name: str) -> bool:
        """Vérifie si une colonne existe"""
        return any(col.get('field') == field_name for col in self.columns)

    def index_of(self, field_name: str) -> int:
        """Retourne l'index de la colonne, ou -1 si absente"""
        for i, col in enumerate(self.columns):
            if col.get('field') == field_name:
                return i
        return -1

    def add_after(self, after_field: str, new_column: dict) :
        """Ajoute une colonne après celle spécifiée"""
        index = self.index_of(after_field)
        if index == -1:
            raise ValueError(f"Colonne '{after_field}' non trouvée")
        self.columns.insert(index + 1, new_column)
        return self.columns

    def remove(self, field_name: str) :
        """Supprime la première occurrence d'une colonne"""
        index = self.index_of(field_name)
        if index != -1:
            self.columns.pop(index)
        return self.columns

    def remove_all(self, field_name: str):
        """Supprime toutes les occurrences d'une colonne"""
        self.columns = [col for col in self.columns if col.get('field') != field_name]
        return self.columns

    def get(self) -> list[dict]:
        """Retourne la liste des colonnes actuelle"""
        return self.columns.copy()

    def __repr__(self):
        return f"ColumnManager({self.columns})"

if __name__ == '__main__':
    ut = PortfolioUtilities()
    ut.CheckPklFile('FTSE Mib.pkl')
    ut.RefreshPkl()
