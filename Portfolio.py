import numpy as np
import pandas as pd
import random
import os
import PortfolioUtilities as pu
import assets

# import assets as a

class ModernPortfolioTheory():
    weight_list = []

    def __init__(self, ):
        a = 1

    def MoveInSphere(self, w, epsilon=0.009):
        """Déplace w dans une petite boule de rayon epsilon et renormalise."""
        perturbation = np.random.uniform(-epsilon, epsilon, len(w))  # Bruit aléatoire
        # perturbation = perturbation/perturbation.sum() * 0.1
        new_w = w + perturbation  # Ajoute la perturbation
        new_w = np.clip(new_w, 0, 1)  # S'assure que les valeurs restent entre 0 et 1
        # new_w /= new_w.sum()  # Renormalisation pour que la somme soit 1
        return np.round(new_w, 3)

    def Volatility(self, returns, r, optimization, nbOfRandomWeights, epsilonIncrease, bestWeights):
        try:
            weightsList = []
            # Rendement moyen annuel
            mean_returns = returns.mean() * 252  # 252 jours de bourse par an
            # Matrice de covariance annuelle
            cov_matrix = returns.cov() * 252
            # Rendement, Volatilité, Sharpe
            # Simulation Monte-Carlo
            for i in range(nbOfRandomWeights):
                # Générer des poids aléatoires qui summent à 1
                # weights = np.random.dirichlet(np.ones(len(returns.columns)), size=1).flatten()
                if not optimization:
                    weightsList.append(self.generate_weights(len(returns.columns)))
                else:
                    weightsList.append(self.MoveInSphere(bestWeights, 0.009 + epsilonIncrease))
                # Calcul du rendement du portefeuille
                portfolio_return = np.sum(weightsList[-1] * mean_returns)
                # Calcul de la volatilité du portefeuille
                portfolio_volatility = np.sqrt(np.dot(weightsList[-1].T, np.dot(cov_matrix, weightsList[-1])))
                # Stocker les résultats
                try:
                    r[0][0, i] = portfolio_return
                    r[0][1, i] = portfolio_volatility
                    r[0][2, i] = portfolio_return / portfolio_volatility
                except Exception as b:
                    print("Volatility 1:", b)
        except Exception as a:
            print("Volatility b:", a)
        return weightsList

    def TransformToPickle(self, fileName):
        assets = pd.read_csv(fileName, on_bad_lines="skip", encoding_errors="ignore", sep=";")
        assets.to_pickle("C:/Users/paul.milic/Modern Portfolio/ETF Swiss Equity Themes.pkl")

    def FindInSphere(self, data, weight, nbOfSimulation):
        r = []
        results = np.zeros((3, nbOfSimulation * 10000))
        r.append(results)
        for i in range(nbOfSimulation):
            self.Volatility(data, i * 10000, r, weight, 100000)

    def ReturnDataset(self, portfolio, fullDataset):
        dataSetList = []
        for p in portfolio:
            dataSetList.append(fullDataset[[p]])
        dataset = pd.concat(dataSetList, axis=1)
        return dataset.sort_index(axis=1)

    def BuilHeterogeneousPortfolio(self, fileNames):
        if os.name != "Windows" and os.name != 'nt':
            path = "/Users/paul/Documents/Modern Portfolio Theory Data/"
        else:
            path = "C:/Users/paul.milic/Modern Portfolio/"
        data = []
        assets = []
        i = 0
        for f in fileNames:
            localDf = pd.read_pickle(path + f[0])
            localDf = localDf.loc[:, localDf.isna().mean() * 100 < 95]
            localDf = localDf.loc[localDf.isna().mean(axis=1) * 100 < 95, :]
            assets.append(list(localDf.columns))
            data.append(localDf)
        fullData = pd.concat(data, axis=1)
        fullData = fullData.loc[:, ~fullData.columns.duplicated()]
        return fullData, assets

    def generate_weights(self, n, min_val=0.01, max_val=0.9):
        alpha = np.ones(n)  # Distribution équilibrée
        vecteur = np.random.dirichlet(alpha, size=1)[0]
        # Transformer les valeurs dans l'intervalle [0.01, 0.9]
        vecteur = 0.01 + vecteur * (0.9 - 0.01)
        # Normaliser pour que la somme fasse exactement 1
        vecteur = vecteur / vecteur.sum()
        # Ajuster la plus grande valeur pour compenser l'erreur
        erreur = 1.0 - vecteur.sum()
        indice_max = np.argmax(vecteur)
        vecteur[indice_max] += erreur
        return np.round(vecteur, 2)

    def SelectRandomAssets(self, data, isin, nbOfSimulation, percentage):
        timeSeries = data
        data = data.pct_change()  # .dropna()
        bestPortfolios = []
        for j in range(nbOfSimulation):
            print(f"\rSimulation # : {j+1}", end="", flush=True)
            results = np.zeros((3, 10000))
            k = 0
            enoughData = False
            while not enoughData:
                portfolio = []  # random portfolio
                portfolioLenght = 0
                for p in percentage:
                    if len(isin[k]) < percentage[k][1]:
                        quantity = len(isin[k])
                    else:
                        quantity = percentage[k][1]
                    vecteur = random.sample(range(len(isin[k])), quantity)
                    portfolioLenght += quantity
                    for v in vecteur:
                        portfolio.append(isin[k][v])
                    k += 1
                k = 0
                currentData = self.ReturnDataset(portfolio, data)
                currentData = currentData[(currentData.index >= pd.to_datetime('2022-06-15')) & (
                                           currentData.index <= pd.to_datetime('2025-03-15'))]
                originalData = self.ReturnDataset(portfolio, timeSeries)
                enoughData = currentData.shape[1] == portfolioLenght
            weightsList = self.Volatility(currentData, [results], False, 10000, 0, [])
            maxSharpeId = np.argmax(results[2])
            bestPortfolios.append(
                [portfolio, weightsList[maxSharpeId], results[0][maxSharpeId], results[1][maxSharpeId],
                 results[2][maxSharpeId], currentData, originalData])
        return bestPortfolios[np.argmax(list(zip(*bestPortfolios))[4])]

    def FindMaximum(self, bestPortfolio, nbOfSimulation):
        bestPortfolios = []
        for i in range(nbOfSimulation):
            results = np.zeros((3, 10000))
            newWeightsList = self.Volatility(bestPortfolio[5], [results], True, 10000, i * 0.001, bestPortfolio[1])
            maxSharpeId = np.argmax(results[2])
            bestPortfolios.append([[], newWeightsList[maxSharpeId], results[0][maxSharpeId], results[1][maxSharpeId],
                                   results[2][maxSharpeId], []])
        return bestPortfolios[np.argmax(list(zip(*bestPortfolios))[4])]

    def GetOptimalPortfolioName(self, nbOfMonteCarloIteration, maxSharpeIdx):
        return (maxSharpeIdx - 1) // nbOfMonteCarloIteration

    def DisplayResults(self, bestPortfolio):
        try:
            print('')
            print("Optimal portfolio:", bestPortfolio[0])
            print("Optimal weight:", bestPortfolio[1])
            print("Optimal return", bestPortfolio[2] * 100)
            print("Optimal volatility:", bestPortfolio[3] * 100)
            print("Optimal sharpe ratio:", bestPortfolio[2] / bestPortfolio[3])
        except Exception as a:
            bestPortfolio[0] = "N/A"

        with open("/Users/paul/Documents/Modern Portfolio Theory Data/Portfolios results.csv", "a") as myFile:
            myFile.write(str(bestPortfolio[0]) + ';' +
                         str(bestPortfolio[1]) + ";" +
                         str(round(bestPortfolio[2] * 100, 2)) + ";" +
                         str(bestPortfolio[3] * 100) + ';' +
                         str(round(bestPortfolio[2] / bestPortfolio[3], 4)) + "\n"
                         )


#portfolioStructure = [["SMI Components.pkl", 5], ["Swiss Shares CHF.pkl", 0], ["Dow Jones.pkl", 0], ["NASDAQ100.pkl", 0], ['ETF Swiss Bonds.pkl', 0]]
portfolioStructure = [["SMI Mid Components CHF.pkl", 5],
                      ["ETF Equity Developed Markets CHF.pkl", 0],
                      ["SMI Components.pkl", 8],
                      ["ETF Swiss Bonds CHF.pkl", 0],
                      ["ETF Swiss Commodities CHF.pkl", 0]]
portfolio = ModernPortfolioTheory()
portfolioUtilities = pu.PortfolioUtilities()
#portfolioUtilities.GetAssetsTimeSeries(assets.nasdaq_100_tickers, "NASDAQ100.pkl")
#portfolioUtilities.GetTimeSeries("ETF Equity Developed Markets CHF.csv", False)
# portfolio.GetIsin("C:/Users/paul.milic/Modern Portfolio/ETF Swiss Equity Themes.csv")
# portfolio.TransformToPickle("C:/Users/paul.milic/Modern Portfolio/ETF Swiss Equity Themes.csv")
data, isin = portfolio.BuilHeterogeneousPortfolio(portfolioStructure)
bestPortfolio = portfolio.SelectRandomAssets(data, isin, 11, portfolioStructure)
print(portfolioUtilities.ReturnAssetDescription(bestPortfolio[0]))
portfolio.DisplayResults(bestPortfolio)
data = portfolio.ReturnDataset(bestPortfolio[0], data[data.index > pd.to_datetime('2022-06-15')])
data2 = data.pct_change()
portfolioPerformance = np.sum(bestPortfolio[1] * (data2.mean() * 252))
portfolioUtilities.plot_series_temporelles(data, bestPortfolio[2]/bestPortfolio[3], bestPortfolio[2], bestPortfolio[3], portfolioPerformance)
bestPortfolio = portfolio.FindMaximum(bestPortfolio, 2)
portfolio.DisplayResults(bestPortfolio)
print(1)
