import numpy as np
import pandas as pd
import random
import os
import PortfolioUtilities as pu
import multiprocessing
import sys

class ModernPortfolioTheory():
    portfolioStructure = [["AEX Netherland.pkl", 0],
                          ["CAC 40.pkl", 0],
                          ["DAX40.pkl", 0],
                          ["Dow Jones.pkl", 0],
                          ["ETF CHF.pkl", 2],
                          ["ETF CHF_positive_variance_.pkl", 0],
                          ["ETF Equity Developed Markets CHF.pkl", 0],
                          ["ETF MSCI World.pkl", 0],
                          ["ETF Swiss Bonds.pkl", 0],
                          ["ETF Swiss Commodities CHF.pkl", 0],
                          ["FTSE Mib.pkl", 0],
                          ["NASDAQ100.pkl", 0],
                          ["SMI Components.pkl", 2],
                          ["SMI Mid Components CHF.pkl", 0],
                          ["Swiss Bonds ETF.pkl", 2],
                          ["Swiss Bonds.pkl", 2, ],
                          ["Swiss Equities Emerging Market ETF.pkl", 0],
                          ["Swiss Shares CHF.pkl", 0],
                          ["Swiss Shares SMI Expanded.pkl", 0],
                          ["Swiss Shares SMI Mid.pkl", 2],
                          ["Swiss Shares SMI.pkl", 0],
                          ["Swiss Shares.pkl", 0],
                          ["Pietro.pkl", 0]]

    weight_list = []
    nbOfSimulatedWeights = 0
    threshold = 0
    path = ""
    sharpeRatio = 2

    def __init__(self, nbOfSimulatedWeights, threshold, sharpeRatio):
        if os.name != "Windows" and os.name != 'nt':
            self.path = "/Users/paul/Documents/Modern Portfolio Theory Data/"
        else:
            self.path = "C:/Users/paul.milic/Modern Portfolio/"
        self.nbOfSimulatedWeights = nbOfSimulatedWeights
        self.threshold = threshold
        self.sharpeRatio = sharpeRatio

    def UpAndDownVolatility(self, data):
        return np.std(data[data < 0], ddof=1), np.std(data[data > 0], ddof=1)

    def MoveInSphere(self, w, epsilon=0.009):
        """Déplace w dans une petite boule de rayon epsilon et renormalise."""
        perturbation = np.random.uniform(-epsilon, epsilon, len(w))  # Bruit aléatoire
        new_w = w + perturbation  # Ajoute la perturbation
        new_w = np.clip(new_w, 0, 1)  # S'assure que les valeurs restent entre 0 et 1
        return np.round(new_w, 3)

    def Volatility(self, returns, optimization, epsilonIncrease, bestWeights):
        highestReturn = 0
        highestVolatility = 0
        highestSharpe = 0
        lowerVolatility = sys.maxsize
        index = 0
        i = 0
        weightsList = []
        try:
            returns.to_csv(self.path + "test.csv", sep=";")
            mean_returns = returns.mean() * 252  # Rendement moyen annuel
            cov_matrix = returns.cov() * 252  # Matrice de covariance annuelle
            while i < self.nbOfSimulatedWeights:
                if not optimization:
                    weightsList.append(self.generate_weights(len(returns.columns)))
                else:
                    weightsList.append(self.MoveInSphere(bestWeights, 0.009 + epsilonIncrease))
                portfolio_return = np.sum(weightsList[-1] * mean_returns)
                portfolio_volatility = np.sqrt(np.dot(weightsList[-1].T, np.dot(cov_matrix, weightsList[-1])))
                if portfolio_volatility < lowerVolatility:
                    lowerVolatility = portfolio_volatility
                    returnsLowerVolatility = portfolio_return
                    sharpeLowerVolatility = round(portfolio_return/portfolio_volatility, 2)
                if portfolio_volatility > 0 and (portfolio_return / portfolio_volatility > highestSharpe):
                    highestSharpe = portfolio_return / portfolio_volatility
                    highestReturn = portfolio_return
                    highestVolatility = portfolio_volatility
                    index = i
                    stop = portfolio_volatility >= self.threshold
                i += 1
        except Exception as a:
            print("Volatility error:", a)
        return weightsList[index], highestReturn, highestVolatility, highestSharpe, lowerVolatility, returnsLowerVolatility, sharpeLowerVolatility

    def FindInSphere(self, data, weight, nbOfSimulation):
        r = []
        results = np.zeros((3, nbOfSimulation * self.nbOfSimulatedWeights))
        r.append(results)
        for i in range(nbOfSimulation):
            self.Volatility(data, i * self.nbOfSimulatedWeights, r, weight, self.nbOfSimulatedWeights)

    def BuilHeterogeneousPortfolio(self, fileNames):
        fullData = pd.DataFrame()
        data = []
        assets = []
        for f in fileNames:
            localDf = pd.read_pickle(self.path + f[0])
            localDf = localDf.loc[:, localDf.isna().mean() * 100 < 95]
            localDf = localDf.loc[localDf.isna().mean(axis=1) * 100 < 95, :]
            assets.append(list(localDf.columns))
            data.append(localDf)
        if len(assets) > 0:
            fullData = pd.concat(data, axis=1)
            fullData = fullData.loc[:, ~fullData.columns.duplicated()]
        return fullData, assets

    def generate_weights(self, n, min_val=0.01, max_val=0.9):
        alpha = np.ones(n)  # Distribution équilibrée
        vecteur = np.random.dirichlet(alpha, size=1)[0]
        vecteur = 0.01 + vecteur * (0.9 - 0.01)
        vecteur = vecteur / vecteur.sum()
        erreur = 1.0 - vecteur.sum()
        indice_max = np.argmax(vecteur)
        vecteur[indice_max] += erreur
        return np.round(vecteur, 2)

    def SelectRandomAssets(self, data, isin, nbOfSimulation, percentage, process, showDensity=True, random=True, localPortfolio=[]):
        portfolioU = pu.PortfolioUtilities()
        timeSeries = data
        data = data.pct_change(fill_method=None)  # .dropna()
        bestPortfolios = []

        for j in range(nbOfSimulation):
            print(f"\rSimulation # : {j+1}", end="", flush=True)
            enoughData = False
            while not enoughData:
                if random:
                    portfolio = portfolioU.ReturnRandomPortfolio(percentage, isin, process)
                else:
                    portfolio = localPortfolio
                portfolioLength = len(portfolio)
                currentData = data[portfolio]
                currentData = currentData[(currentData.index >= pd.to_datetime('2022-06-15')) & (
                                           currentData.index <= pd.to_datetime('2025-04-15'))]
                originalData = timeSeries[portfolio]
                enoughData = currentData.shape[1] == portfolioLength
            weightsList, highestReturn, highestVolatility, highestSharpe, lowerVolatility, returnsLower, sharpeLower = self.Volatility(currentData, False, 0, [])
            if showDensity:
                pu.PortfolioUtilities.ShowDensity(weightsList)
            bestPortfolios.append(
                [portfolio, weightsList, highestReturn, highestVolatility, highestSharpe, currentData,
                 originalData, lowerVolatility, returnsLower, sharpeLower])
        return bestPortfolios[np.argmax(list(zip(*bestPortfolios))[self.sharpeRatio])]

    def FindMaximum(self, bestPortfolio, nbOfSimulation):
        bestPortfolios = []
        for i in range(nbOfSimulation):
            weightsList, highestReturn, highestVolatility, highestSharpe = self.Volatility(bestPortfolio[5], True, i * 0.001, bestPortfolio[1])
            bestPortfolios.append([[], weightsList, highestReturn, highestVolatility, highestSharpe, bestPortfolio[5], []])
        return bestPortfolios[np.argmax(list(zip(*bestPortfolios))[self.sharpeRatio])]

    def GetOptimalPortfolioName(self, nbOfMonteCarloIteration, maxSharpeIdx):
        return (maxSharpeIdx - 1) // nbOfMonteCarloIteration

    def FindBestPortfolio(self, fullPortfolio):
        nbOfInstruments = len(fullPortfolio)
        for i in range(100):
            v = random.choices(range(0, nbOfInstruments), k=nbOfInstruments)
            for j in range(nbOfInstruments):
                fullPortfolio[j][1] = v[j]
            data, isin = self.BuilHeterogeneousPortfolio(fullPortfolio)
            bestPortfolio = self.SelectRandomAssets(data, isin, 1, self.portfolioStructure, False)
            print(pu.ReturnAssetDescription(bestPortfolio[0]))
            #self.DisplayResults(bestPortfolio)

def main():
    portfolioStructure = [["AEX Netherland.pkl", 0],
                          ["CAC 40.pkl", 0],
                          ["DAX40.pkl", 0],
                          ["Dow Jones.pkl", 0],
                          ["ETF CHF.pkl", 2],
                          ["ETF CHF_positive_variance_.pkl", 0],
                          ["ETF Equity Developed Markets CHF.pkl", 0],
                          ["ETF MSCI World.pkl", 0],
                          ["ETF Swiss Bonds.pkl", 0],
                          ["ETF Swiss Commodities CHF.pkl", 0],
                          ["FTSE Mib.pkl", 0],
                          ["NASDAQ100.pkl", 0],
                          ["SMI Components.pkl", 2],
                          ["SMI Mid Components CHF.pkl", 0],
                          ["Swiss Bonds ETF.pkl", 2],
                          ["Swiss Bonds.pkl", 2, ],
                          ["Swiss Equities Emerging Market ETF.pkl", 0],
                          ["Swiss Shares CHF.pkl", 0],
                          ["Swiss Shares SMI Expanded.pkl", 0],
                          ["Swiss Shares SMI Mid.pkl", 2],
                          ["Swiss Shares SMI.pkl", 0],
                          ["Swiss Shares.pkl", 0],
                          ["Pietro.pkl", 0]]


if __name__ == '__main__':
    multiprocessing.freeze_support()  # Nécessaire si tu crées un exécutable avec PyInstaller
    main()