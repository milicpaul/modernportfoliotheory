import time

import numpy as np
import pandas as pd
import random
import os
import PortfolioUtilities as pu
import multiprocessing
import sys
import ParallelComputing

class ModernPortfolioTheory():
    weight_list = []
    nbOfSimulatedWeights = 0
    threshold = 0
    path = ""
    sharpeRatio = 2

    def __init__(self, nbOfSimulatedWeights, threshold, sharpeRatio, parallel: ParallelComputing):
        if os.name != "Windows" and os.name != 'nt':
            self.path = "/Users/paul/Documents/Modern Portfolio Theory Data/"
        else:
            self.path = "C:/Users/paul.milic/Modern Portfolio/"
        self.nbOfSimulatedWeights = nbOfSimulatedWeights
        self.threshold = threshold
        self.sharpeRatio = sharpeRatio
        self.parallel = parallel

    def UpAndDownVolatility(self, data):
        return np.std(data[data < 0], ddof=1), np.std(data[data > 0], ddof=1)

    @staticmethod
    def MoveInSphere(w, epsilon=0.009):
        """Déplace w dans une petite boule de rayon epsilon et renormalise."""
        perturbation = np.random.uniform(-epsilon, epsilon, len(w))  # Bruit aléatoire
        new_w = w + perturbation  # Ajoute la perturbation
        new_w = np.clip(new_w, 0, 1)  # S'assure que les valeurs restent entre 0 et 1
        return np.round(new_w, 3)

    @staticmethod
    def Volatility(returns, optimization, epsilonIncrease, bestWeights, event, queueMessages):
        highestReturn = 0
        highestVolatility = 0
        highestSharpe = 0
        lowerVolatility = sys.maxsize
        returnsLowerVolatility = 0
        sharpeLowerVolatility = 0
        index = 0
        i = 0
        weightsList = []
        try:
            mean_returns = returns.mean() * 252  # Rendement moyen annuel
            cov_matrix = returns.cov() * 252  # Matrice de covariance annuelle
            while i < 10000:
                if not optimization:
                    weightsList.append(ModernPortfolioTheory.generate_weights(len(returns.columns)))
                else:
                    weightsList.append(ModernPortfolioTheory.MoveInSphere(bestWeights, 0.009 + epsilonIncrease))
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
                i += 1
                if i%500 == 0:
                    while not queueMessages.empty():
                        event.set()
                        time.sleep(0.1)
        except Exception as a:
            pass
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

    @staticmethod
    def generate_weights(n, min_val=0.01, max_val=0.9):
        alpha = np.ones(n)  # Distribution équilibrée
        vecteur = np.random.dirichlet(alpha, size=1)[0]
        vecteur = 0.01 + vecteur * (0.9 - 0.01)
        vecteur = vecteur / vecteur.sum()
        erreur = 1.0 - vecteur.sum()
        indice_max = np.argmax(vecteur)
        vecteur[indice_max] += erreur
        return np.round(vecteur, 2)

    @staticmethod
    def SelectRandomAssets(data, isin, nbOfSimulation, percentage, queueResults, queueMessages, event, showDensity=False, random=True, localPortfolio=[]):
        portfolioU = pu.PortfolioUtilities()
        timeSeries = data
        data = data.pct_change(fill_method=None)
        bestPortfolios = []
        queueMessages.put_nowait(f"[Portfolio]Starting Simulation, Process id {os.getpid()}")
        event.set()
        for j in range(nbOfSimulation):
            queueMessages.put(f"[Portfolio]Running Simulation, Process id {os.getpid()}, simulation {j}")
            enoughData = False
            while not enoughData:
                if random:
                    portfolio = portfolioU.ReturnRandomPortfolio(percentage, isin)
                else:
                    portfolio = localPortfolio
                portfolioLength = len(portfolio)
                currentData = data[portfolio]
                currentData = currentData[(currentData.index >= pd.to_datetime('2022-06-15')) & (
                                           currentData.index <= pd.to_datetime('2025-04-15'))]
                originalData = timeSeries[portfolio]
                enoughData = currentData.shape[1] == portfolioLength
            weightsList, highestReturn, highestVolatility, highestSharpe, lowerVolatility, returnsLower, sharpeLower = ModernPortfolioTheory.Volatility(currentData, False, 0, [], event, queueMessages)
            if showDensity:
                pu.PortfolioUtilities.ShowDensity(weightsList)
            bestPortfolios.append([portfolio, weightsList, highestReturn, highestVolatility, highestSharpe, currentData,
                originalData, lowerVolatility, returnsLower, sharpeLower])
        queueMessages.put_nowait(f"[Portfolio]Ending Simulation, Process id {os.getpid()}")
        queueResults.put(bestPortfolios[np.argmax(list(zip(*bestPortfolios))[4])])

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

def main():
    a = 1

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Nécessaire si tu crées un exécutable avec PyInstaller
    main()
else:
    pass