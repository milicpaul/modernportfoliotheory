import time
import numpy as np
import pandas as pd
import random
import os
import PortfolioUtilities as pu
import multiprocessing
import ParallelComputing
import tensorflow as tf
import Statistics


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
    def Volatility(returns, optimization, epsilonIncrease, bestWeights, event, queueMessages, nbOfSimulation):
        lenReturns = len(returns.columns)
        returns = returns.dropna(thresh=int(returns.shape[0] * 0.5), axis=1)
        if len(returns.columns) < lenReturns:
            queueMessages.put("[Volatility]Not enough data to compute volatility")
            return ([], 0, 0, 0, 0, 0, 0)
        highestReturn = 0
        highestVolatility = 0
        highestSharpe = 0
        lowerVolatility = float('inf')
        returnsLowerVolatility = 0
        sharpeLowerVolatility = 0
        i = 0
        bestWeights = np.zeros(len(returns.columns))

        try:
            cov_matrix = Statistics.Statistics.rendre_definie_positive(returns.cov() * 252)  # Matrice de covariance annuelle
            cov = tf.convert_to_tensor(cov_matrix.values, dtype=tf.float32)

            while i <= nbOfSimulation:
                if not optimization:
                    weights_np = ModernPortfolioTheory.generate_weights(len(returns.columns))
                else:
                    weights_np = ModernPortfolioTheory.MoveInSphere(bestWeights, 0.009 + epsilonIncrease)

                weights_np = weights_np / np.sum(weights_np)
                weights_tf = tf.convert_to_tensor(weights_np, dtype=tf.float32)
                weights_tf = tf.reshape(weights_tf, [-1, 1])

                with tf.device('/GPU:0'):
                    mean_returns = returns.mean() * 252  # Rendement moyen annuel
                    # Calcul du rendement
                    portfolio_return = np.sum(weights_np * mean_returns)
                    # Calcul de la volatilité avec TensorFlow
                    volatility_tf = tf.sqrt(tf.matmul(tf.transpose(weights_tf), tf.matmul(cov, weights_tf)))
                    portfolio_volatility = volatility_tf.numpy().item()  # Scalaire Python

                if portfolio_volatility < lowerVolatility:
                    lowerVolatility = portfolio_volatility
                    returnsLowerVolatility = portfolio_return
                    sharpeLowerVolatility = round(portfolio_return / portfolio_volatility, 2)

                if portfolio_volatility > 0:
                    try:
                        sharpe = portfolio_return / portfolio_volatility
                        if sharpe > highestSharpe:
                            highestSharpe = sharpe
                            highestReturn = portfolio_return
                            highestVolatility = portfolio_volatility
                            bestWeights = weights_np
                    except ZeroDivisionError as e:
                        print(e)
                i += 1
                if i % 10000 == 0:
                    while not queueMessages.empty():
                        event.set()
                        time.sleep(0.03)

        except Exception as e:
            print(f"[Volatility] Erreur : {e}")

        return (
            np.round(bestWeights, 2),
            highestReturn,
            highestVolatility,
            highestSharpe,
            lowerVolatility,
            returnsLowerVolatility,
            sharpeLowerVolatility
        )

    def FindInSphere(self, data, weight, nbOfSimulation):
        r = []
        results = np.zeros((3, nbOfSimulation * self.nbOfSimulatedWeights))
        r.append(results)
        for i in range(nbOfSimulation):
            self.Volatility(data, i * self.nbOfSimulatedWeights, r, weight, self.nbOfSimulatedWeights)

    def BuilHeterogeneousPortfolio(self, fileNames, dateFrom):
        fullData = pd.DataFrame()
        data = []
        assets = []
        for f in fileNames:
            if f[1] > 0:
                localDf = pd.read_pickle(self.path + f[0])
                localDf = localDf[localDf.index >= dateFrom]
                #localDf = localDf.loc[:, localDf.isna().mean() * 100 < 50]
                #localDf = localDf.loc[localDf.isna().mean(axis=1) * 100 < 50, :]
                assets.append(list(localDf.columns))
                data.append(localDf)
        if len(assets) > 0:
            fullData = pd.concat(data, axis=1)
            fullData = fullData.loc[:, ~fullData.columns.duplicated()]
        return fullData, assets

    @staticmethod
    def generate_weights(n, min_val = 0.01):
        # Génère un vecteur Dirichlet
        alpha = np.ones(n)  # Distribution équilibrée
        vec = np.random.dirichlet(alpha)

        while True:
            vec = np.random.dirichlet(alpha)
            rounded = np.round(vec, 2)
            # Corriger la somme
            diff = 1.0 - np.sum(rounded)
            i = np.argmax(rounded)
            rounded[i] += diff
            rounded = np.round(rounded, 2)  # re-arrondir après correction
            if np.all(rounded >= min_val):
                return rounded

    def generate_weights2(n, min_val=0.01):
        alpha = np.ones(n)  # Distribution équilibrée
        samples = []
        while len(samples) < n:
            s = np.random.dirichlet(alpha)
            if np.all(s >= min_val):
                samples.append(s)
        return np.round(np.array(samples), 2)

    @staticmethod
    def SelectRandomAssets(data, isin, nbOfSimulation, percentage, queueResults,
                           queueMessages, event, showDensity=False, random=True,
                           localPortfolio=None):
        if localPortfolio is None:
            localPortfolio = []
        portfolioU = pu.PortfolioUtilities()
        previousSharpeRatio = 0
        timeSeries = data
        data = data.pct_change(fill_method=None)
        bestPortfolios = []
        queueMessages.put_nowait(f"[Portfolio]Starting Simulation, Process id {os.getpid()}")
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
                originalData = timeSeries[portfolio]
                enoughData = currentData.shape[1] == portfolioLength
            weightsList, highestReturn, highestVolatility, highestSharpe, lowerVolatility, returnsLower, sharpeLower = ModernPortfolioTheory.Volatility(currentData, False, 0, [], event, queueMessages, nbOfSimulation)
            if showDensity:
                pu.PortfolioUtilities.ShowDensity(weightsList)

            queueMessages.put_nowait(f"[Portfolio]Ending Simulation, Process id {os.getpid()}")
            while not queueMessages.empty():
                event.set()
                time.sleep(0.05)
            if len(weightsList) > 0:
                try:
                    if highestReturn / highestVolatility > previousSharpeRatio:
                        bestPortfolios = [portfolio, weightsList, highestReturn, highestVolatility, highestSharpe, currentData,
                                          originalData, lowerVolatility, returnsLower, sharpeLower]
                except Exception as e:
                    print(e)
        queueResults.put(bestPortfolios)
        queueMessages.put_nowait(f"[Portfolio]Ending Process, Process id {os.getpid()}")

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