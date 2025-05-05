import itertools
import ParallelComputing
import pandas as pd
import os
import numpy as np

class Statistics():
    def __init__(self):
        if os.name != "Windows" and os.name != 'nt':
            self.path = "/Users/paul/Documents/Modern Portfolio Theory Data/"
        else:
            self.path = "C:/Users/paul.milic/Modern Portfolio/"

    import numpy as np
    @staticmethod
    def rendre_definie_positive(matrice, epsilon=1e-4):
        n = matrice.shape[0]
        return matrice + epsilon * np.eye(n)

    def assurer_definie_positive(matrice, epsilon=1e-6):
        """
        Vérifie si une matrice est définie positive, sinon la corrige.
        - Utilise Cholesky pour tester la positivité définie.
        - Corrige par ajout d’un petit multiple de l'identité si nécessaire.
        """
        try:
            # Test de Cholesky : réussit si la matrice est définie positive
            np.linalg.cholesky(matrice)
            return matrice  # ✅ Rien à faire, tout va bien
        except np.linalg.LinAlgError:
            # 🔧 Pas définie positive : on ajoute epsilon * I
            print("[Info] Matrice corrigée (pas définie positive)")
            n = matrice.shape[0]
            # On peut itérer jusqu'à ce que Cholesky réussisse
            factor = 1
            while True:
                try:
                    matrice_corrigée = matrice + epsilon * factor * np.eye(n)
                    np.linalg.cholesky(matrice_corrigée)
                    return matrice_corrigée
                except np.linalg.LinAlgError:
                    factor *= 10
                    if factor > 1e6:
                        raise RuntimeError("Impossible de rendre la matrice définie positive")

    def PositiveVariance(self, fileName):
        df = pd.read_pickle(self.path + fileName)
        df = df.pct_change(fill_method=None)
        increasingIsin = []
        stdMean = df.std().mean()
        for c in df.columns:
            if df[df[c] > 0][c].sum() > np.abs(df[df[c] < 0][c].sum()) and df[c].std() < stdMean:
                increasingIsin.append(df[c])
        dfNew = pd.concat(increasingIsin, axis=1)
        dfNew.to_pickle(self.path + fileName.replace(".pkl", "_positive_variance_.pkl"))

    def FindLowestCovariance(self, fileName):
        corrDf = pd.read_pickle(self.path + fileName).corr()
        np.fill_diagonal(corrDf.values, 0)
        scores = corrDf.abs().sum()
        return scores.sort_values(key=lambda x: x[1] if isinstance(x, tuple) else x).index.tolist()

    def EstimateVolatility(self, portfolios, data):
        volatilities = []
        weights = np.ones(6)/6
        #weights = np.array([0.42, 0.01, 0.29, 0.01, 0.25, 0.01])
        i = 0
        for portfolio in portfolios:
            df = data[list(portfolio)]
            df = df[df.index >= pd.to_datetime('2022-06-15')]
            volatilities.append([np.sqrt(np.dot(weights.T, np.dot(df.cov()*252, weights))), portfolio])
            i += 1
            if i%10 == 0:
                print(f"\riteration # : {i}, {len(portfolios)}", end="", flush=True)
            if i == 10000:
                return volatilities[np.argmin([x[0] for x in volatilities])]
                break
        return volatilities[np.argmin([x[0] for x in volatilities])]

    def CompareVolatility(self, portfolio, data, weights):
        data = data[data.index >= pd.to_datetime('2022-06-15')]
        df1 = pd.read_pickle(self.path + "NASDAQ100.pkl")
        df1 = df1[df1.index >= pd.to_datetime('2022-06-15')]
        data = data[portfolio]
        cov = data.cov() * 252
        df = pd.read_csv(self.path + "test.csv", sep=";")
        df1 = pd.read_pickle(self.path + "NASDAQ100.pkl")
        return np.sqrt(np.dot(weights.T, np.dot(cov, weights)))


def main():
    stat = Statistics()
    isin = stat.FindLowestCovariance("NASDAQ100_positive_variance_.pkl")
    #print(isin)
    #res = stat.CompareVolatility(['KDP', 'PFE', 'MRK', 'XEL', 'CHKP', 'PEP'], pd.read_pickle(stat.path + "NASDAQ100_positive_variance_.pkl"),np.array([0.01, 0.01, 0.2,  0.11, 0.67, 0.01]))
    portfolios = list(itertools.combinations(isin, 6))
    portfolios = [list(t) for t in portfolios]
    volatility = ParallelComputing.Parallel.compute_volatility(stat, portfolios, pd.read_pickle(stat.path + "NASDAQ100_positive_variance_.pkl"))
    #portfolio = stat.EstimateVolatility(portfolios, pd.read_pickle(stat.path + "NASDAQ100_positive_variance_.pkl"))
    print(volatility)


if __name__ == '__main__':
    main()
