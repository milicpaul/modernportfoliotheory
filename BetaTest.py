import numpy as np
class BetaTest():

    def ComputeBeta(self, asset1, asset2):
        returnAsset1 = asset1.pct_change().dropna()
        returnAsset2 = asset2.pct_change().dropna()
        sigma = np.cov(returnAsset1, returnAsset2)
        marketVariance = np.var(returnAsset2)
        return sigma/marketVariance