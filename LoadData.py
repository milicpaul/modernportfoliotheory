import pandas as pd

import PortfolioUtilities as pu
import assets
import yfinance as yf
import os

class GetISINTimeSeries():
    portfolioUtilities = pu.PortfolioUtilities()
    path = ""
    def __init__(self):
        if os.name != "Windows" and os.name != 'nt':
            self.path = "/Users/paul/Documents/Modern Portfolio Theory Data/"
        else:
            self.path = "C:/Users/paul.milic/Modern Portfolio/"

    def GetAssetsTimeSeries(self, assetComponents, fileName):
        isinDf = yf.download(assetComponents, start="2015-01-01", end="2025-03-01", interval="1d", auto_adjust=True)
        isinDf = isinDf['Close']
        isinDf.to_pickle(self.path + fileName)

    def GetAssetsTimeSeries2(self, assetComponents, fileName):
        for a in assetComponents:
            print(a)
            isinDf = yf.download([a], start="2015-01-01", end="2025-03-01", interval="1d", auto_adjust=True)
        isinDf = isinDf['Close']
        #isinDf.to_pickle(self.path + fileName)

    def GetTimeSeries(self, choice, fileName, asset):
        if choice == 1:
            print(self.portfolioUtilities.FindIsin(assets.ETFMSCIWorld, self.portfolioStructure))
        elif choice == 2:
            self.GetAssetsTimeSeries2(asset, fileName)
        elif choice == 3:
            self.portfolioUtilities.GetTimeSeries(fileName, False)
        elif choice == 4:
            self.portfolioUtilities.GetIsin("C:/Users/paul.milic/Modern Portfolio/ETF Swiss Equity Themes.csv")
        else:
            self.portfolioUtilities.TransformToPickle("C:/Users/paul.milic/Modern Portfolio/ETF Swiss Equity Themes.csv")
        return 0

if __name__ == '__main__':
    getIsin = GetISINTimeSeries()
    #getIsin.GetTimeSeries(2, "Pietro.pkl", assets.Pietro)
    df = pd.read_pickle(getIsin.path + "Pietro.pkl")
    print(1)
    #getIsin.GetTimeSeries(3, "ETF CHF.csv")
    #portfolioUtilities = PortfolioUtilities.PortfolioUtilities()
    #portfolioUtilities.GetAssetsTimeSeries(assets.DAX40, "DAX40.pkl")
