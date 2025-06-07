import pandas as pd
import yfinance as yf
import os
from datetime import date
import assets

from yaml import full_load


class Refresh():
    def __init__(self):
        if os.name == "nt":
            self.path = "C:/Users/paul.milic/Modern Portfolio/"
            self.path2 = "C:/Users/paul.milic/Modern Portfolio/Data/"
        else:
            self.path = "/Users/paul/Documents/Modern Portfolio Theory Data/Data/"
        self.today = date.today()
        self.formattedToday = self.today.strftime('%Y-%m-%d')
        pass

    def Refresh(self, fileName, symbol, separator):
        isinDf = []
        fullData = []
        df = pd.read_csv(self.path + fileName, on_bad_lines='skip', encoding_errors='ignore', sep=separator)
        isin = df.iloc[:, df.columns.get_loc(symbol)].tolist()
        for i in isin:
            isinDf = yf.download(i, start="2015-01-01", interval="1d")
            if len(isinDf) > 0:
                fullData.append(isinDf['Close'])
        finalData = pd.concat(fullData, axis=1)
        finalData.to_pickle(self.path + fileName.replace(".csv", "") + ' ' + self.formattedToday + '.pkl')
        finalData.to_csv(self.path + fileName.replace(".csv", "") + ' ' + self.formattedToday + '.csv')

    def CheckTimeSeries(self, fileName):
        df = pd.read_pickle(self.path + fileName)
        returns = df.dropna(thresh=int(df.shape[0] * 0.8), axis=1)
        df.to_pickle(self.path + fileName)
        pass

    def FromList(self, isin, fileName):
        fullData = []
        for i in isin:
            isinDf = yf.download(i, start="2015-01-01", interval="1d")
            if len(isinDf) > 0:
                fullData.append(isinDf['Close'])
        finalData = pd.concat(fullData, axis=1)
        finalData.to_pickle(self.path + fileName + self.formattedToday + '.pkl')
        finalData.to_csv(self.path + fileName + '.csv')

refresh = Refresh()
#refresh.CheckTimeSeries("ETF  Bonds Fixed Income CHF 2025-05-08.pkl")
#refresh.Refresh('flat-ui__data-Thu May 22 2025.csv', 'Symbol', ',')
refresh.FromList(assets.EuroSTOXX50, 'FTSE 100')