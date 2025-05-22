import pandas as pd
import yfinance as yf
import os
from datetime import date

class Refresh():
    def __init__(self):
        if os.name == "nt":
            self.path = "C:/Users/paul.milic/Modern Portfolio/"
            self.path2 = "C:/Users/paul.milic/Modern Portfolio/Data/"
        else:
            self.path = "/Users/paul/Documents/Modern Portfolio Theory Data/"
        self.today = date.today()
        self.formattedToday = self.today.strftime('%Y-%m-%d')
        pass

    def Refresh(self, fileName):
        isinDf = []
        df = pd.read_csv(self.path + fileName, on_bad_lines='skip', encoding_errors='ignore', sep=";")
        isin = df.iloc[:, df.columns.get_loc('ISIN')].tolist()
        isinDf = yf.download(isin, start="2015-01-01", end=self.formattedToday, interval="1d")
        isinDf = isinDf['Close']
        isinDf.to_pickle(self.path + fileName.replace(".csv", "") + ' ' + self.formattedToday + '.pkl')
        isinDf.to_csv(self.path + fileName.replace(".csv", "") + ' ' + self.formattedToday + '.csv')

    def CheckTimeSeries(self, fileName):
        df = pd.read_pickle(self.path + fileName)
        returns = df.dropna(thresh=int(df.shape[0] * 0.8), axis=1)
        df.to_pickle(self.path + fileName)
        pass

refresh = Refresh()
#refresh.CheckTimeSeries("ETF  Bonds Fixed Income CHF 2025-05-08.pkl")
refresh.Refresh('SMI.csv')
