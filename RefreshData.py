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
        self.df = pd.read_csv(self.path2 + "Assets Description.csv", sep=",", index_col=False)
        self.today = date.today()
        self.formattedToday = self.today.strftime('%Y-%m-%d')
        pass

    def Refresh(self, fileName):
        df = pd.read_csv(self.path + fileName, on_bad_lines='skip', encoding_errors='ignore', sep=";")
        isin = df.iloc[:, df.columns.get_loc('ISIN')].tolist()
        for i in isin:
            isinDf = yf.download(i, start="2000-01-01", end=self.formattedToday, interval="1d")
            fullData = pd.concat([df, isinDf], axis=0)
        fullData.to_pickle(self.path + fileName.replace(".csv", ".pkl"))
        fullData.to_csv(self.path + fileName)


refresh = Refresh()
refresh.Refresh('SMI.csv')



    