import PortfolioUtilities as pu
import assets
import yfinance

class GetISINTimeSeries():
    portfolioUtilities = pu.PortfolioUtilities()
    def GetTimeSeries(self, choice, fileName):
        if choice == 1:
            print(self.portfolioUtilities.FindIsin(assets.ETFMSCIWorld, self.portfolioStructure))
        elif choice == 2:
            self.portfolioUtilities.GetAssetsTimeSeries(assets.fonds_obligataires_suisses, "Swiss Bonds.pkl")
        elif choice == 3:
            self.portfolioUtilities.GetTimeSeries(fileName, False)
        elif choice == 4:
            self.portfolioUtilities.GetIsin("C:/Users/paul.milic/Modern Portfolio/ETF Swiss Equity Themes.csv")
        else:
            self.portfolioUtilities.TransformToPickle("C:/Users/paul.milic/Modern Portfolio/ETF Swiss Equity Themes.csv")
        return 0

#getIsin = GetISINTimeSeries()
#getIsin.GetTimeSeries(3, "ETF CHF.csv")
#portfolioUtilities = PortfolioUtilities.PortfolioUtilities()
#portfolioUtilities.GetAssetsTimeSeries(assets.DAX40, "DAX40.pkl")
