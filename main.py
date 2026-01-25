# =================================================================
# PROJECT: Vietnam Stock Price Predictor
# PURPOSE: Fetch, clean, and prepare HOSE/HNX data for LSTM training
# AUTHOR: Đỗ Mạnh Hoàng
# DATE: 2026
# =================================================================
import sys
from streamlit.web import cli as stcli
from streamlit import runtime
from HCrawlStockData import HCrawlStockData
from HVnStockPredictModel import HVnStockPredictModel
from HStockDatabase import HStockDatabase
from HVnStockPredictModel import HVnStockPredictModel

# =================================================================
# The HStockPricePredictor main class
# =================================================================
class HStockPricePredictor:
    # =================================================================
    # Default constructor
    # =================================================================
    def __init__(self) -> None:
        self.hCrawlStockData = HCrawlStockData()
        self.hVnStockPredictModel = HVnStockPredictModel()
        self.db = HStockDatabase()
        self.hVnStockPredictModel = HVnStockPredictModel()

    # =================================================================
    # Process arguments
    # =================================================================
    def processArguments(self, arg: str) -> None:
        match arg:
            # crawl data from online and store to CSV + DB
            case "CrawlData":
                print(f"Start to crawl data online and save to CSV + database")
                self.hCrawlStockData.crawData()

            # retry crawl data from online and store to CSV + DB in case of any stock got error last time call
            case "RetryCrawlData":
                print(f"Retry to crawl data online and save to CSV + database")
                self.hCrawlStockData.retryCrawData()

            # retry crawl data from online and store to CSV + DB in case of any stock got error last time call
            case "SyncCrawlData":
                print(f"Sync data between CSV files and database")
                self.hCrawlStockData.cleanUpDBAndSynDataFromCSV()

            # train model, each stock will have 1 model file, 1 scaler file and 1 history file
            case "Train":
                print(f"Start to train model for all stocks in database")
                for ticker in self.db.getTickers():
                    print(f"Training stock: {ticker}...")
                    self.hVnStockPredictModel.trainModelFromTicker(ticker)

            # run the HStockPricePredictor application
            case _:
                print(f"Starting web application server")
                sys.argv = ["streamlit", "run", "HStockPricePredictorUI.py"] 
                sys.exit(stcli.main())

    # =================================================================
    # launch the HStockPricePredictor application
    # =================================================================
    def launch(self) -> None:
         for arg in sys.argv:
              self.processArguments(arg)

# =================================================================
# Main method
# =================================================================
if __name__ == "__main__":
        print(f"Start to run the HStockPricePredictor application")
        hStockPricePredictor = HStockPricePredictor()
        hStockPricePredictor.launch()