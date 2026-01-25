# =================================================================
# PROJECT: Vietnam Stock Price Predictor
# PURPOSE: Fetch, clean, and prepare HOSE/HNX data for LSTM training
# AUTHOR: Đỗ Mạnh Hoàng
# DATE: 2026
# =================================================================
from HMLDataHelper import HMLDataHelper
from HStockDatabase import HStockDatabase
import pandas as pd
from vnstock import Listing
import pathlib
from datetime import datetime, timedelta

class HCrawlStockData:
    # =================================================================
    # Default constructor
    # =================================================================
    def __init__(self) -> None:
        self.db = HStockDatabase()
        self.originalStartDate = "2000-07-20" # beginning of Vietnam stock market
        self.now = datetime.now().date()
        self.nowString = self.now.strftime('%Y-%m-%d')
        self.csvFolder = "./csv"

    # =================================================================
    # Load data from Vnstock and save to CSV file and DB
    # =================================================================
    def crawData(self) -> None:
        # crawl laest company
        self.crawlCompanyToDb()

        # crawl online data store to csv and db
        self.crawlStockDataOnlineToCSVDB(False)

    # =================================================================
    # Load data from Vnstock and save to CSV file and DB
    # =================================================================
    def retryCrawData(self) -> None:
        # crawl laest company
        self.crawlCompanyToDb()

        # crawl online data store to csv and db
        self.crawlStockDataOnlineToCSVDB(True)

    # =================================================================
    # Load data from Vnstock and save to CSV file and DB
    # =================================================================
    def crawlStockDataOnlineToCSVDB(self, identifyDistinct:bool):
        # get the ticker list
        if(identifyDistinct):
            tickers = self.db.getDistinctTickers()
        else:
            tickers = self.db.getTickers()
            if(tickers.empty): # if stocks table is empty - get from company table
                tickers = self.db.getDistinctTickers()

        # process crawl data
        hMLDataHelper = HMLDataHelper()
        for ticker in tickers:
            try:
                # compute the start and end date
                dfCrawlDate = self.db.getCrawl(ticker)
                toDate = self.nowString
                if(dfCrawlDate.empty):
                    fromDate = self.originalStartDate
                else:
                    fromDate = dfCrawlDate["CrawlDate"].iloc[0]

                # process crawl data and store to CSV
                print(f"Start loading data from ticker: {ticker} within from: {fromDate} to {toDate} and store to CSV file")
                df = hMLDataHelper.readDataFromVnstock(ticker, fromDate, toDate)
                df['time'] = pd.to_datetime(df['time'], format="YYYY-MM-DD")
                dataFile = f"csv/{ticker}_{fromDate}_{toDate}.csv"
                df.to_csv(dataFile, index=False, encoding='utf-8-sig') # save to csv file for back up
                print(f"Successful store ticker: {ticker} to CSV")

                # save stock data to database
                print(f"Save stock: {ticker} data into database.")
                self.db.save(df, ticker, columns= ["Time", "Open", "High", "Low", "Close", "Volume"])

                # save crawl data into database
                print(f"Save crawl data of ticker: {ticker} data into database.")
                self.db.saveCrawlData(ticker, toDate)
                print(df)
            except Exception as e:
                print(f"Skipping {ticker} due to error: {e}")
                continue  # Moves to the next stock in the list

    # =================================================================
    # Load data from CSV and add new to database
    # =================================================================
    def cleanUpDBAndSynDataFromCSV(self):
        # Convert string path to a Path object
        path = pathlib.Path(self.csvFolder)

        # Iterate through all files ending in .csv
        dfAllStock = pd.DataFrame()
        for file in path.glob("*.csv"):
            # Extract first 3 characters from the filename
            ticker = file.stem[:3]
            print(f"Reading file: {file.name} - ticker: {ticker}...")

            # Read the CSV and store it in the dictionary
            dfCSV = pd.read_csv(file)

            # normalize dfCSV by adding ticker colum + standardize column name
            dfCSV["Ticker"] = ticker.upper()
            dfCSV.columns = ["Time", "Open", "High", "Low", "Close", "Volume", "Ticker"]
            dfAllStock = pd.concat([dfAllStock, dfCSV])

            # update crawl date
            #toDate = datetime.now().date().strftime('%Y-%m-%d')
            toDate = "2026-01-19"
            self.db.saveCrawlData(ticker, toDate)

        # remove duplicate keys
        print(f"Remove all duplicate key in data frame...")
        dfAllStock = dfAllStock.drop_duplicates(subset=["Time", "Ticker"], keep="last") ## drop duplicate to avoid primary key violation

        # save to database
        print(f"Save data frame into database")
        self.db.saveAllStocks(dfAllStock, columns = ["Time", "Open", "High", "Low", "Close", "Volume", "Ticker"])
        print(f"Successfull save all stocks into database with data frame\n{dfAllStock}")

    # =================================================================
    # List out all stock on the market
    # =================================================================
    def listStockId(self):
        # Initialize the Listing object
        ls = Listing()

        # 1. Get all stock symbols across all exchanges (HOSE, HNX, UPCOM)
        return ls.all_symbols()

    # =================================================================
    # Crawl all stock on the market and store to database
    # =================================================================
    def crawlCompanyToDb(self):
        df = self.listStockId()
        self.db.saveCompany(df)

# =================================================================
# Main method
# =================================================================
if __name__ == "__main__":
    #1. Initialize the Object
    hCrawlStockData = HCrawlStockData()

    #2. crawl data
    hCrawlStockData.crawData()

    #3. retry crawl data
    #hCrawlStockData.retryCrawData()

    #4. clean up data in database and sync CSV to DB
    #hCrawlStockData.cleanUpDBAndSynDataFromCSV