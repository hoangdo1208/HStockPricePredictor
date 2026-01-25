# =================================================================
# PROJECT: Vietnam Stock Price Predictor - Data Pipeline
# PURPOSE: Fetch, clean, and prepare HOSE/HNX data for LSTM training
# AUTHOR: Đỗ Mạnh Hoàng
# DATE: 2026
# =================================================================
from vnstock import Vnstock ## load data using Vnstock lib
import pandas as pd ## data processing
import pandas_ta as ta ## help to compute RSI and MA
import numpy as np ## matrix, vector, math, etc processing using GPU
from sklearn.preprocessing import MinMaxScaler ## help to normalize data to scale (0 -> 1)

# =================================================================
# The Machine Learning Data Helper class
# =================================================================
class HMLDataHelper:
    # =================================================================
    # This helper method is used to load online data from Vnstock
    # with the ticker, from date, to date and return data frame
    # Date format: YYYY-MM-DD
    # =================================================================
    def readDataFromVnstock(self, ticker: str, fromDate: str, toDate: str) -> any:
        # Initialize the library
        source = "VCI" ### hard coding source to VCI due to Vnstock can work with VCI only
        vnstock = Vnstock()
        stock = vnstock.stock(symbol=ticker, source=source)

        # Get raw historical data (OHLCV: Open - High - Low - Close - Volume)
        rawData = stock.quote.history(start=fromDate, end=toDate)

        # Fetch raw data to pandas for data processing later
        return pd.DataFrame(rawData);

    # =================================================================
    # This helper method is used to load online data from Vnstock
    # with the ticker, from date, to date and store to data file to
    # data folder
    # Date format: YYYY-MM-DD
    # =================================================================
    def saveDataFromVnstockToFile(self, ticker: str, fromDate: str, toDate: str, dataFolder: str) -> None:
        ## memthod compute the file name base on the format:
        ## dataFolder: input data folder for ex: data
        ## ticker: the ticker for ex: VCB for Vietcombank stock
        ## fromDate: the start date to load data
        ## toDate: the end date to load data
        dataFile = dataFolder + "/" + ticker + "_" + fromDate + "_" + toDate + ".csv"

        # read data online
        dataFrame = self.readDataFromVnstock(ticker, fromDate, toDate)

        # save data to CSV
        dataFrame.to_csv(dataFile, index=False, encoding='utf-8-sig')

    # =================================================================
    # This helper method is used to load data from CSV file and return
    # a data frame
    # =================================================================
    def readDataFromCSV(self, dataFile: str) -> any:
        # pd.read_csv() loads the file and returns a DataFrame object
        return pd.read_csv(dataFile)

# =================================================================
# Test this class
# =================================================================
if __name__ == "__main__":
    mlHelper = HMLDataHelper() ## create ML Data Helper class

    # define meta data
    #ticker: str = "VNM"
    #ticker: str = "FPT"
    #ticker: str = "E1VFVN30"
    #ticker: str = "VCB"
    ticker: str = "MSN"
    fromDate = "2025-01-01"
    toDate = "2025-12-31"
    dataFolder:str = "data/"
    dataFile:str = dataFolder + ticker + "_" + fromDate + "_" + toDate + ".csv"

    ## test extract data online
    #print("Load data from Vnstock")
    #print(mlHelper.readDataFromVnstock(ticker, fromDate, toDate))

    ## test extract data and save file
    print("Load data from Vnstock and save to CSV file")
    mlHelper.saveDataFromVnstockToFile(ticker, fromDate, toDate, "data")

    ## test load data from file
    print("Load data from CSV file")
    dataFrame = mlHelper.readDataFromCSV(dataFile)
    print(dataFrame)