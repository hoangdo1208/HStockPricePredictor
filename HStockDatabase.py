# =================================================================
# PROJECT: Vietnam Stock Price Predictor
# PURPOSE: Fetch, clean, and prepare HOSE/HNX data for LSTM training
# AUTHOR: Đỗ Mạnh Hoàng
# DATE: 2026
# =================================================================
import pandas as pd
from sqlalchemy import create_engine, text

# =================================================================
# The embedded database which help to store stock data to prevent
# too many call on internet to get data and prevent error also
# =================================================================
class HStockDatabase:
    # =================================================================
    # Compute connect to DB engine in the contructor
    # =================================================================
    def __init__(self, db_name="data/hstockdatabase.dat"):
        """Initializes the database connection and creates necessary indexes."""
        self.engine = create_engine(f'sqlite:///{db_name}')
        self._prepare_database()

    # =================================================================
    # Prepare for database
    # =================================================================
    def _prepare_database(self) -> None:
        """Internal method to optimize the database for Ticker/Date queries."""
        with self.engine.connect() as conn:
            # Explicitly create the tables schema if it doesn't exist
            # stock table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS stocks (
                    Time TEXT,
                    Ticker TEXT,
                    Open FLOAT,
                    High FLOAT,
                    Low FLOAT,
                    Close FLOAT,
                    Volume INTEGER,
                    PRIMARY KEY (Time, Ticker)
                );
            """))
            # company table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS company (
                    Ticker TEXT,
                    Name TEXT,
                    PRIMARY KEY (Ticker)
                );
            """))
            # train table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS train (
                    Ticker TEXT,
                    ModelFile TEXT,
                    ScalerFile TEXT,
                    HistoricalFile TEXT,
                    MeanAbsoluteError FLOAT,
                    TrainDate TEXT,
                    PRIMARY KEY (Ticker)
                );
            """))
            # crawl table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS crawl (
                    Ticker TEXT,
                    CrawlDate TEXT,
                    PRIMARY KEY (Ticker)
                );
            """))
            # commit to database
            conn.commit()

    # =================================================================
    # Save stock data frame, ticker into database
    # Stock data frame index must not duplicate with the one in database
    # =================================================================
    def save(self, df: pd.DataFrame, ticker: str, columns = ["Time", "Open", "High", "Low", "Close", "Volume"]) -> None:
        """
        Stores OHLCV data. 
        """
        # Save to SQL
        df["Ticker"] = ticker.upper()
        with self.engine.connect() as conn:
            for row in df.itertuples(index=False):
                # Generating the string
                columns = ', '.join(df.columns)
                # Using repr() to handle strings/quotes automatically
                clean_values = [str(val) if isinstance(val, pd.Timestamp) else val for val in row]
                values = ', '.join([repr(v) for v in clean_values])
                sql = f"INSERT OR REPLACE INTO stocks ({columns}) VALUES ({values});"
                conn.execute(text(sql))
            
            # Commit after the loop finishes for better performance
            conn.commit()
        print(f"Successfully stored ticker: {ticker} into database with data frame below.\n{df}")

    # =================================================================
    # Save all stocks from data frame into database. All old stocks
    # data will be deleted
    # =================================================================
    def saveAllStocks(self, df: pd.DataFrame, columns = ["Time", "Ticker", "Open", "High", "Low", "Close", "Volume"]) -> None:
        """
        Stores OHLCV data. 
        """
        # Ensure we don't modify the original dataframe
        temp_df = df.copy()
        
        # Ensure Ticker column exists
        temp_df.columns = columns
        temp_df = temp_df.set_index(["Time", "Ticker"])
        
        # Save to SQL
        temp_df.to_sql('stocks', self.engine, if_exists='replace', index=True, chunksize=1000, method='multi')
        print(f"Successfully stored all stocks into database with data frame below.\n{temp_df}")

    # =================================================================
    # Load stock data from database - filter by ticker
    # =================================================================
    def load(self, ticker: str, fromDate: str, toDate: str) -> pd.DataFrame:
        """
        Retrieves stock data from the database with optional date filtering.
        """
        query = "SELECT * FROM stocks WHERE Ticker = :ticker"
        params = {"ticker": ticker.upper()}

        if fromDate:
            query += " AND Time >= :fromDate"
            params["fromDate"] = fromDate
        
        if toDate:
            query += " AND Time <= :toDate"
            params["toDate"] = toDate

        # Execute query
        df = pd.read_sql(
            text(query), 
            self.engine, 
            params=params, 
            index_col= ["Time"]
        )

        # drop Ticker columns since we have only 1 ticker in this data set
        return df.drop(columns=["Ticker"])

    # =================================================================
    # Load stock data from database - filter by ticker list
    # =================================================================
    def loadStocks(self, tickers: list, fromDate: str, toDate: str) -> pd.DataFrame:
        """
        Retrieves stock data from the database with optional date filtering.
        """
        tickerList =  ", ".join([f"'{ticker}'" for ticker in tickers])
        query = f"SELECT * FROM stocks WHERE Ticker IN ({tickerList})"

        if fromDate:
            query += f" AND Time >= '{fromDate}'"
        
        if toDate:
            query += f" AND Time <= '{toDate}'"

        # Execute query
        print(f"Query = {query}")
        return pd.read_sql(
            text(query), 
            self.engine, 
            index_col= ["Time", "Ticker"]
        )

    # =================================================================
    # Load stock data from database
    # =================================================================
    def loadAllStock(self) -> pd.DataFrame:
        """
        Retrieves stock data from the database with optional date filtering.
        """
        query = "SELECT * FROM stocks"

        # Execute query
        return pd.read_sql(
            text(query), 
            self.engine,
            index_col= ["Time", "Ticker"]
        )

    # =================================================================
    # Empty stocks table
    # =================================================================
    def emptyStock(self) -> None:
        with self.engine.connect() as conn:
            conn.execute(text("DELETE FROM stocks;"))
            conn.commit()

    # =================================================================
    # This helper method is used to list all ticker
    # =================================================================
    def getTickers(self) -> pd.DataFrame:
        query = "SELECT DISTINCT Ticker FROM stocks"
        df = pd.read_sql(
            text(query), 
            self.engine
        )
        return df["Ticker"]

    # =================================================================
    # This helper method is used to list all ticker which doesn't exist
    # in stocks
    # =================================================================
    def getDistinctTickers(self) -> pd.DataFrame:
        query = "SELECT Ticker FROM company"
        df1 = pd.read_sql(
            text(query), 
            self.engine
        )
        query = "SELECT DISTINCT Ticker FROM stocks"
        df2 = pd.read_sql(
            text(query), 
            self.engine
        )
        df = df1[~df1["Ticker"].isin(df2["Ticker"])]
        return df["Ticker"]

    # =================================================================
    # Save stock data frame, ticker into database
    # =================================================================
    def saveCompany(self, df: pd.DataFrame, columns = ["Ticker", "Name"]) -> None:
        """
        Stores Ticker and Name of each company
        """
        # Ensure we don't modify the original dataframe
        temp_df = df.copy()
        temp_df.columns = columns
        
        # Ensure Ticker column exists
        temp_df = temp_df.set_index('Ticker')
        
        # Save to SQL
        # 'append' allows adding data for different tickers to the same table
        temp_df.to_sql('company', self.engine, if_exists='replace', index=True)
        print(f"Successfully stored companies into database with data frame below.\n{temp_df}")

    # =================================================================
    # Save Crawl data
    # =================================================================
    def saveCrawlData(self, ticker: str, crawlDate: str) -> None:
        with self.engine.connect() as conn:
            query = f"INSERT OR REPLACE INTO crawl(Ticker, CrawlDate) "
            query += f"VALUES ('{ticker}', '{crawlDate}');"
            conn.execute(text(query))
            conn.commit()
        print(f"Successfully stored crawl data into database with data frame below.")

    # =================================================================
    # Save Train data
    # =================================================================
    def saveTrainData(self, ticker: str, modelFile: str, scalerFile: str, historicalFile:str, meanAbsoluteError: float, trainDate: str) -> None:
        with self.engine.connect() as conn:
            query = f"INSERT OR REPLACE INTO train(Ticker, ModelFile, ScalerFile, HistoricalFile, MeanAbsoluteError, TrainDate) "
            query += f"VALUES ('{ticker}', '{modelFile}', '{scalerFile}', '{historicalFile}', {meanAbsoluteError}, '{trainDate}');"
            conn.execute(text(query))
            conn.commit()
        print(f"Successfully stored train data into database with data frame below.")

    # =================================================================
    # This helper method is used to get the train data
    # =================================================================
    def getTrain(self, ticker: str) -> pd.DataFrame:
        query = "SELECT * FROM train WHERE Ticker = :ticker"
        params = {"ticker": ticker.upper()}
        df = pd.read_sql(
            text(query), 
            self.engine,
            params=params,
            index_col= ["Ticker"]
        )
        return df

    # =================================================================
    # This helper method is used to get the crawl data
    # =================================================================
    def getCrawl(self, ticker: str) -> pd.DataFrame:
        query = "SELECT * FROM crawl WHERE Ticker = :ticker"
        params = {"ticker": ticker.upper()}
        df = pd.read_sql(
            text(query), 
            self.engine,
            params=params,
            index_col= ["Ticker"]
        )
        return df

    # =================================================================
    # This helper method is used to get all of the crawl data
    # =================================================================
    def getAllCrawl(self) -> pd.DataFrame:
        query = "SELECT * FROM crawl"
        df = pd.read_sql(
            text(query), 
            self.engine,
            index_col= ["Ticker"]
        )
        return df

# =================================================================
# Main method
# =================================================================
if __name__ == "__main__":
    # Load data with date criteria
    # Example: Load only January 1st
    db = HStockDatabase()

    # load all stocks
    #df = db.loadAllStock()

    # load 1 stock within a duration
    #df = db.load("FPT", "2025-01-01", "2026-01-20")
    #df = db.load("FPT", "", "")

    # load list of stock within a duration
    df = db.loadStocks(["FPT", "VNM", "VIC"], "2025-01-01", "2026-01-20")

    # load crawl data
    #df = db.getCrawl("FPT")

    # load train data
    #df = db.getTrain("FPT")

    # get list of ticket
    #df = db.getTickers()

    # get the distint tickers between company and stocks table
    #df = db.getDistinctTickers()

    # show data frame and describe
    print("\n--- Loaded Data ---")
    print(df)
    print(df.info())