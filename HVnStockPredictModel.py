# =================================================================
# PROJECT: Vietnam Stock Price Predictor
# PURPOSE: Build and train the model
# AUTHOR: Đỗ Mạnh Hoàng
# DATE: 2026
# =================================================================
from HStockDatabase import HStockDatabase
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# =================================================================
# The Vietnam Stock Market Predict model
# =================================================================
class HVnStockPredictModel:
    # =================================================================
    # Default constructor
    # =================================================================
    def __init__(self) -> None:
        self.db = HStockDatabase()
        self.features = ["Close", "MA20", "RSI", "Volume"]
        self.numberOfSamplingDayToPredict = 60
        self.predictionDays = 7
        #self.fromDate = "2000-07-20" # beginning of Vietnam stock market
        self.now = datetime.now().date()
        self.nowString = self.now.strftime('%Y-%m-%d')

    # =================================================================
    # Compute RSI and MA (20/50)
    # =================================================================
    def computeRSI_MA(self, df: pd.DataFrame) -> pd.DataFrame:
        # ==========================================
        # Step #1: compute RSI
        # ==========================================
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df["RSI"] = 100 - (100 / (1 + (gain/loss)))

        # ==========================================
        # Step #2: compute MA20
        # ==========================================
        maWindow = 20
        maStr = f"MA{maWindow}"
        df[maStr] = df["Close"].rolling(window=maWindow).mean()

        # ==========================================
        # Step #2: compute MA50
        # ==========================================
        maWindow = 50
        maStr = f"MA{maWindow}"
        df[maStr] = df["Close"].rolling(window=maWindow).mean()

        # ==========================================
        # Step #3: prepare full data OHLC, RSI and MA for chart drawing and model training
        # ==========================================
        df = df.reset_index() # reset index to return Time column
        df.columns = ["Time", "Open", "High", "Low", "Close", "Volume", "RSI", "MA20", "MA50"]

        return df

    # =================================================================
    # This method is help to load training data from data base
    # train the model and save it to model file
    # =================================================================
    def prepareData(self, ticker: str, fromDate: str, toDate:str) -> pd.DataFrame:
        # ==========================================
        # Step #1: load data from database
        # ==========================================
        df = self.db.load(ticker, fromDate, toDate)
        df = df.reset_index() # reset index to return the Time column back to normal column

        # ==========================================
        # Step #2: compute RSI and MA
        # ==========================================
        df = self.computeRSI_MA(df) ## compute RSI, MA20 & MA50

        return df

    # =================================================================
    # This method is help to create X and Y array
    # =================================================================   
    def createSequences(self, scaledData: pd.DataFrame, closeIndex: int, predictionDays: int, futureDays: int) -> any:
        X, y = [], []
        # moving from predictionDays position in dfClose to length of it - for future days
        for i in range(predictionDays, len(scaledData) - futureDays):
            X.append(scaledData[i-predictionDays:i, :]) # get an array of predictionDays from features array value Close, MA20, MA50
            y.append(scaledData[i:i+futureDays, closeIndex]) #get the y as result of futureDays start from the next comming predictionDays position
        return np.array(X), np.array(y)

    # =================================================================
    # This method is help to get the model name
    # =================================================================   
    def getModelFile(self, ticker: str) -> str:
        return f"./model/{ticker}_lstm_model.keras"

    # =================================================================
    # This method is help to get the scaler name
    # =================================================================   
    def getScalerFile(self, ticker: str) -> str:
        return f"./model/{ticker}_scaler.scaler"

    # =================================================================
    # This method is help to get the history file name
    # =================================================================   
    def getHistoryFile(self, ticker: str) -> str:
        return f"./model/{ticker}_history.csv"

    # =================================================================
    # This method is help to train the model and save it to model file
    # and scaler file
    # =================================================================
    def trainModel(self, df: pd.DataFrame, features: list, numberOfSamplingDayToPredict: int, predictionDays: int) -> any:
        df.dropna(inplace=True) # remove N/A data field, for example first 14 days of RSI or 20 day of MA20, 50 days of MA50
        # Scale Features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[features])

        # build sequences using Close price
        X_train, y_train = self.createSequences(scaled_data, features.index("Close"), numberOfSamplingDayToPredict, predictionDays)

        # Build LSTM Model
        model = Sequential([
            LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(units=64, return_sequences=False),
            Dropout(0.2),
            LSTM(units=32, return_sequences=False),
            Dropout(0.2),
            Dense(units=predictionDays) # Output layer predicts {predictionDays} days at once
        ])

        # Gradient Descent Optimization
        # Adjust Learning Rate (decending)
        def lr_scheduler(epoch, lr):
            if epoch < 10:
                return lr
            else:
                return lr * np.exp(-0.1)

        # Customize Adam Optimizer (Gradient Descent)
        custom_adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

        # Early Stopping: if loss didn't reduce to avoid Overfitting
        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model.compile(optimizer=custom_adam, loss='mean_squared_error')
        
        # Train Model
        print("Initial Training...")
        history = model.fit(X_train, y_train, epochs=50, 
            batch_size=32, 
            callbacks=[LearningRateScheduler(lr_scheduler), early_stop],
            verbose=1
        )
        
        # return model and scaler
        return model, scaler, history

    # =================================================================
    # This method is help to save model, scaler and history to file
    # =================================================================
    def save(self, ticker: str, model: Sequential, scaler: MinMaxScaler, history: dict) -> None:
        model.save(self.getModelFile(ticker))
        joblib.dump(scaler, self.getScalerFile(ticker))
        pd.DataFrame(history.history).to_csv(self.getHistoryFile(ticker), index=False, encoding='utf-8-sig')
        print(f"Model and Scaler saved for {ticker}")

    # =================================================================
    # This method is help to load model, scaler and history from file
    # =================================================================
    def loadModel(self, ticker: str) -> any:
        model = load_model(self.getModelFile(ticker))
        scaler = joblib.load(self.getScalerFile(ticker))
        history = pd.read_csv(self.getHistoryFile(ticker)).to_dict(orient='list')
        return model, scaler, history

    # =================================================================
    # Train model for a stock with all data
    # =================================================================
    def trainModelFromTicker(self, ticker: str) -> any:
        # prepare data
        print(f"Load stock: {ticker} data from database")
        df = self.prepareData(ticker, "", "") # load all data in data base

        # train the model
        model, scaler, history = self.trainModel(df, self.features, self.numberOfSamplingDayToPredict, self.predictionDays)

        # Save Model and Scaler to files
        self.save(ticker, model, scaler, history)

        # save model, scaler file to database
        modelFile = self.getModelFile(ticker)
        scalerFile = self.getScalerFile(ticker)
        historicalFile = self.getHistoryFile(ticker)
        minError = np.mean(history.history['loss'])
        print(f"Save training data into database with Ticcker: {ticker} - Model file: {modelFile} - Scaler file: {scalerFile} - Historical file: {historicalFile} - Error: {minError} - Training Date: {self.nowString}")
        self.db.saveTrainData(ticker, modelFile, scalerFile, historicalFile, minError, self.nowString)

        # return model, scaler and history data for later using
        return model, scaler, history

    # =================================================================
    # Draw loss chart
    # =================================================================
    def drawLossChart(self, history):
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['learning_rate'], label='Learning Rate')
        plt.title('Model Convergence')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE) / Learning Rate')
        plt.legend()
        return plt

    # =================================================================
    # Prepare latest data for prediction
    # =================================================================
    def prepareLatestDataForPrediction(self, df: pd.DataFrame, scaler) -> pd.DataFrame:
        # ==========================================
        # Step #1: reset index
        # ==========================================
        df = df.reset_index() # reset index to return the Time column back to normal column

        # ==========================================
        # Step #2: compute RSI and MA
        # ==========================================
        df = self.computeRSI_MA(df) ## compute RSI, MA20 & MA50
        df = df.dropna() # load data and drop NA field

        # scale data and return
        scaled_data = scaler.fit_transform(df[self.features])
        recent_n_days = scaled_data[-self.predictionDays:]
        return np.reshape(recent_n_days, (1, self.predictionDays, len(self.features)))

    # =================================================================
    # Future Prediction Function
    # =================================================================
    def predict(self, df: pd.DataFrame, model, scaler):
           # Predict 7 days (scaled)
        prediction_scaled = model.predict(self.prepareLatestDataForPrediction(df, scaler)) # Shape: (1, 7)
        tmp_matrix = np.zeros((self.predictionDays, len(self.features)))
        closeIndex = self.features.index("Close")
        tmp_matrix[:, closeIndex] = prediction_scaled[0] # get the N prediction price
        return scaler.inverse_transform(tmp_matrix)[:, closeIndex]

    # =================================================================
    # Prepare scaler data
    # =================================================================
    def prepareScalerData(self, df: pd.DataFrame, scaler: MinMaxScaler) -> any:
        dfNew = df.copy()
        dfNew = self.computeRSI_MA(dfNew) ## compute RSI, MA20 & MA50
        dfNew = dfNew.dropna() # load data and drop NA field
        return scaler.fit_transform(dfNew[self.features])

    # =================================================================
    # Back testing the model
    # =================================================================
    def backtest(self, model: Sequential, scaler: MinMaxScaler, scaled_data: any, original_df: pd.DataFrame, test_days: int=60):
        predictionsList = []
        realPrices = []
        original_df = original_df.reset_index()
        dates = original_df["Time"][-test_days:]
        closeIndex = self.features.index("Close")

        for i in range(test_days, 0, -1):
            endIdx = len(scaled_data) - i
            startIdx = endIdx - self.numberOfSamplingDayToPredict
            inputWindow = scaled_data[startIdx:endIdx, :]
            inputReshaped = np.reshape(inputWindow, (1, self.numberOfSamplingDayToPredict, scaled_data.shape[1]))

            # predict price
            predScaled = model.predict(inputReshaped, verbose=0)

            # get the predict data
            tmp_matrix = np.zeros((self.predictionDays, scaled_data.shape[1]))
            tmp_matrix[:, closeIndex] = predScaled[0]
            final_prices = scaler.inverse_transform(tmp_matrix)[:, closeIndex]

            predictionsList.append(final_prices[0]) # save the predict price to predict list
            realPrices.append(original_df['Close'].iloc[endIdx]) # save the real price to real price list
        
        # combine all data into 1 single data frame
        results = pd.DataFrame({
        'Date': dates,
        'Real': realPrices,
        'Predicted': predictionsList
        })

        # compute the error between predict and real price
        results['Error_VNĐ'] = abs(results['Real'] - results['Predicted'])

        return results

    # =================================================================
    # Draw the back testing chart
    # =================================================================
    def drawBackTestingChart(self, results: pd.DataFrame) -> plt:
        # compute mae
        mae = results['Error_VNĐ'].mean()

        # draw chart
        plt.figure(figsize=(15, 7))
        plt.plot(results['Date'], results['Real'], label='Price', color='blue', linewidth=2)
        plt.plot(results['Date'], results['Predicted'], label='Prediction Price', color='red', linestyle='--', alpha=0.8)
        plt.title('Back testing result: Real vs Predict (T+1)')
        plt.xlabel('Date')
        plt.ylabel('Price (VNĐ)')
        plt.legend()
        plt.grid(True)
        return plt

# =================================================================
# Test this class
# =================================================================
if __name__ == "__main__":
    modelVnStockPredict = HVnStockPredictModel() ## VN Stock Predict model object

    # define train meta data
    #ticker: str = "VNM"
    ticker: str = "FPT"
    #ticker: str = "VCB"
    #ticker: str = "MSN"

    # train
    model, scaler, history = modelVnStockPredict.trainModelFromTicker(ticker)

    # draw loss chart
    modelVnStockPredict.drawLossChart(history)
    #print(history.history)

    # predict
    predictDf = modelVnStockPredict.predict(ticker, model, scaler)
    print(predictDf)
