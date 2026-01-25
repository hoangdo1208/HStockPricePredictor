# =================================================================
# PROJECT: Vietnam Stock Price Predictor - Data Pipeline
# PURPOSE: Build and train the model
# AUTHOR: Đỗ Mạnh Hoàng
# DATE: 2026
# =================================================================
import streamlit as st
from HVnStockPredictModel import HVnStockPredictModel
from HStockDatabase import HStockDatabase
import mplfinance as mpf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential

# =================================================================
# The Vietnam Stock Market Predict model - Train Model UI
# =================================================================
class HTrainModelUI:
    def launch(self):
        # setup main page
        db = HStockDatabase()
        hVnStockPredictModel = HVnStockPredictModel()
        st.set_page_config(page_title="Vietnam Stock Price Predictor - Train Model" , page_icon=":material/model_training:", layout="wide")
        with st.form("train_model_form"):
            st.write("Train Model & Predict")
            ticker = st.selectbox(label="Select Ticker", options=db.getTickers())
            if st.form_submit_button("Train Model"):
                # train the model
                st.write("Start training model...")
                with st.spinner("Loading...", show_time=True):
                    model, scaler, history = hVnStockPredictModel.trainModelFromTicker(ticker)

                # draw history plot chart
                mpl = hVnStockPredictModel.drawLossChart(history)
                st.pyplot(mpl, use_container_width=True) # Display the figure

# =================================================================
# Main method
# =================================================================
if __name__ == "__main__":    
    hTrainModelUI = HTrainModelUI()
    hTrainModelUI.launch()