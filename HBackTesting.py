# =================================================================
# PROJECT: Vietnam Stock Price Predictor - Data Pipeline
# PURPOSE: Build and train the model
# AUTHOR: Đỗ Mạnh Hoàng
# DATE: 2026
# =================================================================
import streamlit as st
from HStockDatabase import HStockDatabase
from HVnStockPredictModel import HVnStockPredictModel
import matplotlib.pyplot as plt

# =================================================================
# The Vietnam Stock Market Predict model - Back Testing UI
# =================================================================
class HBackTesting:
    def launch(self):
        # setup main page
        db = HStockDatabase()
        hVnStockPredictModel = HVnStockPredictModel()
        st.set_page_config(page_title="Vietnam Stock Price Predictor - Back Testing" , page_icon=":material/vital_signs:", layout="wide")
        with st.form("back_testing_form"):
            st.write("Back Testing")
            ticker = st.selectbox(label="Select Ticker", options=db.getTickers())
            #fromDate = st.date_input(label="Historical Data From Date", format="YYYY-MM-DD")
            #toDate = st.date_input(label="Historical Data To Date", format="YYYY-MM-DD")
            if st.form_submit_button("Back Testing"):
                with st.spinner("Loading...", show_time=True):
                    # load model, scaler and history
                    model, scaler, history = hVnStockPredictModel.loadModel(ticker)
                    # load data
                    df = db.load(ticker, "", "")
                    # run back testing in 50 days
                    results = hVnStockPredictModel.backtest(model, scaler, hVnStockPredictModel.prepareScalerData(df, scaler), df)
                    # draw back testing chart
                    plt = hVnStockPredictModel.drawBackTestingChart(results)
                    st.pyplot(plt, use_container_width=True) # Display the figure

# =================================================================
# Main method
# =================================================================
if __name__ == "__main__":
    hBackTesting = HBackTesting()
    hBackTesting.launch()
