# =================================================================
# PROJECT: Vietnam Stock Price Predictor - Data Pipeline
# PURPOSE: Build and train the model
# AUTHOR: Đỗ Mạnh Hoàng
# DATE: 2026
# =================================================================
import streamlit as st

# =================================================================
# The Vietnam Stock Market Predict Application
# =================================================================
class HStockPricePredictorUI:
    # =================================================================
    # Constructor the Stock Market Predict Application
    # =================================================================
    def __init__(self):
        # setup main page
        st.set_page_config(page_title="Vietnam Stock Price Predictor", layout="wide")

        # setup page layout and navigation
        with st.spinner("Loading...", show_time=True):
            downloadCSVUI = st.Page("HDownLoadCSVUI.py", title="Download Stock Data", icon=":material/download:")
            stockPriceChart = st.Page("HStockPriceChart.py", title="Vietnam Stock Price", icon=":material/candlestick_chart:")
            trainModelUI = st.Page("HTrainModelUI.py", title="Train Model", icon=":material/model_training:")
            backTesting = st.Page("HBackTesting.py", title="Back Testing", icon=":material/vital_signs:")
            pg = st.navigation( {
                "Dashboard": [downloadCSVUI, stockPriceChart],
                "Admin": [trainModelUI, backTesting]
                })
            pg.run()

# =================================================================
# Main method
# =================================================================
if __name__ == "__main__":
    hStockPricePredictorUI = HStockPricePredictorUI()