# =================================================================
# PROJECT: Vietnam Stock Price Predictor 
# PURPOSE: Build and train the model
# AUTHOR: Đỗ Mạnh Hoàng
# DATE: 2026
# =================================================================
import streamlit as st
from HStockDatabase import HStockDatabase

# =================================================================
# The Vietnam Stock Market Predict model - Download CSV UI
# =================================================================
class HDownLoadCSVUI:
    # =================================================================
    # Default constructor
    # =================================================================
    def __init__(self) -> None:
        self.db = HStockDatabase()

    # =================================================================
    # Launch the download CSV UI
    # =================================================================
    def launch(self):
        # setup main page
        st.set_page_config(page_title="Vietnam Stock Price Predictor - Download Stock Data" , page_icon=":material/download:", layout="wide")
        with st.form("download_csv_form"):
            st.write("Download Stock Data")
            ticker = st.selectbox(label="Select Ticker", options=self.db.getTickers())
            fromDate = st.date_input(label="From Date", format="YYYY-MM-DD")
            toDate = st.date_input(label="To Date", format="YYYY-MM-DD")
            if st.form_submit_button("Download CSV"):
                with st.spinner("Loading...", show_time=True):
                    df = self.db.load(ticker, f"{fromDate}", f"{toDate}")
                    st.write(df)

# =================================================================
# Main method
# =================================================================
if __name__ == "__main__":
    hDownLoadCSVUI = HDownLoadCSVUI()
    hDownLoadCSVUI.launch()