# =================================================================
# PROJECT: Vietnam Stock Price Predictor
# PURPOSE: Build and train the model
# AUTHOR: Đỗ Mạnh Hoàng
# DATE: 2026
# =================================================================
import streamlit as st
from HStockDatabase import HStockDatabase
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta

# =================================================================
# The Vietnam Stock Price Chart
# =================================================================
class HStockPriceChart:
    # =================================================================
    # Default constructor
    # =================================================================
    def __init__(self) -> None:
        self.db = HStockDatabase()

    # =================================================================
    # Launch the Stock Price Chart
    # =================================================================
    def launch(self) -> None:
        # setup main page
        st.set_page_config(page_title="Vietnam Stock Price Predictor - Vietnam Stock Price Chart" , page_icon=":material/candlestick_chart:", layout="wide")

        ## setup the data filter criterias
        ticker = st.sidebar.selectbox(label="Select Ticker", options=self.db.getTickers())
        fromDate = st.sidebar.date_input(label="From Date", value = pd.to_datetime("2025-01-01"), format="YYYY-MM-DD")
        toDate = st.sidebar.date_input(label="To Date", value = pd.to_datetime("today"), format="YYYY-MM-DD")

        ## load data
        df = self.load_data(ticker, fromDate, toDate)
        if not df.empty:
            ## setup the chart filter criterias
            #chartType = st.sidebar.selectbox("Chart Type", ["candle", "line", "ohlc", "renko"])
            maOptions = [5, 10, 20, 50, 200]
            selectedMas = st.sidebar.multiselect("MA", maOptions, default=[20, 50])
            rsiPeriod = st.sidebar.slider("RSI Period", 5, 30, 14)
            showVolume = st.sidebar.checkbox("Show Volume", value=True)

            # Calculate Indicators MA & RSI
            for ma in maOptions:
                df[f'MA{ma}'] = ta.sma(df['Close'], length=ma)
            df['RSI'] = ta.rsi(df['Close'], length=rsiPeriod)

            # Logic for Volume Color (Green if Close > Open, else Red)
            df['Vol_Color'] = ['green' if close >= open else 'red' for open, close in zip(df['Open'], df['Close'])]

            # --- 3. Create the Combined Chart (3 Rows) ---
            fig = make_subplots(
                rows=3, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.02, 
                row_heights=[0.5, 0.2, 0.2] # 50% Price, 20% Volume, 20% RSI
            )

            # A. Price & MAs (Row 1)
            fig.add_trace(go.Candlestick(
                x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                name="Price"
            ), row=1, col=1)

            colors = {5: 'cyan', 10: 'yellow', 20: 'orange', 50: 'magenta', 200: 'red'}
            for ma in selectedMas:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[f'MA{ma}'], 
                    name=f'MA {ma}', 
                    line=dict(width=1.5, color=colors[ma])
                ), row=1, col=1)

            # B. Volume (Row 2)
            fig.add_trace(go.Bar(
                x=df.index, y=df['Volume'],
                name="Volume",
                marker_color=df['Vol_Color'],
                opacity=0.8
            ), row=2, col=1)

            # C. RSI (Row 3)
            fig.add_trace(go.Scatter(
                x=df.index, y=df['RSI'], 
                name="RSI", 
                line=dict(color='white', width=1)
            ), row=3, col=1)

            # RSI reference lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

            # --- 4. Styling & Interactivity ---
            # Create a descriptive title string
            ma_str = ", ".join([str(x) for x in selectedMas])
            chart_title = f"{ticker} Stock Analysis (MAs: {ma_str if ma_str else 'None'})"
            fig.update_layout(
                title={
                    'text': chart_title,
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 24}
                },
                height=800,
                template="plotly_dark",
                xaxis_rangeslider_visible=False,
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=50, r=50, b=50, t=50)
            )

            # Clean up Y-axis titles
            fig.update_yaxes(title_text="Price (VND)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            fig.update_yaxes(title_text="RSI", row=3, col=1)

            # --- 5. Display ---
            st.plotly_chart(fig, use_container_width=True)

            ## show data
            if st.sidebar.checkbox("Show Data"):
                st.write(df)

    # =================================================================
    # Load ticket data from Database
    # =================================================================
    def load_data(self, ticker, fromDate, toDate) -> any:
        df = self.db.load(ticker, fromDate, toDate)
        df.index = pd.to_datetime(df.index)
        return df

# =================================================================
# Main method
# =================================================================
if __name__ == "__main__":
    hStockPriceChart = HStockPriceChart()
    hStockPriceChart.launch()