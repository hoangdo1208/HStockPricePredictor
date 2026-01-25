# HStockPricePredictor | Documentation Guide

HStockPricePredictor is an end-to-end machine learning framework designed to aggregate Vietnamese stock market data (Vnstock), synchronize local and cloud storage, train deep learning models, and serve predictions through an interactive web dashboard.

---

## 📋 Command Quick Reference

| Action | Command | Description |
| :--- | :--- | :--- |
| **Crawl** | `python main.py CrawlData` | Initial data fetch from Vnstock (TCB). |
| **Retry** | `python main.py RetryCrawlData` | Resume failed data acquisition tasks. |
| **Sync** | `python main.py SyncCrawlData` | Synchronize CSV files to the database. |
| **Train** | `python main.py Train` | Train neural network models for all tickers. |
| **Serve** | `python main.py` | Launch the interactive web dashboard. |

---

## 1. Data Acquisition
### 1.1 Initial Data Crawl
This session connects to the Vnstock (TCB) API to retrieve historical price data. 
* **Dual-Write Storage:** Data is simultaneously saved to individual `.csv` files (one per ticker) and the central application database.
* **Scope:** Ideal for initial environment setup or full data refreshes.

```bash
python main.py CrawlData
```

### 1.2 Retry Mechanism
Data acquisition can occasionally fail due to network instability, API rate limiting, or TCB response timeouts. This session identifies stocks that failed to persist correctly and re-attempts the process only for those specific records.

```bash
python main.py RetryCrawlData
```

## 2. Data Management
### 2.1 CSV to Database Synchronization
If the central database is reset or CSV files are updated manually, this utility ensures the database is perfectly aligned with your local storage. It parses all existing .csv files and updates the database records accordingly.

```bash
python main.py SyncCrawlData
```

## 3. Model Training Pipeline
This session executes the end-to-end training pipeline for all stocks in the system. For every ticker, the application generates and saves three critical artifacts:
1. Model File (.keras): The trained neural network architecture and weights.
2. Scaler File (.scaler): Feature scaling parameters, with all data normalized in the range of 0 to 1.
3. History File (.csv): A comprehensive log of the training session, including Mean Absolute Error (MAE) and Learning Rate (LR) metrics.

```bash
python main.py Train
```

## 4. Web Application Deployment
### 4.1 Launching the Dashboard
To start the interactive web interface for visualizing predictions and market trends, run the main script without additional arguments.

```bash
python main.py
```
### 4.2 Network Access and Configuration
Once the server is initialized, the application is accessible via the following endpoints:
* Local Access: http://localhost:8501
* Network Access (LAN): http://<your-ip-address>:8501
