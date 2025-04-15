import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import streamlit as st

@st.cache_data(ttl=3600)
def get_sp500_tickers():
    """
    Get a list of S&P 500 tickers.
    Uses a backup list if the data cannot be fetched.
    """
    try:
        # Try to get current S&P 500 components
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        tickers = df['Symbol'].tolist()
        
        # Clean tickers (remove dots and special chars)
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        return tickers
    except Exception as e:
        # Fallback to a smaller set of major tickers
        return [
            "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "JPM", 
            "JNJ", "V", "PG", "UNH", "HD", "BAC", "MA", "XOM", "NVDA", "DIS",
            "PYPL", "ADBE", "CMCSA", "VZ", "NFLX", "INTC", "T", "PFE",
            "KO", "MRK", "PEP", "WMT", "CRM"
        ]

@st.cache_data(ttl=3600)
def get_stock_data(ticker, period="1y", interval="1d"):
    """
    Get historical stock data for a ticker
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
    
    Returns:
        pd.DataFrame: DataFrame with historical stock data
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        
        if data.empty:
            return pd.DataFrame()
        
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_stock_info(ticker):
    """
    Get general information about a stock
    
    Args:
        ticker (str): Stock ticker symbol
    
    Returns:
        dict: Dictionary with stock information
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract the most useful information
        relevant_info = {
            'longName': info.get('longName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'website': info.get('website', 'N/A'),
            'marketCap': info.get('marketCap', 'N/A'),
            'forwardPE': info.get('forwardPE', 'N/A'),
            'dividendYield': info.get('dividendYield', 'N/A') * 100 if info.get('dividendYield') else 'N/A',
            'targetHighPrice': info.get('targetHighPrice', 'N/A'),
            'targetLowPrice': info.get('targetLowPrice', 'N/A'),
            'targetMeanPrice': info.get('targetMeanPrice', 'N/A'),
            'recommendation': info.get('recommendationKey', 'N/A')
        }
        
        return relevant_info
    except Exception as e:
        st.error(f"Error fetching info for {ticker}: {str(e)}")
        return {}

@st.cache_data(ttl=86400)
def get_financial_data(ticker):
    """
    Get financial statements data for a company
    
    Args:
        ticker (str): Stock ticker symbol
    
    Returns:
        tuple: (income_statement, balance_sheet, cash_flow)
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Get financial statements
        income_stmt = stock.income_stmt
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        
        return income_stmt, balance_sheet, cash_flow
    except Exception as e:
        st.error(f"Error fetching financial data for {ticker}: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=3600)
def get_risk_free_rate():
    """
    Get current risk-free rate (10Y US Treasury yield)
    
    Returns:
        float: Current risk-free rate (%)
    """
    try:
        treasury_ticker = "^TNX"  # 10-year US Treasury Yield
        data = yf.download(treasury_ticker, period="1d")
        
        if not data.empty:
            return data['Close'].iloc[-1]
        else:
            # Default to a reasonable value if data fetch fails
            return 4.0
    except Exception as e:
        # Fall back to a reasonable default value
        return 4.0

