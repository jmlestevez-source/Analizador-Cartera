import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats
import statsmodels.api as sm
import streamlit as st
from utils.data_loader import get_stock_data, get_risk_free_rate

def calculate_returns(prices):
    """
    Calculate daily returns from a price series
    
    Args:
        prices (pd.Series): Series of prices
    
    Returns:
        pd.Series: Daily returns
    """
    return prices.pct_change().dropna()

def calculate_cumulative_returns(returns):
    """
    Calculate cumulative returns from a returns series
    
    Args:
        returns (pd.Series): Series of returns
    
    Returns:
        pd.Series: Cumulative returns
    """
    return (1 + returns).cumprod() - 1

def calculate_annualized_return(returns, periods_per_year=252):
    """
    Calculate annualized return from a returns series
    
    Args:
        returns (pd.Series): Series of returns
        periods_per_year (int): Number of periods in a year (252 for daily data)
    
    Returns:
        float: Annualized return
    """
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    years = n_periods / periods_per_year
    return (1 + total_return) ** (1 / years) - 1

def calculate_volatility(returns, periods_per_year=252):
    """
    Calculate annualized volatility from a returns series
    
    Args:
        returns (pd.Series): Series of returns
        periods_per_year (int): Number of periods in a year (252 for daily data)
    
    Returns:
        float: Annualized volatility
    """
    return returns.std() * np.sqrt(periods_per_year)

def calculate_sharpe_ratio(returns, risk_free_rate=None, periods_per_year=252):
    """
    Calculate Sharpe ratio from a returns series
    
    Args:
        returns (pd.Series): Series of returns
        risk_free_rate (float): Risk-free rate (annualized)
        periods_per_year (int): Number of periods in a year (252 for daily data)
    
    Returns:
        float: Sharpe ratio
    """
    if risk_free_rate is None:
        risk_free_rate = get_risk_free_rate() / 100  # Convert from % to decimal
    
    excess_return = calculate_annualized_return(returns, periods_per_year) - risk_free_rate
    return excess_return / calculate_volatility(returns, periods_per_year)

def calculate_drawdown(returns):
    """
    Calculate drawdown from a returns series
    
    Args:
        returns (pd.Series): Series of returns
    
    Returns:
        pd.Series: Drawdown series
    """
    wealth_index = (1 + returns).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdown = (wealth_index - previous_peaks) / previous_peaks
    return drawdown

def calculate_max_drawdown(returns):
    """
    Calculate maximum drawdown from a returns series
    
    Args:
        returns (pd.Series): Series of returns
    
    Returns:
        float: Maximum drawdown
    """
    return calculate_drawdown(returns).min()

def calculate_sortino_ratio(returns, risk_free_rate=None, periods_per_year=252):
    """
    Calculate Sortino ratio from a returns series
    
    Args:
        returns (pd.Series): Series of returns
        risk_free_rate (float): Risk-free rate (annualized)
        periods_per_year (int): Number of periods in a year (252 for daily data)
    
    Returns:
        float: Sortino ratio
    """
    if risk_free_rate is None:
        risk_free_rate = get_risk_free_rate() / 100  # Convert from % to decimal
    
    excess_return = calculate_annualized_return(returns, periods_per_year) - risk_free_rate
    
    # Calculate downside deviation (only negative returns)
    negative_returns = returns[returns < 0]
    downside_deviation = np.sqrt(np.sum(negative_returns**2) / len(returns)) * np.sqrt(periods_per_year)
    
    # Avoid division by zero
    if downside_deviation == 0:
        return np.nan
    
    return excess_return / downside_deviation

def calculate_var(returns, confidence_level=0.05):
    """
    Calculate Value at Risk (VaR) from a returns series
    
    Args:
        returns (pd.Series): Series of returns
        confidence_level (float): Confidence level (e.g., 0.05 for 95% confidence)
    
    Returns:
        float: Value at Risk
    """
    return np.percentile(returns, confidence_level * 100)

def calculate_cvar(returns, confidence_level=0.05):
    """
    Calculate Conditional Value at Risk (CVaR) from a returns series
    
    Args:
        returns (pd.Series): Series of returns
        confidence_level (float): Confidence level (e.g., 0.05 for 95% confidence)
    
    Returns:
        float: Conditional Value at Risk
    """
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()

def calculate_beta(stock_returns, market_returns):
    """
    Calculate beta of a stock relative to a market index
    
    Args:
        stock_returns (pd.Series): Series of stock returns
        market_returns (pd.Series): Series of market returns
    
    Returns:
        float: Beta
    """
    # Align the data
    df = pd.DataFrame({'stock': stock_returns, 'market': market_returns})
    df = df.dropna()
    
    if len(df) < 10:  # Require at least 10 data points
        return np.nan
    
    # Calculate beta using regression
    X = sm.add_constant(df['market'])
    model = sm.OLS(df['stock'], X).fit()
    return model.params['market']

def calculate_alpha(stock_returns, market_returns, risk_free_rate=None):
    """
    Calculate Jensen's alpha
    
    Args:
        stock_returns (pd.Series): Series of stock returns
        market_returns (pd.Series): Series of market returns
        risk_free_rate (float): Risk-free rate (annualized)
    
    Returns:
        float: Alpha
    """
    if risk_free_rate is None:
        risk_free_rate = get_risk_free_rate() / 100  # Convert from % to decimal
    
    # Convert annual risk-free rate to daily
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    
    # Align the data
    df = pd.DataFrame({'stock': stock_returns, 'market': market_returns})
    df = df.dropna()
    
    if len(df) < 10:  # Require at least 10 data points
        return np.nan
    
    # Calculate beta
    beta = calculate_beta(df['stock'], df['market'])
    
    # Calculate alpha (intercept)
    stock_ret_avg = df['stock'].mean()
    market_ret_avg = df['market'].mean()
    
    # Jensen's Alpha formula: Alpha = R_p - [R_f + beta * (R_m - R_f)]
    alpha = stock_ret_avg - (daily_rf + beta * (market_ret_avg - daily_rf))
    
    # Annualize alpha
    annualized_alpha = (1 + alpha) ** 252 - 1
    
    return annualized_alpha

@st.cache_data(ttl=3600)
def get_stock_metrics(ticker, period="1y"):
    """
    Calculate stock metrics for a given ticker
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Time period for analysis
    
    Returns:
        dict: Dictionary with stock metrics
    """
    try:
        # Get stock data
        stock_data = get_stock_data(ticker, period=period)
        
        # Get market data (S&P 500)
        market_data = get_stock_data("^GSPC", period=period)
        
        # Calculate returns
        stock_returns = calculate_returns(stock_data['Close'])
        market_returns = calculate_returns(market_data['Close'])
        
        # Calculate metrics
        annualized_return = calculate_annualized_return(stock_returns) * 100
        volatility = calculate_volatility(stock_returns) * 100
        sharpe_ratio = calculate_sharpe_ratio(stock_returns)
        max_drawdown = calculate_max_drawdown(stock_returns) * 100
        sortino_ratio = calculate_sortino_ratio(stock_returns)
        var_95 = calculate_var(stock_returns) * 100
        cvar_95 = calculate_cvar(stock_returns) * 100
        beta = calculate_beta(stock_returns, market_returns)
        alpha = calculate_alpha(stock_returns, market_returns) * 100
        
        # Compile results - convertimos valores de Series a float para evitar problemas de formato
        metrics = {
            'annualized_return': float(annualized_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'sortino_ratio': float(sortino_ratio),
            'var_95': float(var_95),
            'cvar_95': float(cvar_95),
            'beta': float(beta),
            'alpha': float(alpha),
            'returns': stock_returns,
            'prices': stock_data['Close']
        }
        
        return metrics
    except Exception as e:
        st.error(f"Error calculating metrics for {ticker}: {str(e)}")
        return {}

def calculate_correlation_matrix(tickers, period="1y"):
    """
    Calculate correlation matrix for a list of tickers
    
    Args:
        tickers (list): List of ticker symbols
        period (str): Time period for analysis
    
    Returns:
        pd.DataFrame: Correlation matrix
    """
    try:
        returns_data = {}
        
        for ticker in tickers:
            stock_data = get_stock_data(ticker, period=period)
            if not stock_data.empty:
                returns = calculate_returns(stock_data['Close'])
                returns_data[ticker] = returns
        
        # Create a DataFrame with all returns
        returns_df = pd.DataFrame(returns_data)
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        return corr_matrix
    except Exception as e:
        st.error(f"Error calculating correlation matrix: {str(e)}")
        return pd.DataFrame()
