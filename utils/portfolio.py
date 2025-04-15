import pandas as pd
import numpy as np
import streamlit as st
from scipy.optimize import minimize

from utils.data_loader import get_stock_data, get_risk_free_rate
from utils.analysis import (
    calculate_returns, calculate_annualized_return,
    calculate_volatility, calculate_sharpe_ratio,
    calculate_max_drawdown, calculate_beta
)

def get_portfolio_returns(portfolio_df, period="1y"):
    """
    Calculate historical returns for a portfolio
    
    Args:
        portfolio_df (pd.DataFrame): DataFrame with portfolio data
        period (str): Time period for analysis
    
    Returns:
        pd.Series: Portfolio returns
    """
    # Get list of tickers from portfolio
    tickers = portfolio_df['Ticker'].tolist()
    weights = portfolio_df['Shares'] * portfolio_df['Price'] / (portfolio_df['Shares'] * portfolio_df['Price']).sum()
    weights_dict = dict(zip(tickers, weights))
    
    # Dictionary to store returns data for each ticker
    returns_data = {}
    
    # Get historical returns data for each ticker
    for ticker in tickers:
        stock_data = get_stock_data(ticker, period=period)
        
        if not stock_data.empty:
            # Calculate returns
            returns = calculate_returns(stock_data['Close'])
            returns_data[ticker] = returns
    
    if not returns_data:
        return pd.Series()
    
    # Create DataFrame with all returns
    returns_df = pd.DataFrame(returns_data)
    
    # Fill any missing values (NaN) with 0
    returns_df = returns_df.fillna(0)
    
    # Calculate portfolio returns based on weights
    portfolio_returns = pd.Series(0.0, index=returns_df.index)
    
    for ticker in tickers:
        if ticker in returns_df.columns:
            portfolio_returns += returns_df[ticker] * weights_dict.get(ticker, 0)
    
    return portfolio_returns

def get_portfolio_stats(portfolio_df, period="1y"):
    """
    Calculate portfolio statistics
    
    Args:
        portfolio_df (pd.DataFrame): DataFrame with portfolio data
        period (str): Time period for analysis
    
    Returns:
        dict: Portfolio statistics
    """
    # Calculate total portfolio value
    total_value = portfolio_df['Value'].sum()
    
    # Get portfolio returns
    portfolio_returns = get_portfolio_returns(portfolio_df, period=period)
    
    if len(portfolio_returns) == 0:
        return {
            'total_value': total_value,
            'return_pct': 0,
            'annualized_return': 0,
            'volatility': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'beta': 0
        }
    
    # Calculate performance metrics
    total_return = (1 + portfolio_returns).prod() - 1
    annualized_return = calculate_annualized_return(portfolio_returns) * 100
    volatility = calculate_volatility(portfolio_returns) * 100
    
    # Get risk-free rate
    risk_free_rate = get_risk_free_rate() / 100  # Convert from % to decimal
    
    # Calculate Sharpe ratio
    sharpe_ratio = calculate_sharpe_ratio(portfolio_returns, risk_free_rate)
    
    # Calculate maximum drawdown
    max_drawdown = calculate_max_drawdown(portfolio_returns) * 100
    
    # Calculate portfolio beta (vs S&P 500)
    market_data = get_stock_data("^GSPC", period=period)
    market_returns = calculate_returns(market_data['Close'])
    
    # Align the data
    common_index = portfolio_returns.index.intersection(market_returns.index)
    if len(common_index) > 0:
        portfolio_returns_aligned = portfolio_returns.loc[common_index]
        market_returns_aligned = market_returns.loc[common_index]
        beta = calculate_beta(portfolio_returns_aligned, market_returns_aligned)
    else:
        beta = 1.0  # Default to 1.0 if data cannot be aligned
    
    # Compile results - asegurar que todos los valores son tipos nativos de Python
    # Convertir correctamente valores de Series a float si es necesario
    portfolio_stats = {
        'total_value': float(total_value),
        'return_pct': float(total_return * 100),
        'annualized_return': float(annualized_return),
        'volatility': float(volatility),
        'sharpe_ratio': float(sharpe_ratio.iloc[0]) if hasattr(sharpe_ratio, 'iloc') else float(sharpe_ratio),
        'max_drawdown': float(max_drawdown),
        'beta': float(beta.iloc[0]) if hasattr(beta, 'iloc') else float(beta)
    }
    
    return portfolio_stats

def optimize_portfolio(tickers, period="1y", target_return=None, risk_aversion=1):
    """
    Optimize portfolio weights to maximize Sharpe ratio
    
    Args:
        tickers (list): List of ticker symbols
        period (str): Time period for analysis
        target_return (float): Target portfolio return (for minimum variance portfolio)
        risk_aversion (float): Risk aversion parameter (for utility maximization)
    
    Returns:
        dict: Optimized portfolio results
    """
    try:
        # Dictionary to store returns data for each ticker
        returns_data = {}
        
        # Get historical returns data for each ticker
        for ticker in tickers:
            stock_data = get_stock_data(ticker, period=period)
            
            if not stock_data.empty:
                # Calculate returns
                returns = calculate_returns(stock_data['Close'])
                returns_data[ticker] = returns
        
        if not returns_data:
            return {"error": "No valid returns data for tickers"}
        
        # Create DataFrame with all returns
        returns_df = pd.DataFrame(returns_data)
        
        # Remove any tickers with missing data
        returns_df = returns_df.dropna(axis=1)
        
        if returns_df.empty or returns_df.shape[1] < 2:
            return {"error": "Not enough tickers with valid data"}
        
        # Calculate mean returns and covariance matrix
        mean_returns = returns_df.mean()
        cov_matrix = returns_df.cov()
        
        # Number of assets
        num_assets = len(mean_returns)
        
        # Get risk-free rate
        try:
            risk_free_rate = get_risk_free_rate() / 100 / 252  # Daily risk-free rate
        except:
            risk_free_rate = 0.0001  # Default to a small value if there's an error
        
        # Define the objective function (negative Sharpe ratio)
        def objective(weights):
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            if target_return is not None:
                # Minimum variance objective
                return portfolio_volatility
            else:
                # Sharpe ratio objective
                return -(portfolio_return - risk_free_rate) / portfolio_volatility
        
        # Define the constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
        
        if target_return is not None:
            constraints.append({'type': 'eq', 'fun': lambda x: np.sum(mean_returns * x) - target_return})
        
        # Define the bounds (0 <= weight <= 1)
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Initial guess (equal weights)
        initial_weights = np.array([1.0 / num_assets] * num_assets)
        
        # Run the optimization
        optimization_result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Extract the optimized weights
        optimized_weights = optimization_result['x']
        
        # Calculate portfolio performance with optimized weights
        portfolio_return = float(np.sum(mean_returns * optimized_weights) * 252)  # Annualized return
        portfolio_volatility = float(np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix, optimized_weights))) * np.sqrt(252))  # Annualized volatility
        sharpe_ratio = float((portfolio_return - risk_free_rate * 252) / portfolio_volatility)
        
        # Create weights dictionary
        weights_dict = dict(zip(returns_df.columns, optimized_weights))
        
        # Sort weights by allocation (descending)
        sorted_weights = sorted(weights_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Compile results - convertir valores a tipos Python nativos
        optimization_results = {
            'weights': weights_dict,
            'sorted_weights': sorted_weights,
            'portfolio_return': float(portfolio_return * 100),  # Convert to percentage as float
            'portfolio_volatility': float(portfolio_volatility * 100),  # Convert to percentage as float
            'sharpe_ratio': float(sharpe_ratio.iloc[0]) if hasattr(sharpe_ratio, 'iloc') else float(sharpe_ratio)
        }
        
        return optimization_results
    except Exception as e:
        return {"error": str(e)}

def calculate_efficient_frontier(tickers, period="1y", num_portfolios=20):
    """
    Calculate the efficient frontier for a set of assets
    
    Args:
        tickers (list): List of ticker symbols
        period (str): Time period for analysis
        num_portfolios (int): Number of portfolios on the frontier
    
    Returns:
        dict: Efficient frontier data
    """
    try:
        # Dictionary to store returns data for each ticker
        returns_data = {}
        
        # Get historical returns data for each ticker
        for ticker in tickers:
            stock_data = get_stock_data(ticker, period=period)
            
            if not stock_data.empty:
                # Calculate returns
                returns = calculate_returns(stock_data['Close'])
                returns_data[ticker] = returns
        
        if not returns_data:
            return {"error": "No valid returns data for tickers"}
        
        # Create DataFrame with all returns
        returns_df = pd.DataFrame(returns_data)
        
        # Remove any tickers with missing data
        returns_df = returns_df.dropna(axis=1)
        
        if returns_df.empty or returns_df.shape[1] < 2:
            return {"error": "Not enough tickers with valid data"}
        
        # Calculate mean returns and covariance matrix
        mean_returns = returns_df.mean()
        cov_matrix = returns_df.cov()
        
        # Number of assets
        num_assets = len(mean_returns)
        
        # Create lists to store results
        frontier_returns = []
        frontier_volatilities = []
        frontier_sharpe_ratios = []
        frontier_weights = []
        
        # Get risk-free rate
        risk_free_rate = get_risk_free_rate() / 100  # Annual risk-free rate
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1  # Daily risk-free rate
        
        # Find the minimum and maximum returns
        min_return = min(mean_returns) * 252
        max_return = max(mean_returns) * 252
        
        # Generate target returns for the efficient frontier
        target_returns = np.linspace(min_return, max_return, num_portfolios)
        
        # Calculate the optimal portfolio for each target return
        for target_return in target_returns:
            # Daily target return
            daily_target = target_return / 252
            
            # Optimize portfolio for this target return
            result = optimize_portfolio(tickers, period=period, target_return=daily_target)
            
            if "error" in result:
                continue
            
            # Store results
            frontier_returns.append(result['portfolio_return'])
            frontier_volatilities.append(result['portfolio_volatility'])
            frontier_sharpe_ratios.append(result['sharpe_ratio'])
            frontier_weights.append(result['weights'])
        
        # Find the optimal portfolio (maximum Sharpe ratio)
        max_sharpe_idx = np.argmax(frontier_sharpe_ratios) if frontier_sharpe_ratios else 0
        
        # Find the minimum variance portfolio
        min_var_idx = np.argmin(frontier_volatilities) if frontier_volatilities else 0
        
        # Compile results - convert any lists to regular Python lists to avoid issues with Series formatting
        frontier_data = {
            'returns': [float(x) for x in frontier_returns],
            'volatilities': [float(x) for x in frontier_volatilities],
            'sharpe_ratios': [float(x) for x in frontier_sharpe_ratios],
            'weights': frontier_weights,
            'max_sharpe_idx': int(max_sharpe_idx),
            'min_var_idx': int(min_var_idx)
        }
        
        return frontier_data
    except Exception as e:
        return {"error": str(e)}
