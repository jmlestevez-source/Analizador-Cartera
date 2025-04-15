import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta

from utils.data_loader import get_stock_data
from utils.analysis import (
    calculate_returns, calculate_cumulative_returns, 
    calculate_drawdown, calculate_correlation_matrix
)
from utils.translations import get_translation

def plot_stock_price(ticker, period="1y", language="en"):
    """
    Plot stock price chart for a ticker
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Time period for chart
        language (str): Language for chart labels
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure with stock price chart
    """
    _ = get_translation(language)
    
    # Get stock data
    stock_data = get_stock_data(ticker, period=period)
    
    if stock_data.empty:
        return None
    
    # Create figure
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=stock_data.index,
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close'],
            name=ticker
        )
    )
    
    # Add volume as bar chart on secondary y-axis
    fig.add_trace(
        go.Bar(
            x=stock_data.index,
            y=stock_data['Volume'],
            name=_("Volume"),
            marker_color='rgba(0, 0, 100, 0.3)',
            opacity=0.3,
            yaxis="y2"
        )
    )
    
    # Configure chart layout
    fig.update_layout(
        title=_("Price Chart for {}").format(ticker),
        xaxis_title=_("Date"),
        yaxis_title=_("Price"),
        yaxis2=dict(
            title=_("Volume"),
            overlaying="y",
            side="right",
            showgrid=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=600
    )
    
    # Set y-axis range to expand a bit beyond the data
    y_min = stock_data['Low'].min() * 0.95
    y_max = stock_data['High'].max() * 1.05
    fig.update_yaxes(range=[y_min, y_max], side="left")
    
    return fig

def plot_returns(ticker, period="1y", language="en"):
    """
    Plot returns chart for a ticker
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Time period for chart
        language (str): Language for chart labels
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure with returns chart
    """
    _ = get_translation(language)
    
    # Get stock data
    stock_data = get_stock_data(ticker, period=period)
    
    if stock_data.empty:
        return None
    
    # Calculate returns
    returns = calculate_returns(stock_data['Close'])
    cumulative_returns = calculate_cumulative_returns(returns)
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, 
        cols=1,
        subplot_titles=[
            _("Daily Returns for {}").format(ticker),
            _("Cumulative Returns for {}").format(ticker)
        ],
        vertical_spacing=0.1,
        shared_xaxes=True
    )
    
    # Daily returns plot
    fig.add_trace(
        go.Bar(
            x=returns.index,
            y=returns,
            name=_("Daily Returns"),
            marker_color=['rgba(255,0,0,0.8)' if x < 0 else 'rgba(0,128,0,0.8)' for x in returns]
        ),
        row=1, col=1
    )
    
    # Cumulative returns plot
    fig.add_trace(
        go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns,
            name=_("Cumulative Returns"),
            line=dict(color='rgba(0, 0, 255, 0.8)', width=2)
        ),
        row=2, col=1
    )
    
    # Configure chart layout
    fig.update_layout(
        xaxis_title=_("Date"),
        yaxis_title=_("Daily Returns"),
        yaxis2_title=_("Cumulative Returns"),
        height=700,
        showlegend=False
    )
    
    # Format y-axis as percentages
    fig.update_yaxes(tickformat=".2%")
    
    return fig

def plot_drawdown(ticker, period="1y", language="en"):
    """
    Plot drawdown chart for a ticker
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Time period for chart
        language (str): Language for chart labels
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure with drawdown chart
    """
    _ = get_translation(language)
    
    # Get stock data
    stock_data = get_stock_data(ticker, period=period)
    
    if stock_data.empty:
        return None
    
    # Calculate returns and drawdown
    returns = calculate_returns(stock_data['Close'])
    drawdown = calculate_drawdown(returns)
    
    # Create figure
    fig = go.Figure()
    
    # Add drawdown chart
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown,
            name=_("Drawdown"),
            fill='tozeroy',
            line=dict(color='rgba(200, 0, 0, 0.8)', width=1)
        )
    )
    
    # Configure chart layout
    fig.update_layout(
        title=_("Drawdown Chart for {}").format(ticker),
        xaxis_title=_("Date"),
        yaxis_title=_("Drawdown"),
        height=500
    )
    
    # Format y-axis as percentages
    fig.update_yaxes(tickformat=".2%")
    
    return fig

def plot_correlation_heatmap(tickers, period="1y", language="en"):
    """
    Plot correlation heatmap for a list of tickers
    
    Args:
        tickers (list): List of ticker symbols
        period (str): Time period for analysis
        language (str): Language for chart labels
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure with correlation heatmap
    """
    _ = get_translation(language)
    
    # Calculate correlation matrix
    corr_matrix = calculate_correlation_matrix(tickers, period=period)
    
    if corr_matrix.empty:
        return None
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        aspect="equal"
    )
    
    # Configure chart layout
    fig.update_layout(
        title=_("Correlation Matrix"),
        height=600,
        coloraxis_colorbar=dict(
            title=_("Correlation"),
            thicknessmode="pixels", thickness=20,
            lenmode="pixels", len=500,
            ticks="outside"
        )
    )
    
    return fig

def plot_portfolio_performance(portfolio_df, period="1y", language="en"):
    """
    Plot portfolio performance chart
    
    Args:
        portfolio_df (pd.DataFrame): DataFrame with portfolio data
        period (str): Time period for chart
        language (str): Language for chart labels
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure with portfolio performance chart
    """
    _ = get_translation(language)
    
    # Get list of tickers from portfolio
    tickers = portfolio_df['Ticker'].tolist()
    
    if not tickers:
        return None
    
    # Dictionary to store price data for each ticker
    price_data = {}
    
    # Get historical price data for each ticker
    for ticker in tickers:
        stock_data = get_stock_data(ticker, period=period)
        if not stock_data.empty:
            price_data[ticker] = stock_data['Close']
    
    # Create DataFrame with all price data
    prices_df = pd.DataFrame(price_data)
    
    # Calculate portfolio value over time
    portfolio_value = pd.Series(0.0, index=prices_df.index)
    
    for ticker, shares in zip(portfolio_df['Ticker'], portfolio_df['Shares']):
        if ticker in prices_df.columns:
            portfolio_value += prices_df[ticker] * shares
    
    # Calculate portfolio returns
    portfolio_returns = portfolio_value.pct_change().dropna()
    portfolio_cum_returns = (1 + portfolio_returns).cumprod() - 1
    
    # Get S&P 500 data for comparison
    sp500_data = get_stock_data("^GSPC", period=period)
    sp500_returns = sp500_data['Close'].pct_change().dropna()
    sp500_cum_returns = (1 + sp500_returns).cumprod() - 1
    
    # Align the data
    common_index = portfolio_cum_returns.index.intersection(sp500_cum_returns.index)
    portfolio_cum_returns = portfolio_cum_returns.loc[common_index]
    sp500_cum_returns = sp500_cum_returns.loc[common_index]
    
    # Create figure
    fig = go.Figure()
    
    # Add portfolio cumulative returns
    fig.add_trace(
        go.Scatter(
            x=portfolio_cum_returns.index,
            y=portfolio_cum_returns,
            name=_("Portfolio"),
            line=dict(color='rgba(0, 100, 255, 0.8)', width=2)
        )
    )
    
    # Add S&P 500 cumulative returns for comparison
    fig.add_trace(
        go.Scatter(
            x=sp500_cum_returns.index,
            y=sp500_cum_returns,
            name=_("S&P 500"),
            line=dict(color='rgba(200, 0, 0, 0.8)', width=2, dash='dash')
        )
    )
    
    # Configure chart layout
    fig.update_layout(
        title=_("Portfolio Performance vs S&P 500"),
        xaxis_title=_("Date"),
        yaxis_title=_("Cumulative Returns"),
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Format y-axis as percentages
    fig.update_yaxes(tickformat=".2%")
    
    return fig

def plot_asset_allocation(portfolio_df, language="en"):
    """
    Plot asset allocation pie chart
    
    Args:
        portfolio_df (pd.DataFrame): DataFrame with portfolio data
        language (str): Language for chart labels
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure with asset allocation pie chart
    """
    _ = get_translation(language)
    
    # Calculate total value for each position
    values = portfolio_df['Value'].values
    labels = portfolio_df['Ticker'].values
    
    # Create pie chart
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                textinfo='label+percent',
                insidetextorientation='radial',
                marker=dict(
                    line=dict(color='#FFFFFF', width=1)
                )
            )
        ]
    )
    
    # Configure chart layout
    fig.update_layout(
        title=_("Asset Allocation"),
        height=500
    )
    
    return fig

def plot_risk_return_scatter(tickers, period="1y", language="en"):
    """
    Plot risk-return scatter chart for a list of tickers
    
    Args:
        tickers (list): List of ticker symbols
        period (str): Time period for analysis
        language (str): Language for chart labels
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure with risk-return scatter chart
    """
    _ = get_translation(language)
    
    # Dictionary to store returns data for each ticker
    returns_data = {}
    annualized_returns = {}
    volatilities = {}
    
    # Get historical returns data for each ticker
    for ticker in tickers:
        stock_data = get_stock_data(ticker, period=period)
        
        if not stock_data.empty:
            # Calculate returns
            returns = calculate_returns(stock_data['Close'])
            returns_data[ticker] = returns
            
            # Calculate annualized return
            if len(returns) > 0:
                if period == "1mo":
                    periods_per_year = 252 / 21
                elif period == "3mo":
                    periods_per_year = 252 / 63
                elif period == "6mo":
                    periods_per_year = 252 / 126
                elif period == "1y":
                    periods_per_year = 1
                elif period == "2y":
                    periods_per_year = 1/2
                elif period == "5y":
                    periods_per_year = 1/5
                else:
                    periods_per_year = 1
                
                total_return = (1 + returns).prod() - 1
                annualized_return = (1 + total_return) ** periods_per_year - 1
                annualized_returns[ticker] = annualized_return * 100
                
                # Calculate volatility
                volatility = returns.std() * np.sqrt(252) * 100
                volatilities[ticker] = volatility
    
    # Get S&P 500 data for reference
    sp500_data = get_stock_data("^GSPC", period=period)
    sp500_returns = calculate_returns(sp500_data['Close'])
    
    if len(sp500_returns) > 0:
        total_return = (1 + sp500_returns).prod() - 1
        sp500_annualized_return = (1 + total_return) ** periods_per_year - 1
        sp500_volatility = sp500_returns.std() * np.sqrt(252) * 100
        
        # Add S&P 500 to the data
        annualized_returns["S&P 500"] = sp500_annualized_return * 100
        volatilities["S&P 500"] = sp500_volatility
    
    # Create DataFrame for scatter plot
    scatter_data = pd.DataFrame({
        'Ticker': list(annualized_returns.keys()),
        'Return': list(annualized_returns.values()),
        'Volatility': list(volatilities.values())
    })
    
    # Create scatter plot
    fig = px.scatter(
        scatter_data,
        x='Volatility',
        y='Return',
        text='Ticker',
        hover_data=['Ticker', 'Return', 'Volatility'],
        size_max=60,
        size=[40] * len(scatter_data)
    )
    
    # Highlight S&P 500 point
    if "S&P 500" in scatter_data['Ticker'].values:
        sp500_index = scatter_data[scatter_data['Ticker'] == "S&P 500"].index[0]
        
        fig.update_traces(
            marker=dict(
                size=40,
                color=['royalblue' if i != sp500_index else 'red' for i in range(len(scatter_data))]
            )
        )
    
    # Configure chart layout
    fig.update_layout(
        title=_("Risk-Return Analysis"),
        xaxis_title=_("Volatility (%)"),
        yaxis_title=_("Annualized Return (%)"),
        height=600,
        showlegend=False
    )
    
    # Add quadrant lines at S&P 500 point
    if "S&P 500" in annualized_returns:
        sp500_return = annualized_returns["S&P 500"]
        sp500_volatility = volatilities["S&P 500"]
        
        fig.add_vline(
            x=sp500_volatility,
            line=dict(color="gray", width=0.5, dash="dash")
        )
        
        fig.add_hline(
            y=sp500_return,
            line=dict(color="gray", width=0.5, dash="dash")
        )
        
        # Add quadrant labels
        x_range = fig.layout.xaxis.range
        y_range = fig.layout.yaxis.range
        
        if x_range is None or y_range is None:
            x_min = min(volatilities.values()) * 0.9
            x_max = max(volatilities.values()) * 1.1
            y_min = min(annualized_returns.values()) * 0.9
            y_max = max(annualized_returns.values()) * 1.1
        else:
            x_min, x_max = x_range
            y_min, y_max = y_range
        
        annotations = [
            dict(
                x=(sp500_volatility + x_min) / 2,
                y=(sp500_return + y_max) / 2,
                text=_("Lower Risk<br>Higher Return"),
                showarrow=False,
                font=dict(size=10, color="green")
            ),
            dict(
                x=(sp500_volatility + x_max) / 2,
                y=(sp500_return + y_max) / 2,
                text=_("Higher Risk<br>Higher Return"),
                showarrow=False,
                font=dict(size=10, color="orange")
            ),
            dict(
                x=(sp500_volatility + x_min) / 2,
                y=(sp500_return + y_min) / 2,
                text=_("Lower Risk<br>Lower Return"),
                showarrow=False,
                font=dict(size=10, color="blue")
            ),
            dict(
                x=(sp500_volatility + x_max) / 2,
                y=(sp500_return + y_min) / 2,
                text=_("Higher Risk<br>Lower Return"),
                showarrow=False,
                font=dict(size=10, color="red")
            )
        ]
        
        fig.update_layout(annotations=annotations)
    
    return fig
