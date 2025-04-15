import streamlit as st
import pandas as pd
import numpy as np

from utils.data_loader import get_sp500_tickers, get_stock_data, get_stock_info
from utils.analysis import get_stock_metrics
from utils.visualization import plot_stock_price, plot_returns, plot_drawdown
from utils.translations import get_translation

def show_page():
    """
    Display the Stock Analysis page
    """
    # Get translations based on selected language
    _ = get_translation(st.session_state.language)
    
    st.title(_("Stock Analysis Tool"))
    
    # Get available tickers
    available_tickers = get_sp500_tickers()
    
    # Stock selection - permitir ingresar cualquier ticker
    ticker_input_method = st.radio(
        _("Método de selección"),
        [_("Seleccionar de la lista"), _("Ingresar símbolo manualmente")],
        horizontal=True
    )
    
    if ticker_input_method == _("Seleccionar de la lista"):
        selected_ticker = st.selectbox(
            _("Select a stock to analyze"),
            options=available_tickers,
            index=0 if len(available_tickers) > 0 else None
        )
    else:
        selected_ticker = st.text_input(
            _("Ingresar símbolo del ticker (ej: AAPL, MSFT, etc.)"), 
            value="AAPL"
        ).upper()
    
    if selected_ticker:
        # Get stock metrics
        metrics = get_stock_metrics(selected_ticker, period=st.session_state.period)
        
        if metrics:
            # Display metrics in cards
            st.subheader(_("Key Metrics"))
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    _("Annualized Return"),
                    f"{metrics['annualized_return']:.2f}%"
                )
            
            with col2:
                st.metric(
                    _("Volatility"),
                    f"{metrics['volatility']:.2f}%"
                )
            
            with col3:
                st.metric(
                    _("Sharpe Ratio"),
                    f"{metrics['sharpe_ratio']:.2f}"
                )
            
            with col4:
                st.metric(
                    _("Max Drawdown"),
                    f"{metrics['max_drawdown']:.2f}%"
                )
            
            # Second row of metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    _("Beta"),
                    f"{metrics['beta']:.2f}"
                )
            
            with col2:
                st.metric(
                    _("Alpha"),
                    f"{metrics['alpha']:.2f}%"
                )
            
            with col3:
                st.metric(
                    _("Value at Risk (95%)"),
                    f"{metrics['var_95']:.2f}%"
                )
            
            with col4:
                st.metric(
                    _("Conditional VaR (95%)"),
                    f"{metrics['cvar_95']:.2f}%"
                )
            
            # Price chart
            st.subheader(_("Price Chart"))
            price_chart = plot_stock_price(
                selected_ticker,
                period=st.session_state.period,
                language=st.session_state.language
            )
            
            if price_chart:
                st.plotly_chart(price_chart, use_container_width=True)
            else:
                st.warning(_("No data available for the selected period"))
            
            # Returns chart
            st.subheader(_("Returns"))
            returns_chart = plot_returns(
                selected_ticker,
                period=st.session_state.period,
                language=st.session_state.language
            )
            
            if returns_chart:
                st.plotly_chart(returns_chart, use_container_width=True)
            else:
                st.warning(_("No data available for the selected period"))
            
            # Drawdown chart
            st.subheader(_("Drawdown"))
            drawdown_chart = plot_drawdown(
                selected_ticker,
                period=st.session_state.period,
                language=st.session_state.language
            )
            
            if drawdown_chart:
                st.plotly_chart(drawdown_chart, use_container_width=True)
            else:
                st.warning(_("No data available for the selected period"))
            
            # Stock information
            st.subheader(_("Stock Information"))
            
            # Get stock info
            info = get_stock_info(selected_ticker)
            
            if info:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**{_('Company')}:** {info.get('longName', 'N/A')}")
                    st.write(f"**{_('Sector')}:** {info.get('sector', 'N/A')}")
                    st.write(f"**{_('Industry')}:** {info.get('industry', 'N/A')}")
                    st.write(f"**{_('Website')}:** {info.get('website', 'N/A')}")
                
                with col2:
                    st.write(f"**{_('Market Cap')}:** ${info.get('marketCap', 0):,.0f}")
                    st.write(f"**{_('Forward P/E')}:** {info.get('forwardPE', 'N/A')}")
                    st.write(f"**{_('Dividend Yield')}:** {info.get('dividendYield', 'N/A')}%")
                    
                    # Target prices
                    target_mean = info.get('targetMeanPrice', 'N/A')
                    target_high = info.get('targetHighPrice', 'N/A')
                    target_low = info.get('targetLowPrice', 'N/A')
                    
                    st.write(f"**{_('Target Price')}:** ${target_mean} (${target_low} - ${target_high})")
                    st.write(f"**{_('Recommendation')}:** {info.get('recommendation', 'N/A')}")
            else:
                st.warning(_("Stock information not available"))
        else:
            st.warning(_("Not enough data for analysis"))
