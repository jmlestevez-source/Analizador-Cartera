import streamlit as st
import pandas as pd
import os
import sys

# Add the utils directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.translations import get_translation
from utils.data_loader import get_sp500_tickers, get_stock_data
from utils.portfolio import get_portfolio_stats
from utils.visualization import plot_portfolio_performance, plot_asset_allocation

# Set page configuration
st.set_page_config(
    page_title="Financial Analysis App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if 'language' not in st.session_state:
    st.session_state.language = 'en'
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame()
if 'selected_tickers' not in st.session_state:
    st.session_state.selected_tickers = []
if 'period' not in st.session_state:
    st.session_state.period = '1y'

# Function to reset session state when changing language
def change_language():
    st.session_state.language = language_option
    st.rerun()

# Sidebar
with st.sidebar:
    # Language selection
    language_option = st.selectbox(
        "Language / Idioma",
        options=["en", "es"],
        index=0 if st.session_state.language == 'en' else 1,
        key="language_select"
    )
    
    if language_option != st.session_state.language:
        change_language()
    
    # Get translations based on selected language
    _ = get_translation(st.session_state.language)
    
    st.title(_("Financial Analysis App"))
    
    # Navigation
    st.header(_("Navigation"))
    page_options = [
        _("Dashboard"),
        _("Stock Analysis"),
        _("Portfolio Analysis"),
        _("Portfolio Manager"),
        _("Valuation Tools"),
        _("Risk Assessment")
    ]
    
    selected_page = st.radio(_("Select Page"), page_options)
    
    # Time period selection
    periods = {
        '1mo': _('1 Month'),
        '3mo': _('3 Months'),
        '6mo': _('6 Months'),
        '1y': _('1 Year'),
        '2y': _('2 Years'),
        '5y': _('5 Years'),
        'max': _('Maximum')
    }
    
    st.session_state.period = st.selectbox(
        _("Select Time Period"),
        options=list(periods.keys()),
        format_func=lambda x: periods[x],
        index=list(periods.keys()).index(st.session_state.period)
    )

# Get available tickers
available_tickers = get_sp500_tickers()

# Main content
if selected_page == _("Dashboard"):
    st.title(_("Financial Dashboard"))
    
    # Portfolio Input Section
    st.subheader(_("Portfolio Composition"))
    
    # Add ticker to portfolio
    col1, col2, col3 = st.columns([2, 1, 1])
    
    # Stock selection - permitir ingresar cualquier ticker
    ticker_input_method = st.radio(
        _("Select input method"),
        [_("Select from list"), _("Enter manually")],
        horizontal=True,
        key="ticker_input_method"
    )
    
    with col1:
        if ticker_input_method == _("Select from list"):
            selected_ticker = st.selectbox(
                _("Select Stock"),
                options=available_tickers,
                index=0 if len(available_tickers) > 0 else None,
                key="ticker_selector"
            )
        else:
            selected_ticker = st.text_input(
                _("Enter ticker symbol (e.g. AAPL, MSFT)"),
                value="",
                key="ticker_manual_input"
            ).upper()
    
    with col2:
        ticker_shares = st.number_input(
            _("Number of Shares"),
            min_value=0.0,
            step=1.0,
            value=0.0,
            key="shares_input"
        )
    
    with col3:
        add_ticker = st.button(_("Add to Portfolio"), key="add_ticker_button")
    
    if add_ticker and selected_ticker and ticker_shares > 0:
        if selected_ticker not in st.session_state.selected_tickers:
            # Get stock data
            stock_data = get_stock_data(selected_ticker, period="1d")
            
            if not stock_data.empty:
                current_price = stock_data['Close'].iloc[-1]
                new_position = pd.DataFrame({
                    'Ticker': [selected_ticker],
                    'Shares': [ticker_shares],
                    'Price': [current_price],
                    'Value': [ticker_shares * current_price]
                })
                
                if st.session_state.portfolio.empty:
                    st.session_state.portfolio = new_position
                else:
                    st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_position], ignore_index=True)
                
                st.session_state.selected_tickers.append(selected_ticker)
                st.success(_("Added {} to portfolio").format(selected_ticker))
            else:
                st.error(_("Could not fetch data for {}").format(selected_ticker))
        else:
            st.warning(_("{} is already in your portfolio").format(selected_ticker))
    
    # Display portfolio
    if not st.session_state.portfolio.empty:
        st.subheader(_("Current Portfolio"))
        st.dataframe(st.session_state.portfolio, use_container_width=True)
        
        # Clear portfolio button
        if st.button(_("Clear Portfolio")):
            st.session_state.portfolio = pd.DataFrame()
            st.session_state.selected_tickers = []
            st.rerun()
        
        # Portfolio Analysis
        if len(st.session_state.selected_tickers) > 0:
            try:
                # Get portfolio statistics
                portfolio_stats = get_portfolio_stats(
                    st.session_state.portfolio,
                    period=st.session_state.period
                )
                
                st.subheader(_("Portfolio Overview"))
                
                # Portfolio metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        _("Total Value"),
                        f"${portfolio_stats['total_value']:,.2f}",
                        delta=f"{portfolio_stats['return_pct']:.2f}%"
                    )
                
                with col2:
                    st.metric(
                        _("Annualized Return"),
                        f"{portfolio_stats['annualized_return']:.2f}%"
                    )
                
                with col3:
                    st.metric(
                        _("Volatility"),
                        f"{portfolio_stats['volatility']:.2f}%"
                    )
                
                with col4:
                    st.metric(
                        _("Sharpe Ratio"),
                        f"{portfolio_stats['sharpe_ratio']:.2f}"
                    )
                
                # Portfolio performance chart
                st.subheader(_("Portfolio Performance"))
                fig_performance = plot_portfolio_performance(
                    st.session_state.portfolio,
                    period=st.session_state.period,
                    language=st.session_state.language
                )
                st.plotly_chart(fig_performance, use_container_width=True)
                
                # Asset allocation chart
                st.subheader(_("Asset Allocation"))
                fig_allocation = plot_asset_allocation(
                    st.session_state.portfolio,
                    language=st.session_state.language
                )
                st.plotly_chart(fig_allocation, use_container_width=True)
                
            except Exception as e:
                st.error(_("Error in portfolio analysis: {}").format(str(e)))
    else:
        st.info(_("Add stocks to your portfolio to see analysis"))

elif selected_page in [_("Stock Analysis"), _("Portfolio Analysis"), _("Portfolio Manager"), _("Valuation Tools"), _("Risk Assessment")]:
    # Map the localized page name back to English for importing the correct page module
    page_mapping = {
        _("Stock Analysis"): "stock_analysis",
        _("Portfolio Analysis"): "portfolio_analysis",
        _("Portfolio Manager"): "portfolio_manager",
        _("Valuation Tools"): "valuation_tools",
        _("Risk Assessment"): "risk_assessment"
    }
    
    page_name = page_mapping.get(selected_page)
    
    if page_name:
        try:
            # Import and run the page module
            import importlib
            page_module = importlib.import_module(f"pages.{page_name}")
            page_module.show_page()
        except Exception as e:
            st.error(f"Error loading page: {str(e)}")
