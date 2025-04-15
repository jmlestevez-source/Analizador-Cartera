import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.translations import get_translation
from utils.data_loader import get_sp500_tickers, get_stock_data, get_stock_info

def show_page():
    """
    Display the Valuation Tools page with manual inputs
    """
    # Get translations based on selected language
    _ = get_translation(st.session_state.language)

    st.title(_("Valuation Tools"))

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
            _("Select a stock to value"),
            options=available_tickers,
            index=0 if len(available_tickers) > 0 else None
        )
    else:
        selected_ticker = st.text_input(
            _("Ingresar símbolo del ticker (ej: AAPL, MSFT, etc.)"), 
            value="AAPL"
        ).upper()

    if selected_ticker:
        # Get stock info for display
        stock_info = get_stock_info(selected_ticker)
        company_name = stock_info.get('longName', selected_ticker) if stock_info else selected_ticker

        st.subheader(f"{company_name} ({selected_ticker})")

        # Current market data
        current_price = stock_info.get('currentPrice', 0) if stock_info else 0

        if current_price > 0:
            st.metric(
                _("Current Price"),
                f"${current_price:.2f}"
            )

        # Valuation methods
        st.subheader(_("Valuation Methods"))

        # Tabs for different valuation methods
        dcf_tab, ddm_tab, relative_tab, capm_tab = st.tabs([
            _("Discounted Cash Flow (DCF)"),
            _("Dividend Discount Model"),
            _("Relative Valuation"),
            _("Cost of Equity (CAPM)")
        ])

        # DCF Valuation
        with dcf_tab:
            st.write(_("Discounted Cash Flow (DCF) Valuation"))
            st.latex(r'''
            \text{Enterprise Value} = \sum_{t=1}^{n} \frac{FCF_t}{(1 + r)^t} + \frac{TV}{(1 + r)^n}
            ''')

            col1, col2 = st.columns(2)

            with col1:
                current_fcf = st.number_input(_("Current Free Cash Flow ($)"), value=1000000.0, step=100000.0)
                growth_rate = st.slider(_("Growth Rate (%)"), min_value=1.0, max_value=30.0, value=10.0, step=1.0) / 100
                terminal_growth = st.slider(_("Terminal Growth (%)"), min_value=1.0, max_value=5.0, value=2.0, step=0.1) / 100

            with col2:
                discount_rate = st.slider(_("Discount Rate (%)"), min_value=5, max_value=20, value=10, step=1) / 100
                forecast_years = st.slider(_("Forecast Years"), min_value=3, max_value=10, value=5, step=1)
                total_debt = st.number_input(_("Total Debt ($)"), value=0.0, step=100000.0)
                cash = st.number_input(_("Cash & Equivalents ($)"), value=0.0, step=100000.0)
                shares_outstanding = st.number_input(_("Shares Outstanding (millions)"), value=100.0, step=1.0)

            if st.button(_("Calculate DCF"), key="dcf_button"):
                # Calculate projected FCF
                projected_fcf = []
                for year in range(1, forecast_years + 1):
                    fcf = current_fcf * (1 + growth_rate) ** year
                    projected_fcf.append(fcf)

                # Calculate terminal value
                terminal_value = projected_fcf[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)

                # Calculate present values
                pv_fcf = []
                for year, fcf in enumerate(projected_fcf, start=1):
                    pv = fcf / (1 + discount_rate) ** year
                    pv_fcf.append(pv)

                # Calculate enterprise and equity values
                enterprise_value = sum(pv_fcf) + (terminal_value / (1 + discount_rate) ** forecast_years)
                equity_value = enterprise_value - total_debt + cash
                fair_value = equity_value / shares_outstanding

                # Get current price from yfinance
                try:
                    import yfinance as yf
                    stock = yf.Ticker(selected_ticker)
                    current_price = stock.info.get('regularMarketPrice', 0.0)
                    if not current_price:
                        current_price = stock.history(period='1d')['Close'].iloc[-1]
                except Exception as e:
                    st.error(f"Error fetching current price: {e}")
                    current_price = 0.0

                # Display results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(_("Fair Value per Share"), f"${fair_value:,.2f}")
                with col2:
                    st.metric(_("Current Market Price"), f"${current_price:,.2f}")
                with col3:
                    if current_price > 0:
                        mos_percentage = ((fair_value - current_price) / current_price) * 100
                        valuation_status = _("Undervalued") if mos_percentage > 0 else _("Overvalued")
                        st.metric(
                            _("Valuation Status"),
                            f"{valuation_status} ({abs(mos_percentage):.2f}%)",
                            delta=f"{mos_percentage:.2f}%",
                            delta_color="normal"
                        )

        # Dividend Discount Model
        with ddm_tab:
            st.write(_("Dividend Discount Model (DDM) Valuation"))
            st.latex(r'''
            \text{Fair Value} = \frac{D_0 \times (1 + g)}{r - g}
            ''')

            col1, col2 = st.columns(2)

            with col1:
                current_dividend = st.number_input(_("Current Annual Dividend ($)"), value=1.0, step=0.1)
                dividend_growth = st.slider(_("Dividend Growth Rate (%)"), min_value=0.0, max_value=15.0, value=5.0, step=0.5) / 100

            with col2:
                required_return = st.slider(_("Required Return (%)"), min_value=5.0, max_value=20.0, value=10.0, step=0.5) / 100

            if st.button(_("Calculate DDM"), key="ddm_button"):
                if required_return <= dividend_growth:
                    st.error(_("Required return must be greater than growth rate"))
                else:
                    fair_value = (current_dividend * (1 + dividend_growth)) / (required_return - dividend_growth)

                    try:
                        import yfinance as yf
                        stock = yf.Ticker(selected_ticker)
                        current_price = stock.info.get('regularMarketPrice', 0.0)
                        if not current_price:
                            current_price = stock.history(period='1d')['Close'].iloc[-1]
                    except Exception as e:
                        st.error(f"Error fetching current price: {e}")
                        current_price = 0.0

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(_("Fair Value"), f"${fair_value:.2f}")
                    with col2:
                        st.metric(_("Current Market Price"), f"${current_price:.2f}")
                    with col3:
                        if current_price > 0:
                            mos_percentage = ((fair_value - current_price) / current_price) * 100
                            valuation_status = _("Undervalued") if mos_percentage > 0 else _("Overvalued")
                            st.metric(
                                _("Valuation Status"),
                                f"{valuation_status} ({abs(mos_percentage):.2f}%)",
                                delta=f"{mos_percentage:.2f}%",
                                delta_color="normal"
                            )

        # Relative Valuation
        with relative_tab:
            st.write(_("Relative Valuation using Multiples"))
            st.latex(r'''
            \text{Fair Value} = \text{Multiple} \times \text{Metric}
            ''')

            multiple_type = st.selectbox(
                _("Multiple Type"),
                options=["P/E", "P/S", "P/B", "EV/EBITDA"]
            )

            col1, col2 = st.columns(2)

            with col1:
                if multiple_type == "P/E":
                    metric = st.number_input(_("Earnings per Share ($)"), value=1.0, step=0.1)
                elif multiple_type == "P/S":
                    metric = st.number_input(_("Sales per Share ($)"), value=1.0, step=0.1)
                elif multiple_type == "P/B":
                    metric = st.number_input(_("Book Value per Share ($)"), value=1.0, step=0.1)
                else:  # EV/EBITDA
                    metric = st.number_input(_("EBITDA per Share ($)"), value=1.0, step=0.1)

            with col2:
                multiple = st.number_input(_("Target Multiple"), value=15.0, step=0.1)

            if st.button(_("Calculate Fair Value"), key="multiple_button"):
                fair_value = metric * multiple

                try:
                    import yfinance as yf
                    stock = yf.Ticker(selected_ticker)
                    current_price = stock.info.get('regularMarketPrice', 0.0)
                    if not current_price:
                        current_price = stock.history(period='1d')['Close'].iloc[-1]
                except Exception as e:
                    st.error(f"Error fetching current price: {e}")
                    current_price = 0.0

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(_("Fair Value"), f"${fair_value:.2f}")
                with col2:
                    st.metric(_("Current Market Price"), f"${current_price:.2f}")
                with col3:
                    if current_price > 0:
                        mos_percentage = ((fair_value - current_price) / current_price) * 100
                        valuation_status = _("Undervalued") if mos_percentage > 0 else _("Overvalued")
                        st.metric(
                            _("Valuation Status"),
                            f"{valuation_status} ({abs(mos_percentage):.2f}%)",
                            delta=f"{mos_percentage:.2f}%",
                            delta_color="normal"
                        )

        # CAPM (Cost of Equity)
        with capm_tab:
            st.write(_("Capital Asset Pricing Model (CAPM)"))

            # CAPM parameters
            capm_period = st.selectbox(
                _("Data Period for Beta Calculation"),
                options=["1y", "2y", "5y"],
                index=2
            )

            # Calculate CAPM button
            if st.button(_("Calculate CAPM"), key="capm_button"):
                with st.spinner(_("Calculating cost of equity...")):
                    capm_results = calculate_capm(
                        selected_ticker,
                        period=capm_period
                    )

                    if "error" not in capm_results:
                        st.subheader(_("CAPM Results"))

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric(
                                _("Cost of Equity"),
                                f"{capm_results['cost_of_equity'] * 100:.2f}%"
                            )

                        with col2:
                            st.metric(
                                _("Beta"),
                                f"{capm_results['beta']:.2f}"
                            )

                        with col3:
                            st.metric(
                                _("Risk-Free Rate"),
                                f"{capm_results['risk_free_rate'] * 100:.2f}%"
                            )

                        # Display CAPM formula
                        st.subheader(_("CAPM Formula"))

                        st.latex(r'''
                        r_e = r_f + \beta \times (r_m - r_f)
                        ''')

                        st.write(_("Where:"))
                        st.write(f"- $r_e$ = {_('Cost of Equity')}: {capm_results['cost_of_equity'] * 100:.2f}%")
                        st.write(f"- $r_f$ = {_('Risk-Free Rate')}: {capm_results['risk_free_rate'] * 100:.2f}%")
                        st.write(f"- $\\beta$ = {_('Beta')}: {capm_results['beta']:.2f}")
                        st.write(f"- $r_m - r_f$ = {_('Market Risk Premium')}: {capm_results['market_risk_premium'] * 100:.2f}%")

                        # Risk interpretation based on beta
                        st.subheader(_("Risk Interpretation"))

                        if capm_results['beta'] < 0.8:
                            st.write(_("Low risk: The stock is less volatile than the market."))
                        elif capm_results['beta'] < 1.2:
                            st.write(_("Medium risk: The stock has similar volatility to the market."))
                        else:
                            st.write(_("High risk: The stock is more volatile than the market."))
                    else:
                        st.error(_("Error in CAPM calculation: {}").format(capm_results['error']))

def calculate_capm(ticker, period="5y"):
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if data.empty:
            return {"error": "No data found for this ticker and period"}

        # Calculate beta using a linear regression
        market_data = yf.download("^GSPC", period=period) # S&P 500 as market proxy
        if market_data.empty:
            return {"error": "No market data found for this period"}

        # Assuming daily data
        stock_returns = data['Close'].pct_change().dropna()
        market_returns = market_data['Close'].pct_change().dropna()

        beta, alpha = np.polyfit(market_returns, stock_returns, 1)

        # Get risk-free rate (e.g., 10-year Treasury yield)
        # In real scenario, obtain from a reliable financial source or API.
        risk_free_rate = 0.05  # Placeholder for demonstration

        # Assume market risk premium (historical average)
        market_risk_premium = 0.07 # Placeholder for demonstration

        # Calculate cost of equity
        cost_of_equity = risk_free_rate + beta * market_risk_premium

        capm_results = {
            "cost_of_equity": cost_of_equity,
            "beta": beta,
            "risk_free_rate": risk_free_rate,
            "market_risk_premium": market_risk_premium
        }
        return capm_results
    except Exception as e:
        return {"error": str(e)}