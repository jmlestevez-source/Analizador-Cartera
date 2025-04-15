import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

from utils.data_loader import get_sp500_tickers, get_stock_data
from utils.analysis import (
    calculate_returns, calculate_volatility, 
    calculate_var, calculate_cvar, calculate_max_drawdown,
    calculate_beta, calculate_sharpe_ratio, calculate_drawdown
)
from utils.visualization import plot_drawdown, plot_risk_return_scatter
from utils.portfolio import get_portfolio_returns, get_portfolio_stats
from utils.translations import get_translation

def show_page():
    """
    Display the Risk Assessment page
    """
    # Get translations based on selected language
    _ = get_translation(st.session_state.language)
    
    st.title(_("Risk Assessment Tool"))
    
    # Create tabs for different risk assessment tools
    portfolio_tab, var_tab, stress_tab, monte_carlo_tab = st.tabs([
        _("Portfolio Risk Metrics"),
        _("Value at Risk (VaR)"),
        _("Stress Testing"),
        _("Monte Carlo Simulation")
    ])
    
    with portfolio_tab:
        st.subheader(_("Portfolio Risk Analysis"))
        
        # Check if portfolio exists
        if st.session_state.portfolio.empty:
            st.warning(_("Please create a portfolio in the Dashboard first"))
        else:
            # Display portfolio risk metrics
            try:
                portfolio_stats = get_portfolio_stats(
                    st.session_state.portfolio,
                    period=st.session_state.period
                )
                
                # Display key risk metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        _("Volatility"),
                        f"{portfolio_stats['volatility']:.2f}%"
                    )
                
                with col2:
                    st.metric(
                        _("Max Drawdown"),
                        f"{portfolio_stats['max_drawdown']:.2f}%"
                    )
                
                with col3:
                    st.metric(
                        _("Beta"),
                        f"{portfolio_stats['beta']:.2f}"
                    )
                
                with col4:
                    st.metric(
                        _("Sharpe Ratio"),
                        f"{portfolio_stats['sharpe_ratio']:.2f}"
                    )
                
                # Get portfolio returns
                portfolio_returns = get_portfolio_returns(
                    st.session_state.portfolio,
                    period=st.session_state.period
                )
                
                if len(portfolio_returns) > 0:
                    # Calculate additional risk metrics
                    var_95 = calculate_var(portfolio_returns) * 100
                    cvar_95 = calculate_cvar(portfolio_returns) * 100
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            _("VaR (95%)"),
                            f"{var_95:.2f}%"
                        )
                    
                    with col2:
                        st.metric(
                            _("CVaR (95%)"),
                            f"{cvar_95:.2f}%"
                        )
                    
                    # Plot portfolio drawdown
                    st.subheader(_("Portfolio Drawdown"))
                    
                    # Create drawdown chart
                    drawdown = calculate_drawdown(portfolio_returns)
                    
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=drawdown.index,
                            y=drawdown,
                            fill='tozeroy',
                            line=dict(color='rgba(200, 0, 0, 0.8)', width=1),
                            name=_("Drawdown")
                        )
                    )
                    
                    fig.update_layout(
                        title=_("Portfolio Drawdown Over Time"),
                        xaxis_title=_("Date"),
                        yaxis_title=_("Drawdown"),
                        height=500
                    )
                    
                    fig.update_yaxes(tickformat=".2%")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Portfolio returns distribution
                    st.subheader(_("Returns Distribution"))
                    
                    # Create histogram of returns
                    fig = go.Figure()
                    fig.add_trace(
                        go.Histogram(
                            x=portfolio_returns,
                            nbinsx=30,
                            name=_("Returns"),
                            marker_color='rgba(0, 0, 200, 0.7)'
                        )
                    )
                    
                    # Add normal distribution curve for comparison
                    mean = portfolio_returns.mean()
                    std = portfolio_returns.std()
                    x = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 100)
                    y = stats.norm.pdf(x, mean, std)
                    
                    # Scale the normal distribution to match histogram
                    y = y * (len(portfolio_returns) * (portfolio_returns.max() - portfolio_returns.min()) / 30)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y,
                            mode='lines',
                            name=_("Normal Distribution"),
                            line=dict(color='rgba(255, 0, 0, 0.8)', width=2)
                        )
                    )
                    
                    # Add vertical lines for VaR and CVaR
                    fig.add_vline(
                        x=var_95/100,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"VaR (95%): {var_95:.2f}%",
                        annotation_position="top right"
                    )
                    
                    fig.add_vline(
                        x=cvar_95/100,
                        line_dash="dash",
                        line_color="darkred",
                        annotation_text=f"CVaR (95%): {cvar_95:.2f}%",
                        annotation_position="bottom right"
                    )
                    
                    fig.update_layout(
                        title=_("Distribution of Portfolio Returns"),
                        xaxis_title=_("Return"),
                        yaxis_title=_("Frequency"),
                        height=500,
                        bargap=0.01
                    )
                    
                    # Format x-axis as percentages
                    fig.update_xaxes(tickformat=".2%")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Risk contribution analysis
                    st.subheader(_("Risk Contribution Analysis"))
                    
                    # Get individual stock returns
                    stock_returns = {}
                    tickers = st.session_state.portfolio['Ticker'].tolist()
                    
                    for ticker in tickers:
                        stock_data = get_stock_data(ticker, period=st.session_state.period)
                        if not stock_data.empty:
                            returns = calculate_returns(stock_data['Close'])
                            stock_returns[ticker] = returns
                    
                    # Calculate volatility for each stock
                    volatilities = {}
                    portfolio_weights = {}
                    total_value = st.session_state.portfolio['Value'].sum()
                    
                    for ticker, row in zip(st.session_state.portfolio['Ticker'], st.session_state.portfolio.iterrows()):
                        if ticker in stock_returns:
                            volatilities[ticker] = calculate_volatility(stock_returns[ticker]) * 100
                            portfolio_weights[ticker] = row[1]['Value'] / total_value
                    
                    # Create risk contribution data
                    risk_data = pd.DataFrame({
                        'Ticker': list(volatilities.keys()),
                        'Weight': [portfolio_weights.get(ticker, 0) for ticker in volatilities.keys()],
                        'Volatility': list(volatilities.values())
                    })
                    
                    # Calculate weighted risk
                    risk_data['Weighted Risk'] = risk_data['Weight'] * risk_data['Volatility']
                    risk_data['Risk Contribution'] = risk_data['Weighted Risk'] / risk_data['Weighted Risk'].sum()
                    
                    # Sort by risk contribution
                    risk_data = risk_data.sort_values('Risk Contribution', ascending=False)
                    
                    # Plot risk contribution
                    fig = px.bar(
                        risk_data,
                        x='Ticker',
                        y='Risk Contribution',
                        title=_("Risk Contribution by Asset"),
                        color='Volatility',
                        color_continuous_scale='Reds',
                        text_auto='.1%'
                    )
                    
                    fig.update_layout(
                        xaxis_title=_("Ticker"),
                        yaxis_title=_("Risk Contribution"),
                        height=500,
                        coloraxis_colorbar=dict(title=_("Volatility (%)"))
                    )
                    
                    # Format y-axis as percentages
                    fig.update_yaxes(tickformat=".1%")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display risk contribution table
                    st.write(_("Risk Contribution Details"))
                    
                    # Format the table data
                    display_data = risk_data.copy()
                    display_data['Weight'] = display_data['Weight'].apply(lambda x: f"{x:.2%}")
                    display_data['Volatility'] = display_data['Volatility'].apply(lambda x: f"{x:.2f}%")
                    display_data['Weighted Risk'] = display_data['Weighted Risk'].apply(lambda x: f"{x:.2f}%")
                    display_data['Risk Contribution'] = display_data['Risk Contribution'].apply(lambda x: f"{x:.2%}")
                    
                    st.dataframe(display_data, use_container_width=True)
                else:
                    st.warning(_("Not enough return data for analysis"))
            except Exception as e:
                st.error(_("Error in risk analysis: {}").format(str(e)))
    
    with var_tab:
        st.subheader(_("Value at Risk (VaR) Analysis"))
        
        # VaR methodology selection
        var_method = st.selectbox(
            _("VaR Methodology"),
            options=[_("Historical"), _("Parametric"), _("Monte Carlo")],
            index=0
        )
        
        # VaR parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            confidence_level = st.slider(
                _("Confidence Level"),
                min_value=80,
                max_value=99,
                value=95,
                step=1
            ) / 100
        
        with col2:
            time_horizon = st.selectbox(
                _("Time Horizon"),
                options=[_("1 Day"), _("1 Week"), _("1 Month")],
                index=0
            )
        
        with col3:
            investment_amount = st.number_input(
                _("Investment Amount ($)"),
                min_value=1000,
                value=100000,
                step=1000
            )
        
        # Map time horizon selection to number of days
        if time_horizon == _("1 Day"):
            horizon_days = 1
        elif time_horizon == _("1 Week"):
            horizon_days = 5
        else:  # 1 Month
            horizon_days = 21
        
        # Check if portfolio exists or use selected stock
        if not st.session_state.portfolio.empty:
            # Use portfolio for VaR calculation
            data_source = st.radio(
                _("Data Source"),
                options=[_("Portfolio"), _("Individual Stock")],
                index=0
            )
            
            if data_source == _("Portfolio"):
                returns = get_portfolio_returns(
                    st.session_state.portfolio,
                    period=st.session_state.period
                )
                asset_name = _("Portfolio")
            else:
                # Stock selection for individual VaR
                available_tickers = get_sp500_tickers()
                selected_ticker = st.selectbox(
                    _("Select a stock"),
                    options=available_tickers,
                    index=0 if len(available_tickers) > 0 else None
                )
                
                stock_data = get_stock_data(selected_ticker, period=st.session_state.period)
                returns = calculate_returns(stock_data['Close'])
                asset_name = selected_ticker
        else:
            # Stock selection for individual VaR
            available_tickers = get_sp500_tickers()
            selected_ticker = st.selectbox(
                _("Select a stock"),
                options=available_tickers,
                index=0 if len(available_tickers) > 0 else None
            )
            
            stock_data = get_stock_data(selected_ticker, period=st.session_state.period)
            returns = calculate_returns(stock_data['Close'])
            asset_name = selected_ticker
        
        # Calculate VaR button
        if st.button(_("Calculate VaR"), key="var_button"):
            if len(returns) == 0:
                st.warning(_("Not enough return data for analysis"))
            else:
                # Historical VaR calculation
                if var_method == _("Historical"):
                    # Calculate VaR
                    var_pct = np.percentile(returns, (1 - confidence_level) * 100) * np.sqrt(horizon_days)
                    var_amount = investment_amount * abs(var_pct)
                    
                    # Display results
                    st.subheader(_("Historical VaR Results"))
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            _("VaR (as % of investment)"),
                            f"{abs(var_pct) * 100:.2f}%"
                        )
                    
                    with col2:
                        st.metric(
                            _("VaR (in $)"),
                            f"${var_amount:,.2f}"
                        )
                    
                    # Create histogram of returns with VaR
                    fig = go.Figure()
                    fig.add_trace(
                        go.Histogram(
                            x=returns,
                            nbinsx=30,
                            name=_("Returns"),
                            marker_color='rgba(0, 0, 200, 0.7)'
                        )
                    )
                    
                    # Add VaR line
                    fig.add_vline(
                        x=var_pct / np.sqrt(horizon_days),
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"VaR ({confidence_level:.1%}): {abs(var_pct) * 100:.2f}%",
                        annotation_position="top right"
                    )
                    
                    fig.update_layout(
                        title=_("Historical Returns Distribution with VaR for {}").format(asset_name),
                        xaxis_title=_("Return"),
                        yaxis_title=_("Frequency"),
                        height=500,
                        bargap=0.01
                    )
                    
                    # Format x-axis as percentages
                    fig.update_xaxes(tickformat=".2%")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # VaR interpretation
                    st.subheader(_("VaR Interpretation"))
                    
                    st.write(_(
                        "With {:.1f}% confidence, we can say that the maximum loss over a {} period "
                        "will not exceed ${:,.2f} or {:.2f}% of your investment."
                    ).format(
                        confidence_level * 100,
                        time_horizon.lower(),
                        var_amount,
                        abs(var_pct) * 100
                    ))
                
                # Parametric VaR calculation
                elif var_method == _("Parametric"):
                    # Calculate mean and standard deviation
                    mean_return = returns.mean()
                    std_return = returns.std()
                    
                    # Calculate z-score based on confidence level
                    z_score = stats.norm.ppf(1 - confidence_level)
                    
                    # Calculate VaR
                    var_pct = (mean_return + z_score * std_return) * np.sqrt(horizon_days)
                    var_amount = investment_amount * abs(var_pct)
                    
                    # Display results
                    st.subheader(_("Parametric VaR Results"))
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            _("VaR (as % of investment)"),
                            f"{abs(var_pct) * 100:.2f}%"
                        )
                    
                    with col2:
                        st.metric(
                            _("VaR (in $)"),
                            f"${var_amount:,.2f}"
                        )
                    
                    # Create normal distribution plot with VaR
                    x = np.linspace(mean_return - 4 * std_return, mean_return + 4 * std_return, 1000)
                    y = stats.norm.pdf(x, mean_return, std_return)
                    
                    fig = go.Figure()
                    
                    # Add normal distribution curve
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y,
                            mode='lines',
                            name=_("Normal Distribution"),
                            line=dict(color='blue', width=2),
                            fill='tozeroy',
                            fillcolor='rgba(0, 0, 200, 0.2)'
                        )
                    )
                    
                    # Fill the VaR area
                    var_x = x[x <= mean_return + z_score * std_return]
                    var_y = stats.norm.pdf(var_x, mean_return, std_return)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=var_x,
                            y=var_y,
                            mode='lines',
                            name=_("VaR Region"),
                            line=dict(color='red', width=0),
                            fill='tozeroy',
                            fillcolor='rgba(255, 0, 0, 0.3)'
                        )
                    )
                    
                    # Add VaR line
                    fig.add_vline(
                        x=mean_return + z_score * std_return,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"VaR ({confidence_level:.1%}): {abs(var_pct) * 100 / np.sqrt(horizon_days):.2f}% (daily)",
                        annotation_position="top right"
                    )
                    
                    fig.update_layout(
                        title=_("Parametric VaR for {}").format(asset_name),
                        xaxis_title=_("Return"),
                        yaxis_title=_("Probability Density"),
                        height=500
                    )
                    
                    # Format x-axis as percentages
                    fig.update_xaxes(tickformat=".2%")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # VaR interpretation
                    st.subheader(_("VaR Interpretation"))
                    
                    st.write(_(
                        "With {:.1f}% confidence, we can say that the maximum loss over a {} period "
                        "will not exceed ${:,.2f} or {:.2f}% of your investment, assuming returns are normally distributed."
                    ).format(
                        confidence_level * 100,
                        time_horizon.lower(),
                        var_amount,
                        abs(var_pct) * 100
                    ))
                
                # Monte Carlo VaR calculation
                else:  # Monte Carlo
                    # Set number of simulations
                    num_simulations = 10000
                    
                    # Calculate mean and standard deviation
                    mean_return = returns.mean()
                    std_return = returns.std()
                    
                    # Generate random returns
                    np.random.seed(42)  # For reproducibility
                    random_returns = np.random.normal(mean_return, std_return, num_simulations)
                    
                    # Apply time horizon
                    random_returns = random_returns * np.sqrt(horizon_days)
                    
                    # Calculate VaR
                    var_pct = np.percentile(random_returns, (1 - confidence_level) * 100)
                    var_amount = investment_amount * abs(var_pct)
                    
                    # Display results
                    st.subheader(_("Monte Carlo VaR Results"))
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            _("VaR (as % of investment)"),
                            f"{abs(var_pct) * 100:.2f}%"
                        )
                    
                    with col2:
                        st.metric(
                            _("VaR (in $)"),
                            f"${var_amount:,.2f}"
                        )
                    
                    # Create histogram of simulated returns with VaR
                    fig = go.Figure()
                    fig.add_trace(
                        go.Histogram(
                            x=random_returns,
                            nbinsx=50,
                            name=_("Simulated Returns"),
                            marker_color='rgba(0, 200, 0, 0.7)'
                        )
                    )
                    
                    # Add VaR line
                    fig.add_vline(
                        x=var_pct,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"VaR ({confidence_level:.1%}): {abs(var_pct) * 100:.2f}%",
                        annotation_position="top right"
                    )
                    
                    fig.update_layout(
                        title=_("Monte Carlo Simulated Returns with VaR for {}").format(asset_name),
                        xaxis_title=_("Return"),
                        yaxis_title=_("Frequency"),
                        height=500,
                        bargap=0.01
                    )
                    
                    # Format x-axis as percentages
                    fig.update_xaxes(tickformat=".2%")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # VaR interpretation
                    st.subheader(_("VaR Interpretation"))
                    
                    st.write(_(
                        "Based on {n:,} Monte Carlo simulations, with {p:.1f}% confidence, "
                        "we can say that the maximum loss over a {t} period "
                        "will not exceed ${a:,.2f} or {pct:.2f}% of your investment."
                    ).format(
                        n=num_simulations,
                        p=confidence_level * 100,
                        t=time_horizon.lower(),
                        a=var_amount,
                        pct=abs(var_pct) * 100
                    ))
    
    with stress_tab:
        st.subheader(_("Stress Testing"))
        
        # Historical crisis scenarios
        st.write(_("Historical Crisis Scenarios"))
        
        historical_scenarios = {
            _("2008 Financial Crisis"): (-0.40, "2008-09-01", "2009-03-01"),  # ~40% drop
            _("Covid-19 Crash"): (-0.30, "2020-02-15", "2020-03-23"),  # ~30% drop
            _("2000 Dot-com Bubble"): (-0.45, "2000-03-01", "2002-10-01"),  # ~45% drop
            _("2018 Q4 Selloff"): (-0.20, "2018-10-01", "2018-12-24"),  # ~20% drop
        }
        
        selected_scenario = st.selectbox(
            _("Select Historical Scenario"),
            options=list(historical_scenarios.keys()),
            index=0
        )
        
        # Custom stress scenario
        st.write(_("Custom Stress Scenario"))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            market_change = st.slider(
                _("Market Change (%)"),
                min_value=-50.0,
                max_value=0.0,
                value=-20.0,
                step=5.0
            ) / 100
        
        with col2:
            interest_rate_change = st.slider(
                _("Interest Rate Change (%)"),
                min_value=-2.0,
                max_value=5.0,
                value=1.0,
                step=0.25
            )
        
        with col3:
            volatility_increase = st.slider(
                _("Volatility Increase (%)"),
                min_value=0.0,
                max_value=200.0,
                value=50.0,
                step=10.0
            ) / 100
        
        # Run stress test button
        if st.button(_("Run Stress Test"), key="stress_button"):
            # Check if portfolio exists
            if st.session_state.portfolio.empty:
                st.warning(_("Please create a portfolio in the Dashboard first"))
            else:
                # Get portfolio data
                portfolio_value = st.session_state.portfolio['Value'].sum()
                tickers = st.session_state.portfolio['Ticker'].tolist()
                
                # Get beta for each stock
                betas = {}
                
                # Get S&P 500 data
                sp500_data = get_stock_data("^GSPC", period=st.session_state.period)
                sp500_returns = calculate_returns(sp500_data['Close'])
                
                for ticker in tickers:
                    stock_data = get_stock_data(ticker, period=st.session_state.period)
                    returns = calculate_returns(stock_data['Close'])
                    
                    # Align dates
                    common_idx = returns.index.intersection(sp500_returns.index)
                    if len(common_idx) > 0:
                        stock_returns_aligned = returns.loc[common_idx]
                        market_returns_aligned = sp500_returns.loc[common_idx]
                        
                        # Calculate beta
                        beta = calculate_beta(stock_returns_aligned, market_returns_aligned)
                        betas[ticker] = beta
                    else:
                        betas[ticker] = 1.0  # Default to market beta
                
                # Calculate impact based on historical scenario
                if selected_scenario in historical_scenarios:
                    scenario_impact, start_date, end_date = historical_scenarios[selected_scenario]
                    
                    # Get historical performance during the scenario
                    scenario_impacts = {}
                    scenario_betas = {}
                    
                    for ticker in tickers:
                        try:
                            # Get stock data for the crisis period
                            stock_data = get_stock_data(ticker, start=start_date, end=end_date)
                            
                            if not stock_data.empty and len(stock_data) > 1:
                                start_price = stock_data['Close'].iloc[0]
                                end_price = stock_data['Close'].iloc[-1]
                                ticker_impact = (end_price / start_price) - 1
                                scenario_impacts[ticker] = ticker_impact
                            else:
                                # If no data for that period, use beta approximation
                                scenario_impacts[ticker] = scenario_impact * betas.get(ticker, 1.0)
                                
                            scenario_betas[ticker] = betas.get(ticker, 1.0)
                        except Exception:
                            scenario_impacts[ticker] = scenario_impact * betas.get(ticker, 1.0)
                            scenario_betas[ticker] = betas.get(ticker, 1.0)
                    
                    # Calculate portfolio impact
                    weighted_impacts = []
                    for ticker, row in zip(st.session_state.portfolio['Ticker'], st.session_state.portfolio.iterrows()):
                        weight = row[1]['Value'] / portfolio_value
                        impact = scenario_impacts.get(ticker, scenario_impact * betas.get(ticker, 1.0))
                        weighted_impacts.append(weight * impact)
                    
                    portfolio_impact = sum(weighted_impacts)
                    dollar_impact = portfolio_value * portfolio_impact
                    
                    # Display historical scenario results
                    st.subheader(_("Historical Scenario Results"))
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            _("Portfolio Impact (%)"),
                            f"{portfolio_impact * 100:.2f}%",
                            delta=None
                        )
                    
                    with col2:
                        st.metric(
                            _("Dollar Impact"),
                            f"${dollar_impact:,.2f}",
                            delta=None
                        )
                    
                    # Stock-by-stock impact
                    impact_data = []
                    
                    for ticker, row in zip(st.session_state.portfolio['Ticker'], st.session_state.portfolio.iterrows()):
                        impact = scenario_impacts.get(ticker, scenario_impact * betas.get(ticker, 1.0))
                        beta = scenario_betas.get(ticker, 1.0)
                        dollar_value = row[1]['Value']
                        dollar_loss = dollar_value * impact
                        
                        impact_data.append({
                            'Ticker': ticker,
                            'Beta': beta,
                            'Impact (%)': impact * 100,
                            'Current Value ($)': dollar_value,
                            'Value Change ($)': dollar_loss
                        })
                    
                    impact_df = pd.DataFrame(impact_data)
                    
                    # Create bar chart of impacts
                    fig = px.bar(
                        impact_df,
                        x='Ticker',
                        y='Impact (%)',
                        title=_("Impact by Stock in {} Scenario").format(selected_scenario),
                        color='Beta',
                        color_continuous_scale='RdBu_r',
                        text_auto='.1f'
                    )
                    
                    fig.update_layout(
                        xaxis_title=_("Ticker"),
                        yaxis_title=_("Impact (%)"),
                        height=500,
                        coloraxis_colorbar=dict(title='Beta')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display impact table
                    st.write(_("Stock-by-Stock Impact"))
                    
                    # Format the DataFrame for display
                    impact_df_display = impact_df.copy()
                    impact_df_display['Beta'] = impact_df_display['Beta'].apply(lambda x: f"{x:.2f}")
                    impact_df_display['Impact (%)'] = impact_df_display['Impact (%)'].apply(lambda x: f"{x:.2f}%")
                    impact_df_display['Current Value ($)'] = impact_df_display['Current Value ($)'].apply(lambda x: f"${x:,.2f}")
                    impact_df_display['Value Change ($)'] = impact_df_display['Value Change ($)'].apply(lambda x: f"${x:,.2f}")
                    
                    st.dataframe(impact_df_display, use_container_width=True)
                
                # Calculate custom scenario impact
                st.subheader(_("Custom Scenario Results"))
                
                # Calculate portfolio impact based on betas
                weighted_impacts = []
                
                for ticker, row in zip(st.session_state.portfolio['Ticker'], st.session_state.portfolio.iterrows()):
                    weight = row[1]['Value'] / portfolio_value
                    beta = betas.get(ticker, 1.0)
                    impact = market_change * beta
                    
                    # Adjust for interest rate sensitivity (simplified)
                    # Higher beta stocks often more affected by interest rates
                    if beta > 1.2:
                        ir_sensitivity = -0.05  # High sensitivity
                    elif beta > 0.8:
                        ir_sensitivity = -0.03  # Medium sensitivity
                    else:
                        ir_sensitivity = -0.01  # Low sensitivity
                    
                    interest_impact = ir_sensitivity * interest_rate_change
                    total_impact = impact + interest_impact
                    
                    # Apply volatility increase effect (simplified)
                    # Assume higher volatility adds 10-30% additional negative impact
                    vol_effect = min(0.3, volatility_increase / 2)
                    if total_impact < 0:
                        total_impact = total_impact * (1 + vol_effect)
                    
                    weighted_impacts.append(weight * total_impact)
                
                portfolio_impact = sum(weighted_impacts)
                dollar_impact = portfolio_value * portfolio_impact
                
                # Display custom scenario results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        _("Portfolio Impact (%)"),
                        f"{portfolio_impact * 100:.2f}%",
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        _("Dollar Impact"),
                        f"${dollar_impact:,.2f}",
                        delta=None
                    )
                
                # Risk mitigation recommendations
                st.subheader(_("Risk Mitigation Recommendations"))
                
                if portfolio_impact < -0.25:
                    st.warning(_(
                        "Your portfolio may be highly vulnerable to this stress scenario. "
                        "Consider reducing exposure to high-beta stocks and increasing diversification."
                    ))
                elif portfolio_impact < -0.15:
                    st.info(_(
                        "Your portfolio has moderate vulnerability to this stress scenario. "
                        "Consider adding some defensive assets or hedges against market downturns."
                    ))
                else:
                    st.success(_(
                        "Your portfolio appears relatively resilient to this stress scenario. "
                        "Continue monitoring risk factors and maintaining diversification."
                    ))
    
    with monte_carlo_tab:
        st.subheader(_("Monte Carlo Simulation"))
        
        # Simulation parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            simulation_period = st.selectbox(
                _("Simulation Period"),
                options=[_("1 Month"), _("3 Months"), _("6 Months"), _("1 Year"), _("3 Years"), _("5 Years")],
                index=3
            )
        
        with col2:
            num_simulations = st.slider(
                _("Number of Simulations"),
                min_value=100,
                max_value=10000,
                value=1000,
                step=100
            )
        
        with col3:
            confidence_intervals = st.multiselect(
                _("Confidence Intervals"),
                options=["50%", "80%", "90%", "95%", "99%"],
                default=["50%", "90%"]
            )
        
        # Convert simulation period to trading days
        if simulation_period == _("1 Month"):
            sim_days = 21
        elif simulation_period == _("3 Months"):
            sim_days = 63
        elif simulation_period == _("6 Months"):
            sim_days = 126
        elif simulation_period == _("1 Year"):
            sim_days = 252
        elif simulation_period == _("3 Years"):
            sim_days = 756
        else:  # 5 Years
            sim_days = 1260
        
        # Run simulation button
        if st.button(_("Run Simulation"), key="monte_carlo_button"):
            # Check if portfolio exists or use selected stock
            if not st.session_state.portfolio.empty:
                # Use portfolio returns
                returns = get_portfolio_returns(
                    st.session_state.portfolio,
                    period=st.session_state.period
                )
                asset_name = _("Portfolio")
                initial_value = st.session_state.portfolio['Value'].sum()
            else:
                # Stock selection for simulation
                available_tickers = get_sp500_tickers()
                selected_ticker = st.selectbox(
                    _("Select a stock"),
                    options=available_tickers,
                    index=0 if len(available_tickers) > 0 else None,
                    key="monte_carlo_ticker"
                )
                
                stock_data = get_stock_data(selected_ticker, period=st.session_state.period)
                returns = calculate_returns(stock_data['Close'])
                asset_name = selected_ticker
                
                # Use current price as initial value
                initial_value = stock_data['Close'].iloc[-1] * 100  # Assume 100 shares
            
            if len(returns) < 30:
                st.warning(_("Not enough historical data for simulation"))
            else:
                # Calculate mean and standard deviation of returns
                mu = returns.mean()
                sigma = returns.std()
                
                # Set up the simulation
                np.random.seed(42)  # For reproducibility
                sim_returns = np.zeros((sim_days, num_simulations))
                
                # Generate random returns
                for i in range(num_simulations):
                    sim_returns[:, i] = np.random.normal(mu, sigma, sim_days)
                
                # Calculate cumulative returns
                cum_returns = np.zeros((sim_days, num_simulations))
                cum_returns[0, :] = initial_value
                
                for i in range(1, sim_days):
                    cum_returns[i, :] = cum_returns[i-1, :] * (1 + sim_returns[i, :])
                
                # Calculate percentiles for confidence intervals
                percentiles = {}
                for ci in confidence_intervals:
                    ci_value = float(ci.strip('%')) / 100
                    lower_percentile = (1 - ci_value) / 2
                    upper_percentile = 1 - lower_percentile
                    percentiles[ci] = (lower_percentile, upper_percentile)
                
                # Calculate percentile values at each time step
                percentile_values = {}
                for ci, (lower, upper) in percentiles.items():
                    lower_values = np.percentile(cum_returns, lower * 100, axis=1)
                    upper_values = np.percentile(cum_returns, upper * 100, axis=1)
                    percentile_values[ci] = (lower_values, upper_values)
                
                # Calculate median path
                median_path = np.percentile(cum_returns, 50, axis=1)
                
                # Plot the simulation results
                fig = go.Figure()
                
                # Add a subset of simulation paths (for better visualization)
                num_paths_to_show = min(100, num_simulations)
                for i in range(num_paths_to_show):
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(sim_days)),
                            y=cum_returns[:, i],
                            mode='lines',
                            line=dict(width=0.5, color='rgba(0, 0, 255, 0.05)'),
                            showlegend=False
                        )
                    )
                
                # Add confidence intervals
                for ci in confidence_intervals:
                    lower_values, upper_values = percentile_values[ci]
                    
                    # Add lower bound
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(sim_days)),
                            y=lower_values,
                            mode='lines',
                            line=dict(width=0, color='rgba(0, 0, 0, 0)'),
                            showlegend=False
                        )
                    )
                    
                    # Add upper bound
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(sim_days)),
                            y=upper_values,
                            mode='lines',
                            line=dict(width=0, color='rgba(0, 0, 0, 0)'),
                            fill='tonexty',
                            fillcolor=f'rgba(0, 100, 255, {0.1 if ci=="50%" else 0.05})',
                            name=f"{ci} Confidence Interval"
                        )
                    )
                
                # Add median path
                fig.add_trace(
                    go.Scatter(
                        x=list(range(sim_days)),
                        y=median_path,
                        mode='lines',
                        line=dict(width=2, color='blue'),
                        name=_("Median Path")
                    )
                )
                
                # Add initial value line
                fig.add_trace(
                    go.Scatter(
                        x=list(range(sim_days)),
                        y=[initial_value] * sim_days,
                        mode='lines',
                        line=dict(width=1.5, color='red', dash='dash'),
                        name=_("Initial Value")
                    )
                )
                
                # Format the x-axis as days/months
                if sim_days <= 63:  # Up to 3 months, show days
                    x_labels = [f"Day {i+1}" for i in range(0, sim_days, 5)]
                    x_positions = list(range(0, sim_days, 5))
                elif sim_days <= 252:  # Up to 1 year, show months
                    months_per_tick = 1
                    days_per_month = 21
                    x_labels = [f"Month {i+1}" for i in range(0, sim_days // days_per_month + 1, months_per_tick)]
                    x_positions = [i * days_per_month for i in range(0, sim_days // days_per_month + 1, months_per_tick)]
                else:  # Multiple years, show years and months
                    days_per_month = 21
                    months_per_year = 12
                    days_per_year = days_per_month * months_per_year
                    
                    years = sim_days // days_per_year
                    remaining_months = (sim_days % days_per_year) // days_per_month
                    
                    x_labels = []
                    x_positions = []
                    
                    for year in range(years + 1):
                        if year < years:
                            x_labels.append(f"Year {year + 1}")
                            x_positions.append(year * days_per_year)
                        else:
                            if remaining_months > 0:
                                x_labels.append(f"Year {year + 1}")
                                x_positions.append(year * days_per_year)
                
                # Update layout
                fig.update_layout(
                    title=_("Monte Carlo Simulation for {} - {} Paths").format(asset_name, num_simulations),
                    xaxis=dict(
                        title=_("Time"),
                        tickvals=x_positions,
                        ticktext=x_labels
                    ),
                    yaxis=dict(
                        title=_("Value ($)"),
                        tickformat="$,.0f"
                    ),
                    height=600,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate final value statistics
                final_values = cum_returns[-1, :]
                mean_final = np.mean(final_values)
                median_final = np.median(final_values)
                
                # Probability of profit/loss
                prob_profit = np.mean(final_values > initial_value) * 100
                prob_loss = 100 - prob_profit
                
                # Expected value
                expected_return_pct = (mean_final / initial_value - 1) * 100
                
                # VaR at different confidence levels
                var_90 = initial_value - np.percentile(final_values, 10)
                var_95 = initial_value - np.percentile(final_values, 5)
                var_99 = initial_value - np.percentile(final_values, 1)
                
                # Display statistics
                st.subheader(_("Simulation Statistics"))
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        _("Mean Final Value"),
                        f"${mean_final:,.2f}",
                        delta=f"{(mean_final/initial_value - 1) * 100:.2f}%",
                        delta_color="normal"
                    )
                
                with col2:
                    st.metric(
                        _("Median Final Value"),
                        f"${median_final:,.2f}",
                        delta=f"{(median_final/initial_value - 1) * 100:.2f}%",
                        delta_color="normal"
                    )
                
                with col3:
                    st.metric(
                        _("Initial Value"),
                        f"${initial_value:,.2f}"
                    )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        _("Probability of Profit"),
                        f"{prob_profit:.1f}%"
                    )
                
                with col2:
                    st.metric(
                        _("Probability of Loss"),
                        f"{prob_loss:.1f}%"
                    )
                
                # Display VaR metrics
                st.subheader(_("Value at Risk (VaR) Metrics"))
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        _("90% VaR"),
                        f"${var_90:,.2f}",
                        delta=f"{(var_90/initial_value) * 100:.2f}%",
                        delta_color="inverse"
                    )
                
                with col2:
                    st.metric(
                        _("95% VaR"),
                        f"${var_95:,.2f}",
                        delta=f"{(var_95/initial_value) * 100:.2f}%",
                        delta_color="inverse"
                    )
                
                with col3:
                    st.metric(
                        _("99% VaR"),
                        f"${var_99:,.2f}",
                        delta=f"{(var_99/initial_value) * 100:.2f}%",
                        delta_color="inverse"
                    )
                
                # Display final value distribution
                st.subheader(_("Distribution of Final Values"))
                
                fig = go.Figure()
                fig.add_trace(
                    go.Histogram(
                        x=final_values,
                        nbinsx=50,
                        marker_color='rgba(0, 0, 200, 0.7)'
                    )
                )
                
                # Add vertical lines for key values
                fig.add_vline(
                    x=initial_value,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=_("Initial Value"),
                    annotation_position="top right"
                )
                
                fig.add_vline(
                    x=mean_final,
                    line_dash="dash",
                    line_color="green",
                    annotation_text=_("Mean Final Value"),
                    annotation_position="top left"
                )
                
                fig.update_layout(
                    title=_("Distribution of Final Values after {}").format(simulation_period),
                    xaxis_title=_("Value ($)"),
                    yaxis_title=_("Frequency"),
                    height=500
                )
                
                fig.update_xaxes(tickformat="$,.0f")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Simulation interpretation
                st.subheader(_("Simulation Interpretation"))
                
                if expected_return_pct > 15:
                    risk_level = _("high potential return but likely with significant volatility")
                elif expected_return_pct > 5:
                    risk_level = _("moderate potential return with some volatility")
                elif expected_return_pct > 0:
                    risk_level = _("modest potential return with limited volatility")
                else:
                    risk_level = _("potential losses, suggesting heightened risk")
                
                st.write(_(
                    "Based on {n:,} simulations over a {p} period, the analysis shows "
                    "a {prob:.1f}% probability of profit and an expected return of {ret:.2f}%. "
                    "This suggests {risk_level}."
                ).format(
                    n=num_simulations,
                    p=simulation_period.lower(),
                    prob=prob_profit,
                    ret=expected_return_pct,
                    risk_level=risk_level
                ))
                
                # Add risk/reward advice
                if expected_return_pct > 0 and prob_profit > 60:
                    st.success(_(
                        "The risk/reward profile appears favorable, with positive expected returns "
                        "and a good probability of profit."
                    ))
                elif expected_return_pct > 0:
                    st.info(_(
                        "The risk/reward profile shows positive expected returns, "
                        "but with significant uncertainty."
                    ))
                else:
                    st.warning(_(
                        "The risk/reward profile suggests caution, with negative expected returns "
                        "and a high probability of loss."
                    ))
