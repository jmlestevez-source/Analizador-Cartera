import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import get_sp500_tickers
from utils.analysis import calculate_correlation_matrix
from utils.visualization import plot_correlation_heatmap, plot_risk_return_scatter
from utils.portfolio import get_portfolio_stats, optimize_portfolio, calculate_efficient_frontier
from utils.translations import get_translation

def show_page():
    """
    Display the Portfolio Analysis page
    """
    # Get translations based on selected language
    _ = get_translation(st.session_state.language)
    
    st.title(_("Portfolio Analysis Tool"))
    
    # Check if portfolio exists
    if st.session_state.portfolio.empty:
        st.warning(_("Please create a portfolio in the Dashboard first"))
        return
    
    # Portfolio stats
    st.subheader(_("Portfolio Statistics"))
    
    try:
        portfolio_stats = get_portfolio_stats(
            st.session_state.portfolio,
            period=st.session_state.period
        )
        
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
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                _("Max Drawdown"),
                f"{portfolio_stats['max_drawdown']:.2f}%"
            )
        
        with col2:
            st.metric(
                _("Beta"),
                f"{portfolio_stats['beta']:.2f}"
            )
    except Exception as e:
        st.error(_("Error in calculation: {}").format(str(e)))
    
    # Asset correlation
    st.subheader(_("Asset Correlation"))
    
    tickers = st.session_state.portfolio['Ticker'].tolist()
    
    if len(tickers) > 1:
        corr_heatmap = plot_correlation_heatmap(
            tickers,
            period=st.session_state.period,
            language=st.session_state.language
        )
        
        if corr_heatmap:
            st.plotly_chart(corr_heatmap, use_container_width=True)
        else:
            st.warning(_("Could not calculate correlation matrix"))
    else:
        st.info(_("Need at least 2 assets to calculate correlation"))
    
    # Risk-return analysis
    st.subheader(_("Risk-Return Analysis"))
    
    if len(tickers) > 0:
        risk_return_scatter = plot_risk_return_scatter(
            tickers,
            period=st.session_state.period,
            language=st.session_state.language
        )
        
        if risk_return_scatter:
            st.plotly_chart(risk_return_scatter, use_container_width=True)
        else:
            st.warning(_("Could not calculate risk-return data"))
    
    # Portfolio optimization
    st.subheader(_("Portfolio Optimization"))
    
    # Current portfolio weights
    current_weights = {}
    total_value = st.session_state.portfolio['Value'].sum()
    
    for _, row in st.session_state.portfolio.iterrows():
        weight = row['Value'] / total_value
        current_weights[row['Ticker']] = weight
    
    current_weights_df = pd.DataFrame({
        'Ticker': list(current_weights.keys()),
        'Weight': list(current_weights.values())
    })
    
    current_weights_df = current_weights_df.sort_values('Weight', ascending=False)
    
    # Display current weights
    st.write(_("Current Weights"))
    
    fig_current = px.bar(
        current_weights_df,
        x='Ticker',
        y='Weight',
        title=_("Current Portfolio Allocation"),
        text_auto='.1%'
    )
    
    fig_current.update_layout(
        yaxis=dict(
            title=_("Weight"),
            tickformat='.1%'
        )
    )
    
    st.plotly_chart(fig_current, use_container_width=True)
    
    # Optimization parameters
    st.write(_("Optimize Portfolio"))
    
    col1, col2 = st.columns(2)
    
    with col1:
        optimize_for = st.radio(
            _("Optimization Target"),
            options=[_("Maximum Sharpe Ratio"), _("Minimum Volatility"), _("Target Return")]
        )
    
    with col2:
        if optimize_for == _("Target Return"):
            target_return = st.slider(
                _("Target Return (%)"),
                min_value=0.0,
                max_value=50.0,
                value=10.0,
                step=0.5
            ) / 100
        else:
            target_return = None
        
        risk_aversion = st.slider(
            _("Risk Aversion"),
            min_value=1,
            max_value=10,
            value=5,
            step=1
        )
    
    # Optimize button
    if st.button(_("Optimize Portfolio")):
        with st.spinner(_("Optimizing portfolio...")):
            if len(tickers) > 1:
                optimization_result = optimize_portfolio(
                    tickers,
                    period=st.session_state.period,
                    target_return=target_return,
                    risk_aversion=risk_aversion
                )
                
                if "error" not in optimization_result:
                    # Display optimized weights
                    st.write(_("Optimized Weights"))
                    
                    optimized_weights_df = pd.DataFrame({
                        'Ticker': list(optimization_result['weights'].keys()),
                        'Weight': list(optimization_result['weights'].values())
                    })
                    
                    # Sort by weight descending
                    optimized_weights_df = optimized_weights_df.sort_values('Weight', ascending=False)
                    
                    fig_optimized = px.bar(
                        optimized_weights_df,
                        x='Ticker',
                        y='Weight',
                        title=_("Optimized Portfolio Allocation"),
                        text_auto='.1%'
                    )
                    
                    fig_optimized.update_layout(
                        yaxis=dict(
                            title=_("Weight"),
                            tickformat='.1%'
                        )
                    )
                    
                    st.plotly_chart(fig_optimized, use_container_width=True)
                    
                    # Display optimization results
                    st.write(_("Optimization Results"))
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            _("Expected Return"),
                            f"{optimization_result['portfolio_return']:.2f}%"
                        )
                    
                    with col2:
                        st.metric(
                            _("Expected Volatility"),
                            f"{optimization_result['portfolio_volatility']:.2f}%"
                        )
                    
                    with col3:
                        st.metric(
                            _("Sharpe Ratio"),
                            f"{optimization_result['sharpe_ratio']:.2f}"
                        )
                else:
                    st.error(_("Error in optimization: {}").format(optimization_result['error']))
            else:
                st.warning(_("Need at least 2 assets for optimization"))
    
    # Efficient frontier
    st.subheader(_("Efficient Frontier"))
    
    if st.button(_("Calculate Efficient Frontier")):
        with st.spinner(_("Calculating efficient frontier...")):
            if len(tickers) > 1:
                frontier_data = calculate_efficient_frontier(
                    tickers,
                    period=st.session_state.period,
                    num_portfolios=20
                )
                
                if "error" not in frontier_data:
                    # Plot efficient frontier
                    fig = go.Figure()
                    
                    # Add efficient frontier
                    fig.add_trace(
                        go.Scatter(
                            x=frontier_data['volatilities'],
                            y=frontier_data['returns'],
                            mode='lines+markers',
                            name=_("Efficient Frontier"),
                            marker=dict(
                                size=6,
                                color='rgba(0, 0, 255, 0.8)'
                            )
                        )
                    )
                    
                    # Add maximum Sharpe ratio portfolio
                    if frontier_data.get('max_sharpe_idx') is not None:
                        max_sharpe_idx = frontier_data['max_sharpe_idx']
                        fig.add_trace(
                            go.Scatter(
                                x=[frontier_data['volatilities'][max_sharpe_idx]],
                                y=[frontier_data['returns'][max_sharpe_idx]],
                                mode='markers',
                                name=_("Maximum Sharpe Ratio"),
                                marker=dict(
                                    size=12,
                                    color='rgba(0, 255, 0, 0.8)'
                                )
                            )
                        )
                    
                    # Add minimum variance portfolio
                    if frontier_data.get('min_var_idx') is not None:
                        min_var_idx = frontier_data['min_var_idx']
                        fig.add_trace(
                            go.Scatter(
                                x=[frontier_data['volatilities'][min_var_idx]],
                                y=[frontier_data['returns'][min_var_idx]],
                                mode='markers',
                                name=_("Minimum Variance"),
                                marker=dict(
                                    size=12,
                                    color='rgba(255, 0, 0, 0.8)'
                                )
                            )
                        )
                    
                    # Add current portfolio
                    current_stats = get_portfolio_stats(
                        st.session_state.portfolio,
                        period=st.session_state.period
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=[current_stats['volatility']],
                            y=[current_stats['annualized_return']],
                            mode='markers',
                            name=_("Current Portfolio"),
                            marker=dict(
                                size=12,
                                color='rgba(255, 255, 0, 0.8)'
                            )
                        )
                    )
                    
                    # Configure chart layout
                    fig.update_layout(
                        title=_("Efficient Frontier"),
                        xaxis=dict(
                            title=_("Expected Volatility (%)"),
                            tickformat='.2f'
                        ),
                        yaxis=dict(
                            title=_("Expected Return (%)"),
                            tickformat='.2f'
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
                    
                    # Display portfolio compositions
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(_("Maximum Sharpe Ratio Portfolio"))
                        
                        if frontier_data.get('max_sharpe_idx') is not None:
                            max_sharpe_weights = frontier_data['weights'][max_sharpe_idx]
                            max_sharpe_df = pd.DataFrame({
                                'Ticker': list(max_sharpe_weights.keys()),
                                'Weight': list(max_sharpe_weights.values())
                            })
                            max_sharpe_df = max_sharpe_df.sort_values('Weight', ascending=False)
                            
                            st.dataframe(max_sharpe_df, use_container_width=True)
                    
                    with col2:
                        st.write(_("Minimum Variance Portfolio"))
                        
                        if frontier_data.get('min_var_idx') is not None:
                            min_var_weights = frontier_data['weights'][min_var_idx]
                            min_var_df = pd.DataFrame({
                                'Ticker': list(min_var_weights.keys()),
                                'Weight': list(min_var_weights.values())
                            })
                            min_var_df = min_var_df.sort_values('Weight', ascending=False)
                            
                            st.dataframe(min_var_df, use_container_width=True)
                else:
                    st.error(_("Error calculating efficient frontier: {}").format(frontier_data['error']))
            else:
                st.warning(_("Need at least 2 assets to calculate efficient frontier"))
