import pandas as pd
import numpy as np
import streamlit as st
from utils.data_loader import get_stock_data, get_stock_info, get_financial_data, get_financial_data_from_stockanalysis
from utils.analysis import calculate_returns, calculate_beta

def get_shares_outstanding(stock_info):
    """
    Obtiene el número de acciones en circulación de manera confiable
    Si no está disponible directamente, intenta calcularlo con marketCap/currentPrice

    Args:
        stock_info (dict): Información del ticker

    Returns:
        float: Número de acciones en circulación (en millones)
    """
    shares_outstanding = float(stock_info.get('sharesOutstanding', 0))

    # Si no está disponible, intentamos calcular
    if shares_outstanding == 0:
        market_cap = float(stock_info.get('marketCap', 0))
        current_price = float(stock_info.get('currentPrice', 1))
        shares_outstanding = market_cap / current_price

    # Convertir a millones
    return shares_outstanding / 1_000_000

def calculate_dcf_value(ticker, growth_rate=0.05, discount_rate=0.1, terminal_growth=0.02, forecast_years=5):
    """
    Calculate Discounted Cash Flow (DCF) valuation for a stock

    Args:
        ticker (str): Stock ticker symbol
        growth_rate (float): Annual growth rate for cash flows
        discount_rate (float): Discount rate (WACC)
        terminal_growth (float): Terminal growth rate
        forecast_years (int): Number of years to forecast

    Returns:
        dict: DCF analysis results including fair value per share
    """
    try:
        # Get financial data
        income_stmt, balance_sheet, cash_flow = get_financial_data(ticker)

        if income_stmt.empty or cash_flow.empty or balance_sheet.empty:
            return {"error": "Financial data not available"}

        # Get latest free cash flow
        try:
            if 'Free Cash Flow' in cash_flow.index:
                latest_fcf = float(cash_flow.loc['Free Cash Flow'].iloc[0])
            else:
                operating_cash_flow = float(cash_flow.loc['Operating Cash Flow'].iloc[0])
                capital_expenditures = float(cash_flow.loc['Capital Expenditure'].iloc[0])
                latest_fcf = operating_cash_flow + capital_expenditures

            # Get cash and cash equivalents
            cash = float(balance_sheet.loc['Cash And Cash Equivalents'].iloc[0])
            short_term_investments = float(balance_sheet.loc['Short Term Investments'].iloc[0]) if 'Short Term Investments' in balance_sheet.index else 0
            cash_and_equivalents = cash + short_term_investments

        except (KeyError, IndexError, ValueError) as e:
            return {"error": f"Error processing financial data: {str(e)}"}

        # Get current stock data for shares outstanding and current price
        stock_info = get_stock_info(ticker)

        if not stock_info:
            return {"error": "Stock information not available"}

        # Get shares outstanding (in millions)
        shares_outstanding = get_shares_outstanding(stock_info) / 1_000_000

        # Calculate projected cash flows
        projected_fcf = []
        for year in range(1, forecast_years + 1):
            fcf = latest_fcf * (1 + growth_rate) ** year
            projected_fcf.append(fcf)

        # Calculate terminal value
        terminal_value = projected_fcf[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)

        # Discount cash flows
        pv_fcf = []
        for year, fcf in enumerate(projected_fcf, start=1):
            pv = fcf / (1 + discount_rate) ** year
            pv_fcf.append(pv)

        # Discount terminal value
        pv_terminal = terminal_value / (1 + discount_rate) ** forecast_years

        # Calculate enterprise value
        enterprise_value = sum(pv_fcf) + pv_terminal

        # Get debt and cash from balance sheet
        try:
            total_debt = balance_sheet.loc['Total Debt'].iloc[0]
        except (KeyError, IndexError):
            total_debt = 0

        try:
            cash_and_equivalents = float(balance_sheet.loc['Cash And Cash Equivalents'].iloc[0])
            if 'Short Term Investments' in balance_sheet.index:
                cash_and_equivalents += float(balance_sheet.loc['Short Term Investments'].iloc[0])
        except (KeyError, IndexError, ValueError):
            cash_and_equivalents = 0

        # Calculate equity value
        equity_value = enterprise_value - total_debt + cash_and_equivalents

        # Calculate fair value per share
        fair_value_per_share = equity_value / shares_outstanding

        # Get current price
        stock_data = get_stock_data(ticker, period="1d")
        current_price = stock_data['Close'].iloc[-1] if not stock_data.empty else 0

        # Calculate potential upside/downside
        if current_price > 0:
            upside_percentage = (fair_value_per_share / current_price - 1) * 100
        else:
            upside_percentage = 0

        # Compile results
        dcf_results = {
            "latest_fcf": latest_fcf,
            "projected_fcf": projected_fcf,
            "terminal_value": terminal_value,
            "pv_fcf": pv_fcf,
            "pv_terminal": pv_terminal,
            "enterprise_value": enterprise_value,
            "total_debt": total_debt,
            "cash_and_equivalents": cash_and_equivalents,
            "equity_value": equity_value,
            "shares_outstanding": shares_outstanding,
            "fair_value_per_share": fair_value_per_share,
            "current_price": current_price,
            "upside_percentage": upside_percentage
        }

        return dcf_results
    except Exception as e:
        return {"error": str(e)}

def calculate_dividend_discount(ticker, current_dividend=None, growth_rate=0.05, discount_rate=0.1, terminal_growth=0.02):
    """
    Calculate Dividend Discount Model (DDM) valuation for a stock

    Args:
        ticker (str): Stock ticker symbol
        current_dividend (float): Current annual dividend per share (if None, will attempt to fetch)
        growth_rate (float): Annual dividend growth rate
        discount_rate (float): Discount rate (cost of equity)
        terminal_growth (float): Terminal growth rate

    Returns:
        dict: DDM analysis results including fair value per share
    """
    try:
        # Get stock info
        stock_info = get_stock_info(ticker)

        if not stock_info:
            return {"error": "Stock information not available"}

        # Get current dividend yield or use provided dividend
        if current_dividend is None:
            dividend_yield = stock_info.get('dividendYield', 0)
            stock_data = get_stock_data(ticker, period="1d")
            current_price = stock_data['Close'].iloc[-1] if not stock_data.empty else 0

            if dividend_yield > 0 and current_price > 0:
                current_dividend = (dividend_yield / 100) * current_price
            else:
                return {"error": "Dividend data not available"}

        # Calculate fair value using Gordon Growth Model (assuming constant growth)
        if discount_rate <= growth_rate:
            return {"error": "Discount rate must be greater than growth rate"}

        fair_value = current_dividend * (1 + growth_rate) / (discount_rate - growth_rate)

        # Get current price
        stock_data = get_stock_data(ticker, period="1d")
        current_price = stock_data['Close'].iloc[-1] if not stock_data.empty else 0

        # Calculate potential upside/downside
        if current_price > 0:
            upside_percentage = (fair_value / current_price - 1) * 100
        else:
            upside_percentage = 0

        # Compile results
        ddm_results = {
            "current_dividend": current_dividend,
            "growth_rate": growth_rate,
            "discount_rate": discount_rate,
            "fair_value": fair_value,
            "current_price": current_price,
            "upside_percentage": upside_percentage
        }

        return ddm_results
    except Exception as e:
        return {"error": str(e)}

def calculate_multiples_valuation(ticker, multiple_type="PE", benchmark_multiple=None, peer_tickers=None):
    """
    Calculate relative valuation based on multiples

    Args:
        ticker (str): Stock ticker symbol
        multiple_type (str): Type of multiple to use (PE, PS, PB, EV/EBITDA)
        benchmark_multiple (float): Benchmark multiple to use (if None, will use peer average)
        peer_tickers (list): List of peer tickers for comparison (if None, will use sector average)

    Returns:
        dict: Relative valuation results
    """
    try:
        # Get stock info
        stock_info = get_stock_info(ticker)

        if not stock_info:
            return {"error": "Stock information not available"}

        # Get financial data
        income_stmt, balance_sheet, _ = get_financial_data(ticker)

        if income_stmt.empty or balance_sheet.empty:
            return {"error": "Financial data not available"}

        # Get current price
        stock_data = get_stock_data(ticker, period="1d")
        current_price = stock_data['Close'].iloc[-1] if not stock_data.empty else 0

        if current_price == 0:
            return {"error": "Current price data not available"}

        # Calculate the stock's multiple
        stock_multiple = None

        if multiple_type == "PE":
            try:
                earnings_per_share = income_stmt.loc['Diluted EPS'].iloc[0]
                stock_multiple = current_price / earnings_per_share if earnings_per_share > 0 else None
            except (KeyError, IndexError):
                stock_multiple = stock_info.get('forwardPE', None)

        elif multiple_type == "PS":
            try:
                revenue = income_stmt.loc['Total Revenue'].iloc[0]
                shares_outstanding = stock_info.get('sharesOutstanding', 0)
                revenue_per_share = revenue / shares_outstanding
                stock_multiple = current_price / revenue_per_share if revenue_per_share > 0 else None
            except (KeyError, IndexError):
                stock_multiple = stock_info.get('priceToSalesTrailing12Months', None)

        elif multiple_type == "PB":
            try:
                book_value = balance_sheet.loc['Total Stockholder Equity'].iloc[0]
                shares_outstanding = stock_info.get('sharesOutstanding', 0)
                book_value_per_share = book_value / shares_outstanding
                stock_multiple = current_price / book_value_per_share if book_value_per_share > 0 else None
            except (KeyError, IndexError):
                stock_multiple = stock_info.get('priceToBook', None)

        elif multiple_type == "EV/EBITDA":
            try:
                ebitda = income_stmt.loc['EBITDA'].iloc[0]
                market_cap = stock_info.get('marketCap', 0)
                total_debt = balance_sheet.loc['Total Debt'].iloc[0]
                cash_and_equivalents = balance_sheet.loc['Cash and Cash Equivalents'].iloc[0]
                enterprise_value = market_cap + total_debt - cash_and_equivalents
                stock_multiple = enterprise_value / ebitda if ebitda > 0 else None
            except (KeyError, IndexError):
                stock_multiple = stock_info.get('enterpriseValueToEbitda', None)

        # If stock multiple could not be calculated
        if stock_multiple is None:
            return {"error": f"Could not calculate {multiple_type} multiple for {ticker}"}

        # Get benchmark multiple
        if benchmark_multiple is None:
            # If peer tickers are provided, calculate average multiple of peers
            if peer_tickers and len(peer_tickers) > 0:
                peer_multiples = []

                for peer in peer_tickers:
                    peer_info = get_stock_info(peer)

                    if multiple_type == "PE":
                        peer_multiple = peer_info.get('forwardPE', None)
                    elif multiple_type == "PS":
                        peer_multiple = peer_info.get('priceToSalesTrailing12Months', None)
                    elif multiple_type == "PB":
                        peer_multiple = peer_info.get('priceToBook', None)
                    elif multiple_type == "EV/EBITDA":
                        peer_multiple = peer_info.get('enterpriseValueToEbitda', None)

                    if peer_multiple is not None:
                        peer_multiples.append(peer_multiple)

                if peer_multiples:
                    benchmark_multiple = sum(peer_multiples) / len(peer_multiples)
                else:
                    return {"error": "Could not calculate peer average multiple"}
            else:
                # Use sector average or a default multiple
                if multiple_type == "PE":
                    benchmark_multiple = 20.0  # Default P/E ratio
                elif multiple_type == "PS":
                    benchmark_multiple = 2.0  # Default P/S ratio
                elif multiple_type == "PB":
                    benchmark_multiple = 3.0  # Default P/B ratio
                elif multiple_type == "EV/EBITDA":
                    benchmark_multiple = 10.0  # Default EV/EBITDA

        # Calculate fair value based on benchmark multiple
        if multiple_type == "PE":
            try:
                earnings_per_share = income_stmt.loc['Diluted EPS'].iloc[0]
                fair_value = benchmark_multiple * earnings_per_share
            except (KeyError, IndexError):
                earnings_per_share = current_price / stock_multiple if stock_multiple > 0 else 0
                fair_value = benchmark_multiple * earnings_per_share

        elif multiple_type == "PS":
            try:
                revenue = income_stmt.loc['Total Revenue'].iloc[0]
                shares_outstanding = stock_info.get('sharesOutstanding', 0)
                revenue_per_share = revenue / shares_outstanding
                fair_value = benchmark_multiple * revenue_per_share
            except (KeyError, IndexError):
                revenue_per_share = current_price / stock_multiple if stock_multiple > 0 else 0
                fair_value = benchmark_multiple * revenue_per_share

        elif multiple_type == "PB":
            try:
                book_value = balance_sheet.loc['Total Stockholder Equity'].iloc[0]
                shares_outstanding = stock_info.get('sharesOutstanding', 0)
                book_value_per_share = book_value / shares_outstanding
                fair_value = benchmark_multiple * book_value_per_share
            except (KeyError, IndexError):
                book_value_per_share = current_price / stock_multiple if stock_multiple > 0 else 0
                fair_value = benchmark_multiple * book_value_per_share

        elif multiple_type == "EV/EBITDA":
            try:
                ebitda = income_stmt.loc['EBITDA'].iloc[0]
                shares_outstanding = stock_info.get('sharesOutstanding', 0)
                total_debt = balance_sheet.loc['Total Debt'].iloc[0]
                cash_and_equivalents = balance_sheet.loc['Cash and Cash Equivalents'].iloc[0]

                # Calculate enterprise value using benchmark multiple
                enterprise_value = benchmark_multiple * ebitda

                # Calculate implied equity value
                equity_value = enterprise_value - total_debt + cash_and_equivalents

                # Calculate fair value per share
                fair_value = equity_value / shares_outstanding
            except (KeyError, IndexError):
                fair_value = current_price  # Fallback to current price

        # Calculate potential upside/downside
        upside_percentage = (fair_value / current_price - 1) * 100

        # Compile results
        valuation_results = {
            "stock_multiple": stock_multiple,
            "benchmark_multiple": benchmark_multiple,
            "fair_value": fair_value,
            "current_price": current_price,
            "upside_percentage": upside_percentage
        }

        return valuation_results
    except Exception as e:
        return {"error": str(e)}

def calculate_capm(ticker, period="5y"):
    """
    Calculate Cost of Equity using Capital Asset Pricing Model (CAPM)

    Args:
        ticker (str): Stock ticker symbol
        period (str): Time period for analysis

    Returns:
        dict: CAPM analysis results
    """
    try:
        # Get stock data
        stock_data = get_stock_data(ticker, period=period)

        # Get market data (S&P 500)
        market_data = get_stock_data("^GSPC", period=period)

        if stock_data.empty or market_data.empty:
            return {"error": "Price data not available"}

        # Calculate returns
        stock_returns = calculate_returns(stock_data['Close'])
        market_returns = calculate_returns(market_data['Close'])

        # Align the data
        df = pd.DataFrame({'stock': stock_returns, 'market': market_returns})
        df = df.dropna()

        if len(df) < 30:  # Require at least 30 data points
            return {"error": "Not enough data points for reliable analysis"}

        # Calculate beta
        beta = calculate_beta(df['stock'], df['market'])

        # Get risk-free rate (10Y Treasury Yield)
        risk_free_data = get_stock_data("^TNX", period="1mo")

        if not risk_free_data.empty:
            risk_free_rate = risk_free_data['Close'].iloc[-1] / 100  # Convert from percentage
        else:
            risk_free_rate = 0.04  # Default to 4% if data fetch fails

        # Calculate market risk premium (historical average is around 5-6%)
        market_risk_premium = 0.05

        # Calculate cost of equity using CAPM formula: Rf + Beta * (Rm - Rf)
        cost_of_equity = risk_free_rate + beta * market_risk_premium

        # Compile results
        capm_results = {
            "risk_free_rate": risk_free_rate,
            "beta": beta,
            "market_risk_premium": market_risk_premium,
            "cost_of_equity": cost_of_equity
        }

        return capm_results
    except Exception as e:
        return {"error": str(e)}