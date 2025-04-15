def get_translation(language):
    """
    Get translation function for the selected language
    
    Args:
        language (str): Language code ('en' or 'es')
    
    Returns:
        function: Translation function
    """
    # English translations (base language)
    en_translations = {}
    
    # Spanish translations
    es_translations = {
        # General UI
        "Financial Analysis App": "Aplicación de Análisis Financiero",
        "Dashboard": "Panel Principal",
        "Stock Analysis": "Análisis de Acciones",
        "Portfolio Analysis": "Análisis de Cartera",
        "Valuation Tools": "Herramientas de Valoración",
        "Risk Assessment": "Evaluación de Riesgo",
        "Navigation": "Navegación",
        "Select Page": "Seleccionar Página",
        "Language": "Idioma",
        
        # Time periods
        "Select Time Period": "Seleccionar Periodo de Tiempo",
        "1 Month": "1 Mes",
        "3 Months": "3 Meses",
        "6 Months": "6 Meses",
        "1 Year": "1 Año",
        "2 Years": "2 Años",
        "5 Years": "5 Años",
        "Maximum": "Máximo",
        
        # Dashboard
        "Financial Dashboard": "Panel de Control Financiero",
        "Portfolio Composition": "Composición de la Cartera",
        "Select Stock": "Seleccionar Acción",
        "Number of Shares": "Número de Acciones",
        "Add to Portfolio": "Añadir a la Cartera",
        "Current Portfolio": "Cartera Actual",
        "Clear Portfolio": "Limpiar Cartera",
        "Portfolio Overview": "Resumen de la Cartera",
        "Portfolio Performance": "Rendimiento de la Cartera",
        "Asset Allocation": "Asignación de Activos",
        "Added {} to portfolio": "Se añadió {} a la cartera",
        "Could not fetch data for {}": "No se pudo obtener datos para {}",
        "{} is already in your portfolio": "{} ya está en tu cartera",
        "Add stocks to your portfolio to see analysis": "Añade acciones a tu cartera para ver el análisis",
        "Error in portfolio analysis: {}": "Error en el análisis de cartera: {}",
        
        # Portfolio metrics
        "Total Value": "Valor Total",
        "Annualized Return": "Rendimiento Anualizado",
        "Volatility": "Volatilidad",
        "Sharpe Ratio": "Ratio de Sharpe",
        "Max Drawdown": "Máxima Caída",
        "Beta": "Beta",
        "Alpha": "Alfa",
        
        # Stock Analysis
        "Stock Analysis Tool": "Herramienta de Análisis de Acciones",
        "Select a stock to analyze": "Selecciona una acción para analizar",
        "Key Metrics": "Métricas Clave",
        "Price Chart": "Gráfico de Precios",
        "Returns": "Rendimientos",
        "Drawdown": "Caída",
        "Stock Information": "Información de la Acción",
        "Company": "Empresa",
        "Sector": "Sector",
        "Industry": "Industria",
        "Market Cap": "Capitalización de Mercado",
        "Forward P/E": "P/E Futuro",
        "Dividend Yield": "Rendimiento de Dividendos",
        "Target Price": "Precio Objetivo",
        "Recommendation": "Recomendación",
        
        # Valuation
        "Valuation Methods": "Métodos de Valoración",
        "Discounted Cash Flow (DCF)": "Flujo de Caja Descontado (DCF)",
        "Dividend Discount Model": "Modelo de Descuento de Dividendos",
        "Relative Valuation": "Valoración Relativa",
        "Growth Rate": "Tasa de Crecimiento",
        "Discount Rate": "Tasa de Descuento",
        "Terminal Growth": "Crecimiento Terminal",
        "Forecast Years": "Años de Pronóstico",
        "Calculate": "Calcular",
        "Fair Value": "Valor Justo",
        "Current Price": "Precio Actual",
        "Upside Potential": "Potencial de Subida",
        "Multiple Type": "Tipo de Múltiplo",
        "Benchmark Multiple": "Múltiplo de Referencia",
        
        # Portfolio Analysis
        "Portfolio Analysis Tool": "Herramienta de Análisis de Cartera",
        "Portfolio Statistics": "Estadísticas de Cartera",
        "Asset Correlation": "Correlación de Activos",
        "Portfolio Optimization": "Optimización de Cartera",
        "Efficient Frontier": "Frontera Eficiente",
        "Optimize Portfolio": "Optimizar Cartera",
        "Target Return": "Rendimiento Objetivo",
        "Risk Aversion": "Aversión al Riesgo",
        "Optimized Weights": "Pesos Optimizados",
        "Current Weights": "Pesos Actuales",
        
        # Risk Assessment
        "Risk Assessment Tool": "Herramienta de Evaluación de Riesgo",
        "Risk Metrics": "Métricas de Riesgo",
        "Risk-Return Analysis": "Análisis de Riesgo-Retorno",
        "Value at Risk (VaR)": "Valor en Riesgo (VaR)",
        "Conditional VaR (CVaR)": "VaR Condicional (CVaR)",
        "Confidence Level": "Nivel de Confianza",
        "Historical VaR": "VaR Histórico",
        "Portfolio VaR": "VaR de la Cartera",
        
        # Chart labels
        "Date": "Fecha",
        "Price": "Precio",
        "Volume": "Volumen",
        "Daily Returns": "Rendimientos Diarios",
        "Cumulative Returns": "Rendimientos Acumulados",
        "Drawdown": "Caída",
        "Correlation": "Correlación",
        "Portfolio": "Cartera",
        "S&P 500": "S&P 500",
        "Price Chart for {}": "Gráfico de Precios para {}",
        "Daily Returns for {}": "Rendimientos Diarios para {}",
        "Cumulative Returns for {}": "Rendimientos Acumulados para {}",
        "Drawdown Chart for {}": "Gráfico de Caída para {}",
        "Correlation Matrix": "Matriz de Correlación",
        "Portfolio Performance vs S&P 500": "Rendimiento de la Cartera vs S&P 500",
        
        # Risk-Return quadrants
        "Lower Risk<br>Higher Return": "Menor Riesgo<br>Mayor Retorno",
        "Higher Risk<br>Higher Return": "Mayor Riesgo<br>Mayor Retorno",
        "Lower Risk<br>Lower Return": "Menor Riesgo<br>Menor Retorno",
        "Higher Risk<br>Lower Return": "Mayor Riesgo<br>Menor Retorno",
        
        # Valuation result messages
        "DCF Valuation Results": "Resultados de Valoración DCF",
        "DDM Valuation Results": "Resultados de Valoración DDM",
        "Relative Valuation Results": "Resultados de Valoración Relativa",
        "Undervalued by": "Infravalorado por",
        "Overvalued by": "Sobrevalorado por",
        "Fair value estimate": "Estimación del valor justo",
        
        # Error messages
        "Error loading page: {}": "Error al cargar la página: {}",
        "No data available for the selected period": "No hay datos disponibles para el período seleccionado",
        "Not enough data for analysis": "No hay suficientes datos para el análisis",
        "Error in calculation: {}": "Error en el cálculo: {}",
        "Financial data not available": "Datos financieros no disponibles",
        "Stock information not available": "Información de la acción no disponible"
    }
    
    # Select the translation dictionary based on language
    translations = es_translations if language == 'es' else en_translations
    
    # Create the translation function
    def translate(text):
        return translations.get(text, text)
    
    return translate
