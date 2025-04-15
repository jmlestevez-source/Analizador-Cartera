import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

from utils.data_loader import get_sp500_tickers, get_stock_data
from utils.analysis import calculate_returns
from utils.portfolio import get_portfolio_stats
from utils.visualization import plot_asset_allocation
from utils.translations import get_translation
from utils.database import (
    User, Portfolio, PortfolioStock, Session, 
    get_user_portfolios, get_portfolio_by_id, save_portfolio
)

def show_page():
    """
    Muestra la página de gestión de carteras
    """
    # Obtener función de traducción
    _ = get_translation(st.session_state.language)
    
    st.title(_("Gestor de Carteras"))
    
    # Inicializar usuario si no existe en la sesión
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    
    # Tab para login/registro y gestión de usuario
    user_tab, portfolio_tab = st.tabs([_("Usuario"), _("Carteras")])
    
    with user_tab:
        handle_user_section(_)
    
    with portfolio_tab:
        handle_portfolio_section(_)


def handle_user_section(_):
    """
    Maneja la sección de usuario (login/registro)
    """
    if st.session_state.user_id is None:
        # Login/registro
        login_tab, register_tab = st.tabs([_("Iniciar Sesión"), _("Registrarse")])
        
        with login_tab:
            st.subheader(_("Iniciar Sesión"))
            username = st.text_input(_("Nombre de Usuario"))
            password = st.text_input(_("Contraseña"), type="password")
            
            if st.button(_("Iniciar Sesión"), key="login_button"):
                # Comprobar credenciales
                session = Session()
                user = session.query(User).filter_by(username=username).first()
                session.close()
                
                if user and user.password_hash == password:  # En producción usar hash seguro
                    st.session_state.user_id = user.id
                    st.session_state.username = user.username
                    st.success(_("¡Inicio de sesión exitoso!"))
                    st.rerun()
                else:
                    st.error(_("Nombre de usuario o contraseña incorrectos"))
        
        with register_tab:
            st.subheader(_("Registrarse"))
            new_username = st.text_input(_("Nombre de Usuario"), key="reg_username")
            new_email = st.text_input(_("Correo Electrónico"))
            new_password = st.text_input(_("Contraseña"), type="password", key="reg_password")
            confirm_password = st.text_input(_("Confirmar Contraseña"), type="password")
            
            if st.button(_("Registrarse"), key="register_button"):
                if new_password != confirm_password:
                    st.error(_("Las contraseñas no coinciden"))
                else:
                    # Comprobar si el usuario ya existe
                    session = Session()
                    existing_user = session.query(User).filter_by(username=new_username).first()
                    
                    if existing_user:
                        st.error(_("El nombre de usuario ya está en uso"))
                        session.close()
                    else:
                        # Crear nuevo usuario
                        user = User(
                            username=new_username,
                            email=new_email,
                            password_hash=new_password,  # En producción usar hash seguro
                            language=st.session_state.language
                        )
                        session.add(user)
                        session.commit()
                        
                        # Guardar ID en la sesión
                        st.session_state.user_id = user.id
                        st.session_state.username = user.username
                        
                        session.close()
                        st.success(_("¡Registro exitoso!"))
                        st.rerun()
    else:
        # Mostrar perfil de usuario
        st.subheader(_("Perfil de Usuario"))
        st.write(_("Usuario: {}").format(st.session_state.username))
        
        if st.button(_("Cerrar Sesión"), key="logout_button"):
            st.session_state.user_id = None
            st.session_state.username = None
            st.rerun()


def handle_portfolio_section(_):
    """
    Maneja la sección de gestión de carteras
    """
    if st.session_state.user_id is None:
        st.info(_("Inicia sesión o regístrate para gestionar tus carteras"))
        return
    
    # Obtener carteras del usuario
    portfolios = get_user_portfolios(st.session_state.user_id)
    
    # Crear nueva cartera o seleccionar existente
    create_tab, select_tab = st.tabs([_("Crear Cartera"), _("Seleccionar Cartera")])
    
    with create_tab:
        st.subheader(_("Crear Nueva Cartera"))
        portfolio_name = st.text_input(_("Nombre de la Cartera"))
        portfolio_description = st.text_area(_("Descripción (opcional)"))
        
        # Sección para añadir acciones
        st.subheader(_("Añadir Acciones"))
        
        # Inicializar lista de acciones si no existe
        if 'portfolio_stocks' not in st.session_state:
            st.session_state.portfolio_stocks = []
        
        # Seleccionar acción
        ticker_input_method = st.radio(
            _("Método de ingreso"),
            options=[_("Seleccionar de lista"), _("Ingresar manualmente")],
            horizontal=True
        )
        
        if ticker_input_method == _("Seleccionar de lista"):
            available_tickers = get_sp500_tickers()
            selected_ticker = st.selectbox(
                _("Seleccionar Acción"),
                options=available_tickers,
                index=0 if len(available_tickers) > 0 else None
            )
        else:
            selected_ticker = st.text_input(_("Ingresar símbolo de acción (ej: AAPL, MSFT, GOOG)")).upper()
        
        col1, col2 = st.columns(2)
        
        with col1:
            shares = st.number_input(_("Número de Acciones"), min_value=0.01, value=1.0, step=0.01)
        
        with col2:
            # Obtener precio actual como referencia
            try:
                stock_data = get_stock_data(selected_ticker, period="1d")
                current_price = stock_data['Close'].iloc[-1]
                st.write(_("Precio actual: ${:.2f}").format(current_price))
                price = st.number_input(_("Precio de Compra"), min_value=0.01, value=current_price, step=0.01)
            except:
                price = st.number_input(_("Precio de Compra"), min_value=0.01, value=100.0, step=0.01)
        
        if st.button(_("Añadir a la Cartera"), key="add_stock_button"):
            # Añadir acción a la lista temporal
            st.session_state.portfolio_stocks.append({
                'ticker': selected_ticker,
                'shares': shares,
                'purchase_price': price,
                'value': shares * price
            })
            st.success(_("Acción añadida a la cartera"))
        
        # Mostrar acciones en la cartera
        if st.session_state.portfolio_stocks:
            st.subheader(_("Acciones en la Cartera"))
            df = pd.DataFrame(st.session_state.portfolio_stocks)
            df = df.rename(columns={
                'ticker': _('Símbolo'),
                'shares': _('Acciones'),
                'purchase_price': _('Precio de Compra'),
                'value': _('Valor')
            })
            df[_('Valor')] = df[_('Acciones')] * df[_('Precio de Compra')]
            total_value = df[_('Valor')].sum()
            df[_('Peso (%)')] = (df[_('Valor')] / total_value * 100).round(2)
            
            st.dataframe(df, use_container_width=True)
            
            # Mostrar valoración total
            st.metric(_("Valor Total de la Cartera"), f"${total_value:,.2f}")
            
            # Visualización de la asignación de activos
            fig = px.pie(
                df, 
                values=_('Valor'), 
                names=_('Símbolo'),
                title=_("Asignación de Activos")
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Botón para guardar cartera
            if st.button(_("Guardar Cartera"), key="save_portfolio_button"):
                if not portfolio_name:
                    st.error(_("Por favor, introduce un nombre para la cartera"))
                else:
                    # Guardar cartera en la base de datos
                    stocks_data = []
                    for stock in st.session_state.portfolio_stocks:
                        stocks_data.append({
                            'ticker': stock['ticker'],
                            'shares': stock['shares'],
                            'purchase_price': stock['purchase_price']
                        })
                    
                    save_portfolio(
                        user_id=st.session_state.user_id,
                        name=portfolio_name,
                        description=portfolio_description,
                        stocks_data=stocks_data
                    )
                    
                    # Limpiar estado
                    st.session_state.portfolio_stocks = []
                    st.success(_("¡Cartera guardada exitosamente!"))
                    
                    # Actualizar portafolio en sesión para análisis
                    portfolio_df = pd.DataFrame(stocks_data)
                    portfolio_df = portfolio_df.rename(columns={
                        'ticker': 'Symbol',
                        'shares': 'Shares',
                        'purchase_price': 'Price'
                    })
                    portfolio_df['Value'] = portfolio_df['Shares'] * portfolio_df['Price']
                    portfolio_df['Weight'] = portfolio_df['Value'] / portfolio_df['Value'].sum()
                    
                    st.session_state.portfolio = portfolio_df
        
        # Botón para limpiar cartera
        if st.session_state.portfolio_stocks and st.button(_("Limpiar Cartera"), key="clear_portfolio_button"):
            st.session_state.portfolio_stocks = []
            st.success(_("Cartera limpiada"))
            st.rerun()
    
    with select_tab:
        st.subheader(_("Seleccionar Cartera Existente"))
        
        if not portfolios:
            st.info(_("No tienes carteras guardadas"))
        else:
            # Crear opciones de portafolio para selectbox
            portfolio_options = {f"{p.name} (ID: {p.id})": p.id for p in portfolios}
            selected_portfolio_name = st.selectbox(
                _("Seleccionar Cartera"),
                options=list(portfolio_options.keys()),
                index=0
            )
            
            selected_portfolio_id = portfolio_options[selected_portfolio_name]
            portfolio = get_portfolio_by_id(selected_portfolio_id)
            
            if portfolio:
                st.write(_("Creada: {}").format(portfolio.created_at.strftime('%Y-%m-%d')))
                st.write(_("Descripción: {}").format(portfolio.description or _("Sin descripción")))
                
                # Obtener DataFrame del portafolio
                portfolio_df = portfolio.to_dataframe()
                
                if not portfolio_df.empty:
                    # Mostrar tabla de acciones
                    st.subheader(_("Composición de la Cartera"))
                    
                    # Obtener precios actuales
                    portfolio_df['Current Price'] = float('nan')
                    portfolio_df['Current Value'] = float('nan')
                    portfolio_df['Gain/Loss %'] = float('nan')
                    
                    for idx, row in portfolio_df.iterrows():
                        try:
                            current_data = get_stock_data(row['Symbol'], period="1d")
                            current_price = current_data['Close'].iloc[-1]
                            portfolio_df.at[idx, 'Current Price'] = current_price
                            portfolio_df.at[idx, 'Current Value'] = row['Shares'] * current_price
                            portfolio_df.at[idx, 'Gain/Loss %'] = (current_price / row['Price'] - 1) * 100
                        except:
                            pass
                    
                    # Traducir nombres de columnas
                    display_df = portfolio_df.copy()
                    display_df = display_df.rename(columns={
                        'Symbol': _('Símbolo'),
                        'Shares': _('Acciones'),
                        'Price': _('Precio de Compra'),
                        'Value': _('Valor Inicial'),
                        'Weight': _('Peso (%)'),
                        'Current Price': _('Precio Actual'),
                        'Current Value': _('Valor Actual'),
                        'Gain/Loss %': _('Ganancia/Pérdida %')
                    })
                    
                    # Formatear porcentajes y valores monetarios
                    display_df[_('Peso (%)')] = display_df[_('Peso (%)')] * 100
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Mostrar valoración total
                    initial_value = portfolio_df['Value'].sum()
                    current_value = portfolio_df['Current Value'].sum()
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            _("Valor Inicial"),
                            f"${initial_value:,.2f}"
                        )
                    
                    with col2:
                        st.metric(
                            _("Valor Actual"),
                            f"${current_value:,.2f}"
                        )
                    
                    with col3:
                        gain_loss = (current_value / initial_value - 1) * 100
                        st.metric(
                            _("Ganancia/Pérdida"),
                            f"{gain_loss:.2f}%"
                        )
                    
                    # Visualizar asignación de activos
                    st.subheader(_("Asignación de Activos"))
                    fig = px.pie(
                        portfolio_df,
                        values='Value',
                        names='Symbol',
                        title=_("Asignación por Valor Inicial")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Botón para usar esta cartera para análisis
                    if st.button(_("Usar esta Cartera para Análisis"), key="use_portfolio_button"):
                        st.session_state.portfolio = portfolio_df
                        st.success(_("¡Cartera seleccionada para análisis!"))
                        
                        # Redirigir a la página de análisis de cartera
                        st.info(_("Ahora puedes ir a la página de Análisis de Cartera para realizar un análisis detallado"))
                    
                    # Botón para eliminar cartera (marcar como inactiva)
                    if st.button(_("Eliminar Cartera"), key="delete_portfolio_button"):
                        session = Session()
                        try:
                            portfolio = session.query(Portfolio).filter_by(id=selected_portfolio_id).first()
                            if portfolio:
                                portfolio.is_active = False
                                session.commit()
                                st.success(_("Cartera eliminada exitosamente"))
                                st.rerun()
                        finally:
                            session.close()