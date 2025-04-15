import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime

# Configuración de la base de datos
DATABASE_URL = os.environ.get('DATABASE_URL')
engine = create_engine(DATABASE_URL)
Base = declarative_base()
Session = sessionmaker(bind=engine)

# Definición de modelos
class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True)
    password_hash = Column(String(200))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    language = Column(String(10), default='es')  # Idioma preferido (es, en, etc.)
    
    portfolios = relationship("Portfolio", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(username='{self.username}')>"


class Portfolio(Base):
    __tablename__ = 'portfolios'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    user_id = Column(Integer, ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    user = relationship("User", back_populates="portfolios")
    stocks = relationship("PortfolioStock", back_populates="portfolio", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Portfolio(name='{self.name}')>"
    
    def to_dataframe(self):
        """Convierte el portafolio a un DataFrame para análisis"""
        if not self.stocks:
            return pd.DataFrame()
        
        data = []
        for stock in self.stocks:
            data.append({
                'Symbol': stock.ticker,
                'Shares': stock.shares,
                'Price': stock.purchase_price,
                'Value': stock.shares * stock.purchase_price
            })
        
        df = pd.DataFrame(data)
        df['Weight'] = df['Value'] / df['Value'].sum()
        return df


class PortfolioStock(Base):
    __tablename__ = 'portfolio_stocks'
    
    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey('portfolios.id'))
    ticker = Column(String(20), nullable=False)
    shares = Column(Float, nullable=False)
    purchase_price = Column(Float, nullable=False)
    purchase_date = Column(DateTime, default=datetime.datetime.utcnow)
    
    portfolio = relationship("Portfolio", back_populates="stocks")
    
    def __repr__(self):
        return f"<PortfolioStock(ticker='{self.ticker}', shares={self.shares})>"


class AnalysisHistory(Base):
    __tablename__ = 'analysis_history'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    analysis_type = Column(String(50), nullable=False)  # portfolio, stock, valuation, risk
    ticker = Column(String(20), nullable=True)  # Para análisis de acciones individuales
    portfolio_id = Column(Integer, ForeignKey('portfolios.id'), nullable=True)  # Para análisis de portafolio
    parameters = Column(Text)  # Parámetros del análisis en formato JSON
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    def __repr__(self):
        return f"<AnalysisHistory(type='{self.analysis_type}')>"


# Inicialización de la base de datos
def init_db():
    Base.metadata.create_all(engine)


# Funciones de utilidad
def get_user_portfolios(user_id):
    """Obtiene todos los portafolios de un usuario"""
    session = Session()
    try:
        portfolios = session.query(Portfolio).filter_by(user_id=user_id, is_active=True).all()
        return portfolios
    finally:
        session.close()


def get_portfolio_by_id(portfolio_id):
    """Obtiene un portafolio por su ID"""
    session = Session()
    try:
        portfolio = session.query(Portfolio).filter_by(id=portfolio_id).first()
        return portfolio
    finally:
        session.close()


def save_portfolio(user_id, name, description, stocks_data):
    """
    Guarda un nuevo portafolio en la base de datos
    
    Args:
        user_id (int): ID del usuario
        name (str): Nombre del portafolio
        description (str): Descripción del portafolio
        stocks_data (list): Lista de diccionarios con información de acciones
            [{'ticker': 'AAPL', 'shares': 10, 'purchase_price': 150.0}, ...]
    
    Returns:
        Portfolio: El objeto portafolio creado
    """
    session = Session()
    try:
        portfolio = Portfolio(
            name=name,
            description=description,
            user_id=user_id
        )
        
        for stock_data in stocks_data:
            stock = PortfolioStock(
                ticker=stock_data['ticker'],
                shares=stock_data['shares'],
                purchase_price=stock_data['purchase_price']
            )
            portfolio.stocks.append(stock)
        
        session.add(portfolio)
        session.commit()
        return portfolio
    finally:
        session.close()


def save_analysis_history(user_id, analysis_type, ticker=None, portfolio_id=None, parameters=None):
    """
    Guarda un registro de análisis en el historial
    
    Args:
        user_id (int): ID del usuario (puede ser None para usuarios anónimos)
        analysis_type (str): Tipo de análisis (portfolio, stock, valuation, risk)
        ticker (str): Símbolo de la acción (para análisis de acciones individuales)
        portfolio_id (int): ID del portafolio (para análisis de portafolio)
        parameters (dict): Parámetros del análisis
    
    Returns:
        AnalysisHistory: El objeto de historial creado
    """
    session = Session()
    try:
        history = AnalysisHistory(
            user_id=user_id,
            analysis_type=analysis_type,
            ticker=ticker,
            portfolio_id=portfolio_id,
            parameters=str(parameters) if parameters else None
        )
        
        session.add(history)
        session.commit()
        return history
    finally:
        session.close()


# Inicializar la base de datos al importar el módulo
init_db()