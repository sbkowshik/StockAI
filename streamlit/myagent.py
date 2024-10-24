from langchain.agents import tool, Tool
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import talib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#general functions
def calculateRsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculateMacd(prices, short_window=12, long_window=26, signal_window=9):
    short_ema = prices.ewm(span=short_window, adjust=False).mean()
    long_ema = prices.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal_line
  
@tool
def get_company_symbol(symbol:str) -> str:
  """Returns the ticker of the company inputted"""
  ticker = yf.Ticker(symbol)
  return ticker

@tool
def get_stock_price(symbol: str) -> float:
    """
    Returns the latest stock price for the given ticker symbol.
    If today's data is not available, returns the previous day's closing price.
    Raises ValueError if no data can be retrieved.
    """
    try:
        symbol=symbol.strip()
        ticker = yf.Ticker(symbol)
        # Fetch data for the past 2 days (in case today's data is missing)
        data = ticker.history(period='5d')

        if data.empty:
            raise ValueError(f"No data available for symbol: {symbol}")

        # Check if we have at least one day of data
        latest_available_day = data['Close'].iloc[-1] 
        return round(latest_available_day, 2)

    except Exception as e:
        raise ValueError(f"Error getting price for {symbol}: {str(e)}")

@tool
def calculate_rsi(symbol:str) -> float:
    """Return RSI Calculation of ticker"""
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=100)).strftime('%Y-%m-%d')
    data = yf.download(symbol, start=start_date, end=end_date)
    delta = data['Adj Close'].diff(1)
    delta = delta[1:]
    up = delta.copy()
    down = delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    avg_gain = up.rolling(window=14).mean()
    avg_loss = abs(down.rolling(window=14).mean())
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    data['RSI'] = rsi
    temp = data['RSI'].iloc[-1]
    rsi = format(temp,".2f")
    return rsi

@tool
def moving_average(ticker:str) -> str:
    """Returns Moving Average of ticker"""
    # Define the function to calculate WMA
    def calculate_wma(prices):
        weights = np.arange(1, 15)
        wma = np.convolve(prices.flatten(), weights.flatten(), mode='valid') / weights.sum()
        return np.concatenate((np.full(13, np.nan), wma), axis=0)

    # Download stock data
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=100)).strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        return None  # Return None if no data is retrieved

    # Calculate SMA
    data['SMA'] = data['Close'].rolling(window=14).mean()

    # Calculate EMA
    data['EMA'] = data['Close'].ewm(span=14, adjust=False).mean()

    # Calculate WMA
    data['WMA'] = calculate_wma(data['Close'].values)
    mac = format(data['Close'].iloc[-1].item(), ".2f")
    mas = format(data['SMA'].iloc[-1].item(), ".2f")
    mae = format(data['EMA'].iloc[-1].item(), ".2f")
    maw = format(data['WMA'].iloc[-1].item(), ".2f")
    
    if (mac < mas):
      ta1 = f"Closing Value is less than SMA, It may indicate a downtrend. Closing Value : {mac} , SMA : {mas}. "
    else:
      ta1 = f"Closing Value is greater than SMA, It may indicate a uptrend. Closing Value : {mac} , SMA : {mas}. "
    if (mac < mas):
      ta2 = f"Closing Value is less than EMA, It may indicate a downtrend. Closing Value : {mac} , EMA : {mae}. "
    else:
      ta2 = f"Closing Value is greater than EMA, It may indicate a uptrend. Closing Value : {mac} , EMA : {mae}. "
    if (mac < mas):
      ta3 = f"Closing Value is less than WMA, It may indicate a downtrend. Closing Value : {mac} , WMA : {maw}. "
    else:
      ta3 = f"Closing Value is greater than WMA, It may indicate a uptrend. Closing Value : {mac} , WMA : {maw}. "

    x = ta1 + "\n" + ta2 + "\n" + ta3 + "\n"
    return x

@tool
def predict_stock(ticker: str) -> float:
    """Predicts the next day's closing value using ticker"""
    # Fetch historical stock data from Yahoo Finance
    stock_data = yf.download(ticker, period='max')
    
    # Calculate additional features
    stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()
    stock_data['Daily_Return'] = stock_data['Close'].pct_change()
    stock_data['Volatility'] = stock_data['Daily_Return'].rolling(window=20).std()
    
    # Add more features
    stock_data['RSI'] = calculateRsi(stock_data['Close'], window=14)
    stock_data['MACD'], stock_data['Signal_Line'] = calculateMacd(stock_data['Close'])
    
    # Drop rows with NaN values
    stock_data.dropna(inplace=True)
    
    # Define features and target variable
    X = stock_data.drop(['Adj Close', 'Close'], axis=1)  # Features
    y = stock_data['Close']  # Target
    
    # Create and train the model using all data
    regressor = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
    regressor.fit(X, y)
    
    # Evaluate the model on all data
    train_score = regressor.score(X, y)
    print(f"R-squared on all data: {train_score:.4f}")
    
    # Make prediction for the next day
    X_latest = X.iloc[-1].to_frame().T
    predicted_price = regressor.predict(X_latest)[0]
    
    return round(predicted_price, 2)

@tool
def candlestick(ticker: str) -> str:
    start_date = datetime.today() - timedelta(days=100)
    df = yf.download(ticker, start=start_date, end=datetime.today())
    df = df.reset_index()
    df = df[['Date', 'Open', 'High', 'Low', 'Close']]
    patterns = []
    for i in range(1, len(df)):
        open_price = df['Open'].iloc[i]
        close_price = df['Close'].iloc[i]
        previous_open = df['Open'].iloc[i-1]
        previous_close = df['Close'].iloc[i-1]
        
        # Simple pattern checks
        if close_price > open_price and previous_close < previous_open:  # Bullish Engulfing
            patterns.append('Bullish Engulfing')
        elif close_price < open_price and previous_close > previous_open:  # Bearish Engulfing
            patterns.append('Bearish Engulfing')
        elif abs(close_price - open_price) < 0.01 * open_price:  # Doji
            patterns.append('Doji')
        else:
            patterns.append('No Pattern')
            
    df['Pattern'] = [''] + assess_candlestick_patterns(df)
    return df['Pattern'].iloc[-1]
