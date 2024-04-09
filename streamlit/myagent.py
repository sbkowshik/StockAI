from langchain.tools import BaseTool
from typing import Optional, Type
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.agents import load_tools, initialize_agent, AgentExecutor
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain, LLMChain, create_extraction_chain
from langchain.agents import load_tools, ZeroShotAgent
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from googleapiclient.discovery import build
import matplotlib.pyplot as plt
import requests
import os
import yfinance as yf
import json
import pandas as pd
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from pydantic import BaseModel, Field
import re
from langchain import hub
from langchain_community.llms import OpenAI
from langchain.agents import AgentExecutor, create_react_agent
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from langchain import hub
from requests_html import HTMLSession
from urllib.parse import urlencode
from datetime import datetime
from datetime import date
from datetime import timedelta

from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain import hub

from dotenv import load_dotenv

load_dotenv()

def get_stock_price(symbol):
    ticker = yf.Ticker(symbol)
    todays_data = ticker.history(period='1d')
    return round(todays_data['Close'][0], 2)

def calculate_trend_analysis(symbol):
  today = date.today() - timedelta(days=1)
  now = datetime.now()
  today_date = now.strftime('%Y-%m-%d')
  current_year = datetime.now().year
  current_year_start_date = str(current_year) + "-01-01"
  last_year = current_year - 1
  year_before_last = last_year - 1
  last_year_start_date = str(last_year) + "-01-01"
  last_year_end_date = str(last_year) + "-12-31"
  year_before_last_start_date = str(year_before_last) + "-01-01"
  year_before_last_ending_date = str(year_before_last) + "-12-31"
  historical_stock_data = yf.download(symbol, year_before_last_start_date, today_date)
  historical_stock_data_df = pd.DataFrame(historical_stock_data)
  historical_stock_data_df['lastDayOpen'] = historical_stock_data['Open'].shift(1)
  historical_stock_data_df['lastDayHigh'] = historical_stock_data['High'].shift(1)
  historical_stock_data_df['lastDayLow'] = historical_stock_data['Low'].shift(1)
  historical_stock_data_df['lastDayClose'] = historical_stock_data['Close'].shift(1)
  historical_stock_data_df['lastDayVolume'] = historical_stock_data['Volume'].shift(1)
  historical_stock_data_df.reset_index(inplace = True)
  today_date = now.strftime('%Y-%m-%d')
  yesterday = today - timedelta(days=4)
  yesterday_date = yesterday.strftime('%Y-%m-%d')
  yesterday_data = historical_stock_data_df[historical_stock_data_df['Date'] == yesterday_date]

  if not yesterday_data.empty:
    yesterday_lastDayHigh = yesterday_data['lastDayHigh'].values[0]
    yesterday_lastDayClose = yesterday_data['lastDayClose'].values[0]
    yesterday_lastDayLow = yesterday_data['lastDayLow'].values[0]
  df=historical_stock_data_df
  df['Price Change'] = df['Close'] - df['lastDayClose']
  df['Gain'] = df['Price Change'].apply(lambda x: x if x > 0 else 0)
  df['Loss'] = df['Price Change'].apply(lambda x: -x if x < 0 else 0)
  period = 14
  df['Avg Gain'] = df['Gain'].rolling(window=period).mean()
  df['Avg Loss'] = df['Loss'].rolling(window=period).mean()
  df['RS'] = df['Avg Gain'] / df['Avg Loss']
  df['RSI'] = 100 - (100 / (1 + df['RS']))
  df.dropna(inplace=True)
  df.drop(['Price Change', 'Gain', 'Loss', 'Avg Gain', 'Avg Loss', 'RS'], axis=1, inplace=True)
  df1 = df[['Date', 'RSI']]
  yesterday_RSIDF = df1[df1['Date'] == yesterday_date]
  window_size = 20  # You can adjust the window size as needed
  df['SMA'] = df['Close'].rolling(window=window_size).mean()

# Calculate Exponential Moving Average (EMA)
  alpha = 2 / (window_size + 1)  # You can adjust the smoothing factor (alpha) as needed
  df['EMA'] = df['Close'].ewm(alpha=alpha, adjust=False).mean()

# Calculate Weighted Moving Average (WMA)
  weights = pd.Series(range(1, window_size + 1))  # You can adjust the weights as needed
  df['WMA'] = df['Close'].rolling(window=window_size).apply(lambda prices: (prices * weights).sum() / weights.sum(), raw=True)
  df1=df[['Date','Close','SMA','EMA','WMA']]
  yesterday_MA = df1[df1['Date'] == yesterday_date]
# Note: For the first window_size - 1 rows, NaN will be produced as there won't be enough data for the moving averages.
# You may choose to drop or handle these NaN values as per your requirement.
  mac = yesterday_MA['Close'].values[0]
  mas = yesterday_MA['SMA'].values[0]
  mae = yesterday_MA['EMA'].values[0]
  maw = yesterday_MA['WMA'].values[0]
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

def rsi_calculation(symbol):
  today = date.today() - timedelta(days=4)
  now = datetime.now()
  today_date = now.strftime('%Y-%m-%d')
  current_year = datetime.now().year
  current_year_start_date = str(current_year) + "-01-01"
  last_year = current_year - 1
  year_before_last = last_year - 1
  last_year_start_date = str(last_year) + "-01-01"
  last_year_end_date = str(last_year) + "-12-31"
  year_before_last_start_date = str(year_before_last) + "-01-01"
  year_before_last_ending_date = str(year_before_last) + "-12-31"
  historical_stock_data = yf.download(symbol, year_before_last_start_date, today_date)
  historical_stock_data_df = pd.DataFrame(historical_stock_data)
  historical_stock_data_df['lastDayOpen'] = historical_stock_data['Open'].shift(1)
  historical_stock_data_df['lastDayHigh'] = historical_stock_data['High'].shift(1)
  historical_stock_data_df['lastDayLow'] = historical_stock_data['Low'].shift(1)
  historical_stock_data_df['lastDayClose'] = historical_stock_data['Close'].shift(1)
  historical_stock_data_df['lastDayVolume'] = historical_stock_data['Volume'].shift(1)
  historical_stock_data_df.reset_index(inplace = True)
  today_date = now.strftime('%Y-%m-%d')
  yesterday = today - timedelta(days=7)
  yesterday_date = yesterday.strftime('%Y-%m-%d')
  yesterday_data = historical_stock_data_df[historical_stock_data_df['Date'] == yesterday_date]

# Check if there are any rows for yesterday's date
  if not yesterday_data.empty:
    yesterday_lastDayHigh = yesterday_data['lastDayHigh'].values[0]
    yesterday_lastDayClose = yesterday_data['lastDayClose'].values[0]
    yesterday_lastDayLow = yesterday_data['lastDayLow'].values[0]
  df=historical_stock_data_df
  df['Price Change'] = df['Close'] - df['lastDayClose']
  df['Gain'] = df['Price Change'].apply(lambda x: x if x > 0 else 0)
  df['Loss'] = df['Price Change'].apply(lambda x: -x if x < 0 else 0)
  period = 14
  df['Avg Gain'] = df['Gain'].rolling(window=period).mean()
  df['Avg Loss'] = df['Loss'].rolling(window=period).mean()
  df['RS'] = df['Avg Gain'] / df['Avg Loss']
  df['RSI'] = 100 - (100 / (1 + df['RS']))
  df.dropna(inplace=True)
  df.drop(['Price Change', 'Gain', 'Loss', 'Avg Gain', 'Avg Loss', 'RS'], axis=1, inplace=True)
  df1 = df[['Date', 'RSI']]
  yesterday_RSIDF = df1[df1['Date'] == yesterday_date]
  yRSI = yesterday_RSIDF['RSI'].values[0]
  if (yRSI>70):
    rsi = f"Security may be overbought and could potentially drop RSI : {yRSI}. "
  elif (yRSI>30 and yRSI<70):
    rsi = f"It falls in between the range security is neither significantly overbought nor oversold RSI : {yRSI}. "
  elif (yRSI <30):
    rsi = f"Security may be oversold and could potentially rise RSI : {yRSI}. "
  return rsi

import numpy as np

def probability(symbol):
  def rsi_value(sentence):
    pattern = r"RSI\s*:\s*(\d+\.?\d*)"

    match = re.search(pattern, sentence)
    if match:
        try:
            # Convert matched string to float, handling potential decimal separator variations
            yRSI = float(match.group(1).replace(",", ""))
            return yRSI
        except ValueError:
            raise ValueError("Extracted yRSI value is not numeric")
    else:
        return None
  
  def ta_value(sentence):
  # Regular expression pattern to match "uptrend" or "downtrend"
    pattern = r"(uptrend|downtrend)"

    # Extract sentiment
    match = re.search(pattern, sentence, flags=re.IGNORECASE)
    if match:
        sentiment = match.group(1).lower()  # Convert to lowercase
    else:
        raise ValueError("Sentence must contain 'uptrend' or 'downtrend'")
    if sentiment == "uptrend":
      ta_probability = 1.0
    elif sentiment == "downtrend":
      ta_probability = 0.0
    else:
      raise ValueError("Invalid TA value: {}".format(sentence))
    return ta_probability

  def volatility_calc(symbol):
    historical_data = yf.download(symbol, period="10d")
    if len(historical_data) < 2:
        raise ValueError("Need at least 2 closing prices to calculate volatility.")

    # Calculate daily percentage change
    daily_pct_change = historical_data['Close'].pct_change().dropna()

    # Calculate daily standard deviation
    daily_std = np.std(daily_pct_change)

    # Calculate annualized standard deviation
    annualized_std = daily_std * np.sqrt(252)

    # Calculate weekly percentage change
    weekly_price_change = ((historical_data["Close"].iloc[-1] - historical_data["Close"].iloc[0]) / historical_data["Close"].iloc[0]) * 100

    return annualized_std, weekly_price_change

  normalized_rsi = rsi_value(rsi_calculation(symbol)) / 100

  # Determine prob based on trend analysis
  ta_probability = ta_value(calculate_trend_analysis(symbol))

  # Determine prob based on  volitility & price change
  annualized_std_vol, weekly_price_change = volatility_calc(symbol)

  # Combine factors using a weighted average
  weights = np.array([0.25, 0.25, 0.25, 0.25])  # Adjust weights as needed
  factors = np.array([normalized_rsi, ta_probability, annualized_std_vol, weekly_price_change])
  probability1 = np.dot(weights, factors)
  #print(normalized_rsi, ta_probability, annualized_std_vol, weekly_price_change)
  prob = (probability1/1)*100
  prob1 = f"Probability of future earnings: {prob}. "
  return prob1

class StockPriceRSITrendAnalysisInput(BaseModel):
    """Input for Stock price check."""

    stockticker: str = Field(..., description="Ticker symbol for stock or index")

class StockChangePercentageCheckInput(BaseModel):
    """Input for Stock ticker check. for percentage check"""

    stockticker: str = Field(..., description="Ticker symbol for stock or index")
    days_ago: int = Field(..., description="Int number of days to look back")

class CompanyStockPriceRSITrendAnalysisTool(BaseTool):
    name = "get_stock_ticker_price"
    description = "Useful for when you need to find out the price of stock. You should input the stock ticker used on the yfinance API"

    def _run(self, stockticker: str):

        def PriceResponse(stockticker):
          #getting current price
          return f"Price is {get_stock_price(stockticker)}. "
          #getting past data
        def TACalculation(stockticker):
          return f"{calculate_trend_analysis(stockticker)}."
        def RSICalculation(stockticker):
          return f"{rsi_calculation(stockticker)}."
        def ProbabilityCalculation(stockticker):
            return f"{probability(stockticker)},"
        pr = PriceResponse(stockticker)
        ta = TACalculation(stockticker)
        rsi = RSICalculation(stockticker)
        prob = ProbabilityCalculation(stockticker)
        
        result = pr + "\n" + ta + "\n" + rsi+"\n" + prob
        return result

    def _arun(self, stockticker: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = StockPriceRSITrendAnalysisInput
class PercentageChangeTool(BaseTool):
    name = "get_price_change_percent"
    description = "Useful for when you need to find out the percentage change in a stock's value. You should input the stock ticker used on the yfinance API and also input the number of days to check the change over"

    def _run(self, stockticker: str, days_ago: int):
        def get_price_change_percent(symbol, days_ago):
          ticker = yf.Ticker(symbol)

    # Get today's date
          end_date = datetime.now()

    # Get the date N days ago
          start_date = end_date - timedelta(days=days_ago)

    # Convert dates to string format that yfinance can accept
          start_date = start_date.strftime('%Y-%m-%d')
          end_date = end_date.strftime('%Y-%m-%d')

    # Get the historical data
          historical_data = ticker.history(start=start_date, end=end_date)

    # Get the closing price N days ago and today's closing price
          old_price = historical_data['Close'].iloc[0]
          new_price = historical_data['Close'].iloc[-1]

    # Calculate the percentage change
          percent_change = ((new_price - old_price) / old_price) * 100

          return round(percent_change, 2)
        price_change_response = get_price_change_percent(stockticker, days_ago)

        return price_change_response

    def _arun(self, stockticker: str, days_ago: int):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = StockChangePercentageCheckInput

