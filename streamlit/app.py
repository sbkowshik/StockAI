from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain.agents import initialize_agent

from myagent import get_company_symbol,get_stock_price, calculate_rsi, ma, predict_stock,candlestick

import os
import yfinance as yf
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

os.environ["HUGGINGFACEHUB_API_KEY"]=st.secrets["HUGGINGFACEHUB_API_TOKEN"]

st.title("StockAI")

tools = [get_company_symbol,get_stock_price,calculate_rsi,ma,predict_stock,candlestick]

llm = HuggingFaceEndpoint(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1")

chat_model = ChatHuggingFace(llm=llm)

agent = initialize_agent(tools,
                         llm,
                         agent="zero-shot-react-description",
                         verbose = True,
                         handle_parsing_errors=True
                         )

sys_message = "You are StockAI, a stock market assistant. Answer the following questions as best you can. You have access to the following tools:\n\nget_company_symbol: get_company_symbol(symbol: str) -> str - Returns the ticker of the company inputted - Use this before all other tools get_stock_price: get_stock_price(symbol: str) -> float - Returns current price of ticker calculate_rsi: calculate_rsi(symbol: str) -> float - Return RSI Calculation of ticker ma: ma(ticker: str) -> str - Returns Moving Average of ticker predict_stock: predict_stock(ticker: str) -> float - Predicts the next day's closing value using ticker candlestick: candlestick(ticker: str) -> str - Returns the candlestick pattern as well as its learning by ticker\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [get_company_symbol, get_stock_price, calculate_rsi, ma, predict_stock, candlestick]\nAction Input: the input to the action\nObservation: the result of the action... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: show outputs of all observations and answer to the input question\n\nBegin!\n\nQuestion: {input}\nThought:{agent_scratchpad}"

agent.agent.llm_chain.prompt.template = sys_message

st.title('Stock Analysis Agent')

user_input = st.text_input("Enter question or name of company","")

response = agent.invoke(user_input)
st.write(response['output'])
