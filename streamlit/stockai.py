from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain.agents import initialize_agent
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
from langchain.tools import Tool
from myagent import get_company_symbol,get_stock_price, calculate_rsi, moving_average, predict_stock,candlestick
import os
import yfinance as yf
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
    
load_dotenv()
os.environ["MISTRAL_API_KEY"]=st.secrets["MISTRAL_API_KEY"]
st.title("StockAI")
st.caption("Analyzes technical factors of stocks to provide investment recommendations and comparisons.")
tools = [
    Tool(
        name="GetCompanySymbol",
        func=get_company_symbol,
        description="Gets the stock symbol for a company name"
    ),
    Tool(
        name="GetStockPrice",
        func=get_stock_price,
        description="Gets the current stock price for a given symbol"
    ),
    Tool(
        name="CalculateRSI",
        func=calculate_rsi,
        description="Calculates the Relative Strength Index for a stock"
    ),
    Tool(
        name="MovingAverage",
        func=moving_average,
        description="Calculates moving averages for a stock"
    ),
    Tool(
        name="PredictStock",
        func=predict_stock,
        description="Predicts future stock price movements"
    ),
    Tool(
        name="Candlestick",
        func=candlestick,
        description="Generates candlestick chart analysis"
    )
]

llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0,
)

agent = initialize_agent(tools,
                         llm,
                         agent="zero-shot-react-description",
                         verbose = True,
                         handle_parsing_errors=True
                         )

sys_message = "You are StockAI, a stock market assistant. Answer the following questions as best you can. You have access to the following tools:\n\nget_company_symbol: get_company_symbol(symbol: str) -> str - Returns the ticker of the company inputted - Use this before all other tools get_stock_price: get_stock_price(symbol: str) -> float - Returns current price of ticker calculate_rsi: calculate_rsi(symbol: str) -> float - Return RSI Calculation of ticker moving_average: moving_average(ticker: str) -> str - Returns Moving Average of ticker predict_stock: predict_stock(ticker: str) -> float - Predicts the next day's closing value using ticker candlestick: candlestick(ticker: str) -> str - Returns the candlestick pattern as well as its learning by ticker\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [get_company_symbol, get_stock_price, calculate_rsi, moving_average, predict_stock, candlestick]  Choose the most suitable tool as per the query from the user and for the queries that are similar to 'Should I invest in' make sure you use each and every one of the following tools: [get_company_symbol, get_stock_price, calculate_rsi, moving_average, predict_stock, candlestick] in the same order.  \n Action Input: the input to the action\nObservation: the result of the action... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: show outputs of all observations and answer to the input question\n\nBegin!\n\nQuestion: {input}\nThought:{agent_scratchpad}"

agent.agent.llm_chain.prompt.template = sys_message

question=st.chat_input("Ask your stock related questions")
if question:
    with st.chat_message("user",avatar="😺"):
        st.markdown(question)
    with st.chat_message("assistant",avatar="🦖"):
        st_callback=StreamlitCallbackHandler(st.container())
        response=agent.invoke({"input": question}, {"callbacks": [st_callback]})
        res=response['output'].replace("$", "\$")
        st.write(res)
