from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain.agents import initialize_agent
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
from myagent import get_company_symbol,get_stock_price, calculate_rsi, ma, predict_stock,candlestick

import os
import yfinance as yf
import streamlit as st
import streamlit.components.v1 as components
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

sys_message = "You are StockAI, a stock market assistant. Answer the following questions as best you can. You have access to the following tools:\n\nget_company_symbol: get_company_symbol(symbol: str) -> str - Returns the ticker of the company inputted - Use this before all other tools get_stock_price: get_stock_price(symbol: str) -> float - Returns current price of ticker calculate_rsi: calculate_rsi(symbol: str) -> float - Return RSI Calculation of ticker ma: ma(ticker: str) -> str - Returns Moving Average of ticker predict_stock: predict_stock(ticker: str) -> float - Predicts the next day's closing value using ticker candlestick: candlestick(ticker: str) -> str - Returns the candlestick pattern as well as its learning by ticker\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [get_company_symbol, get_stock_price, calculate_rsi, ma, predict_stock, candlestick]  Choose the most suitable tool as per the query from the user and for the queries that are similar to 'Should I invest in' make sure you use all the tools to get the most information out of the agent. \n Action Input: the input to the action\nObservation: the result of the action... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: show outputs of all observations and answer to the input question\n\nBegin!\n\nQuestion: {input}\nThought:{agent_scratchpad}"

agent.agent.llm_chain.prompt.template = sys_message
def chart():
    with st.sidebar:
                ticker = yf.Ticker(question)
                htmlcode='''<!-- TradingView Widget BEGIN -->
            <div class="tradingview-widget-container">
            <div class="tradingview-widget-container__widget"></div>
            <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-symbol-overview.js" async>
            {
            "symbols": [
                [
                {symb}
                ]
            ],
            "chartOnly": false,
            "width": "100%",
            "height": "100%",
            "locale": "en",
            "colorTheme": "dark",
            "autosize": true,
            "showVolume": false,
            "showMA": false,
            "hideDateRanges": false,
            "hideMarketStatus": false,
            "hideSymbolLogo": false,
            "scalePosition": "right",
            "scaleMode": "Normal",
            "fontFamily": "-apple-system, BlinkMacSystemFont, Trebuchet MS, Roboto, Ubuntu, sans-serif",
            "fontSize": "10",
            "noTimeScale": false,
            "valuesTracking": "1",
            "changeMode": "price-and-percent",
            "chartType": "area",
            "maLineColor": "#2962FF",
            "maLineWidth": 1,
            "maLength": 9,
            "backgroundColor": "rgba(14, 17, 23, 1)",
            "lineWidth": 2,
            "lineType": 0,
            "dateRanges": [
                "1d|1",
                "1m|30",
                "3m|60",
                "12m|1D",
                "60m|1W",
                "all|1M"
            ]
            }
            </script>
            </div>
            <!-- TradingView Widget END -->'''.format(symb=ticker)
                components.html(htmlcode,height=550,width=400)
question=st.chat_input("Ask your stock related Questions")
if question:
    with st.chat_message("user",avatar="😺"):
        st.markdown(question)
        chart()
    with st.chat_message("assistant",avatar="🦖"):
        st_callback=StreamlitCallbackHandler(st.container())
        response=agent.invoke({"input": question}, {"callbacks": [st_callback]})
        st.markdown(response['output'])

