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
os.environ["HUGGINGFACEHUB_API_KEY"]=st.secrets["HUGGINGFACEHUB_API_TOKEN"]
st.title("StockAI")
st.caption("Analyzes technical factors of stocks to provide investment recommendations and comparisons.")
def get_system_prompt():
    return """You are StockAI, a stock market assistant. Answer the following questions as best you can. You have access to the following tools:
1. GetCompanySymbol
   - Input: Company name 
   - Output: Stock symbol / Ticker
   - MUST BE USED FIRST for any company analysis
   - Purpose: Ensures accurate symbol identification
   - Return ONLY THE STOCK SYMBOL NO EXTRA CHARACTERS OR INFORMATION.
2. GetStockPrice
   - Input: Stock symbol
   - Output: Current market price
   - Use: Establish current market position
   - Purpose: Foundation for all other analyses
3. CalculateRSI
   - Input: Stock symbol
   - Output: RSI value and interpretation
   - Interpretation:
     * RSI > 70: Potentially overbought
     * RSI < 30: Potentially oversold
     * RSI 30-70: Neutral range
   - Purpose: Momentum indicator
4. MovingAverage
   - Input: Stock symbol
   - Output: 50-day and 200-day moving averages
   - Interpretation:
     * Price > MA200: Long-term uptrend
     * Price < MA200: Long-term downtrend
     * MA50 crosses MA200: Potential trend change
   - Purpose: Trend analysis
5. Candlestick
   - Input: Stock symbol
   - Output: Candlestick pattern analysis
   - Use: Identify short-term price patterns
   - Purpose: Pattern recognition and support/resistance levels
6. PredictStock
   - Input: Stock symbol
   - Output: Price movement predictions
   - Use: ALWAYS after gathering all other data
   - Purpose: Forward-looking analysis based on historical patterns
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [GetCompanySymbol, GetStockPrice, CalculateRSI, MovingAverage, PredictStock, Candlestick]. Choose the most suitable tool as per the query from the user. For queries like 'Should I invest in', make sure you use each of the following tools: [GetCompanySymbol, GetStockPrice, CalculateRSI, MovingAverage, PredictStock, Candlestick] in that order.
Action Input: the input to the action
Observation: the result of the action... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: Give your answer from understanding the results from the actions along with an explanation, Do not give a netural answer. Understand how the stock is performing and then give your opinion as a Professional Stock Broker.
Begin!
Question: {input}
Thought: {agent_scratchpad}
Action: [The action you are taking]
Action Input: [The input to the action]"""


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

llm = HuggingFaceEndpoint(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1")

chat_model = ChatHuggingFace(llm=llm)
agent = initialize_agent(tools,
                         chat_model,
                         agent="zero-shot-react-description",
                         verbose = True,
                         handle_parsing_errors=True
                         )


agent.agent.llm_chain.prompt.template = get_system_prompt()

question=st.chat_input("Ask your stock related questions")
if question:
    with st.chat_message("user",avatar="😺"):
        st.markdown(question)
    with st.chat_message("assistant",avatar="🦖"):
        st_callback=StreamlitCallbackHandler(st.container())
        response=agent.invoke({"input": question}, {"callbacks": [st_callback]})
        res=response['output'].replace("$", "\$")
        st.write(res)
