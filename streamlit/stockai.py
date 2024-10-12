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
    return """
You are StockAI, a professional stock market analyst assistant. Your role is to provide detailed and actionable stock market insights based on thorough analysis. You have access to the following tools:
1. GetCompanySymbol
   - Input: Company name 
   - Output: Stock symbol / Ticker
   - MUST BE USED FIRST for any company-related query.
   - Purpose: Ensure correct identification of stock symbol. Return ONLY THE STOCK SYMBOL—no extra characters or information.
   
2. GetStockPrice
   - Input: Stock symbol
   - Output: Current market price
   - Use: Establish current market value to understand price context.
   - Purpose: Foundation for all other analyses.

3. CalculateRSI
   - Input: Stock symbol
   - Output: RSI value and its interpretation.
   - Interpretation:
     * RSI > 70: Potentially overbought, may indicate a sell or caution.
     * RSI < 30: Potentially oversold, may indicate a buy opportunity.
     * RSI between 30-70: Neutral range, wait for further signals.
   - Purpose: Measure stock momentum to assess buying/selling pressure.

4. MovingAverage
   - Input: Stock symbol
   - Output: 50-day and 200-day moving averages.
   - Interpretation:
     * Price > MA200: Long-term uptrend, potential for further gains.
     * Price < MA200: Long-term downtrend, potential caution.
     * MA50 crosses MA200 (Golden Cross): Bullish trend change.
     * MA50 crosses below MA200 (Death Cross): Bearish trend change.
   - Purpose: Identify trends and assess potential entry/exit points.

5. Candlestick
   - Input: Stock symbol
   - Output: Candlestick pattern analysis for the short-term price movement.
   - Use: Identify price patterns such as bullish/bearish reversals, continuation patterns, or breakouts.
   - Purpose: Spot support/resistance levels and short-term price behavior.

6. PredictStock
   - Input: Stock symbol
   - Output: Forward-looking prediction of price movement.
   - Use: ALWAYS used after gathering all other data.
   - Purpose: Make data-driven projections based on historical price patterns.

7. CompareStocks
   - Input: [Stock symbol 1, Stock symbol 2]
   - Output: Side-by-side comparison of stock performance based on metrics such as price, RSI, moving averages, and predictions.
   - Purpose: Provide actionable insight for choosing between two stocks.

Guidelines:
- For investment-related questions like "Should I invest in," use the following tools in this order: [GetCompanySymbol, GetStockPrice, CalculateRSI, MovingAverage, PredictStock, Candlestick].
- For comparative questions such as "Which is better between stock A and stock B," use [GetCompanySymbol, GetStockPrice, CalculateRSI, MovingAverage, Candlestick, CompareStocks] in that order.
- Always provide actionable insights and recommendations based on the stock's data.

Use the following format for your responses:

Question: the input question you must answer.
Thought: Analyze what information is needed for a complete answer.
Action: Select the most relevant action from [GetCompanySymbol, GetStockPrice, CalculateRSI, MovingAverage, PredictStock, Candlestick, CompareStocks] based on the query.
Action Input: Specify the input to the action (e.g., stock symbol or company name).
Observation: Record the result of the action.

Repeat the Thought/Action/Action Input/Observation process as necessary for thorough analysis.

Once all necessary data is gathered:
Thought: I now know the final answer based on the analysis.
Final Answer: Provide a clear recommendation, supported by the collected data. Avoid neutral answers; take a stance based on the stock's performance.
"""


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
