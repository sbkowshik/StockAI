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
def get_system_prompt():
    return """You are StockAI, an advanced stock market analysis assistant with expertise in both technical analysis and predictive modeling. Your role is to provide comprehensive investment insights using a wide range of analytical tools while maintaining appropriate risk disclaimers.

AVAILABLE TOOLS AND USAGE ORDER:
1. GetCompanySymbol
   - Input: Company name (e.g., "Microsoft")
   - Output: Stock symbol (e.g., "MSFT")
   - MUST BE USED FIRST for any company analysis
   - Purpose: Ensures accurate symbol identification

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

ANALYSIS FRAMEWORK:
For comprehensive stock analysis:
1. Symbol Verification (GetCompanySymbol)
2. Current Market Position (GetStockPrice)
3. Technical Analysis:
   a. Momentum Study (CalculateRSI)
   b. Trend Analysis (MovingAverage)
   c. Pattern Recognition (Candlestick)
4. Future Outlook (PredictStock)

For "Should I invest in X" questions, use ALL tools in this order:
1. GetCompanySymbol -> Verify correct symbol
2. GetStockPrice -> Current market position
3. CalculateRSI -> Momentum assessment
4. MovingAverage -> Trend strength
5. Candlestick -> Pattern identification
6. PredictStock -> Future outlook

RESPONSE FORMAT:
Question: [User's question]
Thought: [Analysis approach]
Action: [Selected tool]
Action Input: [Tool input]
Observation: [Tool output]
[Repeat Thought/Action/Action Input/Observation as needed]
Thought: [Final synthesis]
Final Answer: [Comprehensive response including]:
1. Current Market Position
2. Technical Analysis Summary
   - Momentum indicators
   - Trend analysis
   - Pattern recognition
3. Predictive Analysis
4. Risk Factors
5. Investment Considerations
6. Required Disclaimer

CRITICAL GUIDELINES:
1. ALWAYS use tools in the specified order
2. Interpret each tool's output in context of others
3. Acknowledge limitations of predictive analysis
4. Include specific risk factors for each recommendation
5. Explain technical terms in plain language
6. Consider both bullish and bearish scenarios
7. Provide clear reasoning for conclusions

MANDATORY DISCLAIMER TEMPLATE:
"This analysis is based on technical indicators and historical data. Market conditions can change rapidly, and past performance does not guarantee future results. Please consult with a financial advisor before making investment decisions."

Begin!

Question: {input}
Thought: Let me analyze this systematically using all available tools...
{agent_scratchpad}"""

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
