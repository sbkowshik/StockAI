from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain.agents import initialize_agent
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
from pathlib import Path
from myagent import get_company_symbol,get_stock_price, calculate_rsi, moving_average, predict_stock,candlestick
import os
import yfinance as yf
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
def create_chat_bubble(text):
    chat_bubble_html = f"""
    <style>
    .chat-container {{
        display: flex;
        flex-direction: column;
        align-items: flex-start;
    }}
    </style>
    <div class="chat-container">
            {text}
    </div>
    """
    return chat_bubble_html
    
load_dotenv()
os.environ["HUGGINGFACEHUB_API_KEY"]=st.secrets["HUGGINGFACEHUB_API_TOKEN"]
os.environ["GOOGLE_API_KEY"]=st.secrets["GOOGLE_API_KEY"]
st.code(Path(".streamlit\config.toml").read_text(), language="toml")
sth=st.Page("stockhelp.py",title="StockHelp",icon=":material/search:")
sta=st.Page("stockai.py",title="StockAI",icon=":material/dashboard:")
pg=st.navigation([sta,sth])
pg.run()
