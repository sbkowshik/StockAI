import google.generativeai as genai
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)

from dotenv import load_dotenv
import os 

def create_chat_bubble(text):
    chat_bubble_html = f"""
    <style>
    .chat-bubble {{
        max-width: 100%;
        margin: 10px;
        padding: 10px;
        background-color: #262730;
        border-radius: 16px;
        border: 1px solid #36454F;
    }}
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
os.environ["GOOGLE_API_KEY"]=st.secrets["GOOGLE_API_KEY"]

st.title("StockHelp")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
prompt = PromptTemplate.from_template("You are a helpful stock market assistant that serves to help users understand stock market terms. Check if {topic} is about stocks. If it is, explain the term like they are a high-schooler unless asked differently, else say the following: 'I am are unable to answer non-stockrelated questions, but if you have any questions about stock or the stock market, feel free to ask!'. Don't declare that the question is about stocks and do not use emojis.")
chain = prompt | llm | StrOutputParser()

if question:= st.chat_input("Ask your stock related questions"):
    with st.chat_message("user",avatar="😺"):
        st.markdown(question)
    with st.chat_message("assistant",avatar="🦖"):
        st_callback=StreamlitCallbackHandler(st.container())
        response=chain.invoke({"topic": question}, {"callbacks": [st_callback]})
        #st.markdown(create_chat_bubble(response), unsafe_allow_html=True)
        st.write(response)