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
prompt = PromptTemplate.from_template("You are a helpful stock market assistant that serves to help users understand stock market terms and concepts as well as questions on how to trade and investments involved in stock market trading. Check if {topic} is about stocks or stock market trading. If it is, explain the answer like they are a high-schooler unless asked differently, else say the following: 'I am are unable to answer non-stock related questions, but if you have any questions about stock or the stock market, feel free to ask!'. Don't declare that the question is about stocks and do not use emojis. Do not inform the user that you are explaining it like a high schooler. If a user asks for advice on investing in companies, end the answer with the following: 'For company specific analysis, a good recommendation would be StockAI!")
chain = prompt | llm | StrOutputParser()

if question:= st.chat_input("Ask your stock related questions"):
    with st.chat_message("user",avatar="😺"):
        st.markdown(question)
    with st.chat_message("assistant",avatar="🦖"):
        st_callback=StreamlitCallbackHandler(st.container())
        response=chain.invoke({"topic": question}, {"callbacks": [st_callback]})
        #st.markdown(create_chat_bubble(response), unsafe_allow_html=True)
        res=response.replace("$", "\$")
        st.write(response)
