import streamlit as st 
from myagent import CompanyStockPriceRSITrendAnalysisTool,PercentageChangeTool,YahooFinanceNewsTool
from myagent import get_stock_price,calculate_trend_analysis,rsi_calculation
from langchain import hub
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import AgentExecutor, create_openai_functions_agent
from dotenv import load_dotenv
import os
load_dotenv()
st.title('Stock Analysis Agent')

user_input = st.text_input("Enter question or name of company","")

tools = [CompanyStockPriceRSITrendAnalysisTool(),PercentageChangeTool(),YahooFinanceNewsTool()]
os.environ["OPENAI_API_KEY"]=st.secrets["OPENAI_API_KEY"]
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
open_ai_agent = initialize_agent(tools,
                        llm,
                        agent=AgentType.OPENAI_FUNCTIONS,
                        verbose=True)


prompt = hub.pull("hwchase17/openai-functions-agent")

llm = ChatOpenAI(temperature=0)

agent = create_openai_functions_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, return_intermediate_steps=True
)

dic = agent_executor.invoke({"input": user_input})

st.write(dic.get('output'))


