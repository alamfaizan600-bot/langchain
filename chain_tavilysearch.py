from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
tools = [TavilySearch()]  # Initialize the Tavily client as a tool
agent = create_agent(model=llm, tools=tools)

output = agent.invoke(
    {"messages": [HumanMessage(content="What is the weather like in Delhi, India?")]}
)
print(output)
