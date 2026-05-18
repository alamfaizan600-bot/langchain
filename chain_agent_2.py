from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

tavily = TavilyClient()  # Initialize the Tavily client
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)


@tool
def search(query: str) -> str:
    """
    Arguments:
    query: The search query to be executed.
    Returns:
    A string representing the search results.
    """
    print(f"Executing search for query: {query}")
    return tavily.search(query=query)


tools = [search]
agent = create_agent(model=llm, tools=tools)

output = agent.invoke(
    {
        "messages": [
            HumanMessage(
                content="Give me the top generative ai based jobs near gurgaon, india and new york, usa? And give me the sources as well."
            )
        ]
    }
)
print(output)
