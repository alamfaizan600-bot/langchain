from typing import List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

load_dotenv()
from pydantic import BaseModel, Field


class Source(BaseModel):
    """
    A class representing a source of information used to generate the answer.
    """

    title: str = Field(..., description="The title of the source")
    url: str = Field(..., description="The URL of the source")


class AgentResponse(BaseModel):
    """
    A class representing the response from the agent, including the answer and the sources used to generate it.
    """

    answer: str = Field(..., description="The answer to the user's query")
    sources: List[Source] = Field(
        ..., description="A list of sources used to generate the answer"
    )


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

output = agent.invoke(
    {
        "messages": [
            HumanMessage(
                content="Give me the top generative ai based jobs near gurgaon, india and new york, usa? And give me the sources as well."
            )
        ]
    }
)
