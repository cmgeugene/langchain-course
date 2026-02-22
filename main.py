from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch

load_dotenv()

@tool
def search(query: str) -> str:
    """
    Tool that searches over Internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)

#llm = ChatOllama(model="gemma3:4b")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages":HumanMessage(content="What is the weather in Incheon?")})
    print(result)


if __name__ == "__main__":
    main()
