from typing import List, Dict, Any
from tavily import TavilyClient
from langchain_ollama import ChatOllama
import os

class FeedbackTools:
    def multiply(a: int, b: int) -> int:
        """Multiply a and b.

        Args:
            a: first int
            b: second int
        """
        return a * b
    

class TutorTools:
        def TavilySearchTool(query: str) -> Dict[str, Any]:
            """Search Tavily for information.
            Use this tool for gaining up to date, topic specific information.
            Args:
                query: the search query
            """
            tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
            response = tavily_client.search(query)
            return response
        
        def LLMTool(query: str) -> Dict[str, Any]:
            """Search LLM for information.
            Use this tool for general queries - those that do not require up to date information.
            Args:
                query: the search query
            """
            llm_client = ChatOllama(model="llama3.2:3b")
            response = llm_client.invoke(query)
            return response
