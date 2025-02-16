from langchain_community.llms import Ollama
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import BaseTool
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Load Ollama Models and test inference
class LLM:
    def __init__(self, model: str):
        self.model = Ollama(model)

    def invoke(self, query: str) -> str:
        return self.model(query)
    
    def test_chain(self, query: str) -> str:
        template = """Question: {question}
        Answer: Let's think step by step."""

        prompt = ChatPromptTemplate.from_template(template)

        test_chain = prompt | self.model

        return test_chain.invoke({"question": "What is LangChain?"})
