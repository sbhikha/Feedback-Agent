from langchain_community.llms import Ollama
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import BaseTool
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from tavily import TavilyClient
import os

class FeedbackTools:
    name: str = "feedback_tools"
    description: str = "Feedback tools for each lesson type"

    def comp_question_feedback(self, question: str, context: str, correct_answer: str, student_response: str) -> str:
        feedback = f"The student was asked the given the following question and context:\
                Question: {question}\
                Context: {context}\
                Student Response: {student_response}\
                Correct Answer: {correct_answer}\
                If the student's response is incorrect, give constructive feedback on the response and ask them to try again\
                If the student's response is correct, congratulate them and proceed to the next exercise."
        print(f"Feedback: {feedback}")
        return feedback
    
    def mcq_question_feedback(self, question: str, context: str, correct_answer: str, student_response: str) -> str:
        feedback = f"The student was asked the given the following question and context:\
                Question: {question}\
                Context: {context}\
                Student Response: {student_response}\
                Correct Answer: {correct_answer}\
                If the student's response is incorrect, give constructive feedback on the response and ask them to try again\
                If the student's response is correct, congratulate them and proceed to the next exercise."
        print(f"Feedback: {feedback}")
        return feedback
    
    def fill_blank_question_feedback(self, question: str, context: str, correct_answer: str, student_response: str) -> str:
        feedback = f"The student was asked the given the following question and context:\
                Question: {question}\
                Context: {context}\
                Student Response: {student_response}\
                Correct Answer: {correct_answer}\
                If the student's response is incorrect, give constructive feedback on the response and ask them to try again\
                If the student's response is correct, congratulate them and proceed to the next exercise."
        print(f"Feedback: {feedback}")
        return feedback
    
    def pronunciation_feedback(self, sentence: str, pronunciation: str, student_response: str) -> str:
        feedback = f"The student was asked to pronounce the sentence: {sentence}\
                Student Response: {student_response}\
                If the student's pronunciation is incorrect, give constructive feedback on the response and ask them to try again\
                If the student's pronunciation is correct, congratulate them and proceed to the next exercise."
        print(f"Feedback: {feedback}")
        return feedback
    
    feedback_tools = [
        Tool(
            name="comp_question_feedback",
            description="Feedback for a comprehension question",
            function=comp_question_feedback
        ),
        Tool(
            name="mcq_question_feedback",
            description="Feedback for a multiple choice question",
            function=mcq_question_feedback
        ),
        Tool(
            name="fill_blank_question_feedback",
            description="Feedback for a fill in the blank question",
            function=fill_blank_question_feedback
        ),
        Tool(
            name="pronunciation_feedback",
            description="Feedback for pronunciation exercise",
            function=pronunciation_feedback
        )
    ]

    
class FeedbackAgent:
    def __init__(self, model: str):
        self.model = OllamaLLM(model)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.prompt = ChatPromptTemplate.from_template("""Question: {question}""")
        self.tools = FeedbackTools.feedback_tools
        self.chain = self.prompt | self.model

        self.agent = initialize_agent(
            tools=self.tools,
            model=self.model,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory
        )
    
    def invoke(self, query: str) -> str:
        return self.chain.run(input=query)
    

class TutorTools:
        class WebSearchTool(BaseTool):
            name: str = "web_search"
            description: str = "Search tool for finding information from the internet"

            def _run(self, query: str) -> str:
                tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
                response = tavily_client.extract(query)
                return response
            
            def _arun(self, query: str) -> str:
                raise NotImplementedError("Async not implemented")
