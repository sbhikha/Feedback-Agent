from langchain_community.llms import Ollama
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import BaseTool
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# ! pip install -r requirements.txt


template = """Question: {question}
Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)

supervisor_model = OllamaLLM(model="deepseek-r1:8b")
tutor_model = OllamaLLM(model="llama3.2:3b")

test_chain = prompt | supervisor_model
test_chain.invoke({"question": "What is LangChain?"})


# Create a memory buffer for the tutor agent
tutor_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

tutor_feedback = ConversationBufferMemory(
    memory_key="feedback",
    return_messages=True
)

# Define the tutor agent's prompt
tutor_prompt = PromptTemplate(
    input_variables=["chat_history", "input"],
    template="""You are an English language tutor. Your role is to:
    1. Provide clear explanations of the lesson materials
    2. Provide constructive feedback to the student after each response
    3. Maintain an encouraging and supportive tone
    4. Keep a list of the student's strengths and weaknesses and add this to the feedback memory

    Lession Materials:
    1. Grammar Basics
    1.1 Identify the subject and predicate in the following sentence: 'The cat sat on the mat.'


    Previous conversation:
    {chat_history}

    Student's input: {input}
    Tutor's response:"""
)

# Create the tutor chain
tutor_chain = LLMChain(
    llm=model,
    prompt=tutor_prompt,
    memory=tutor_memory,
    verbose=True
)

tutor_chain = 

# Define the tutor tool
class TutorTool(BaseTool):
    name: str = "english_tutor"
    description: str = "Use this tool to interact with the English tutor for teaching and feedback"
    
    def _run(self, query: str) -> str:
        return tutor_chain.run(input=query)
    
    def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not implemented")

#Define the tutor tools

class IntroduceLessonTool(BaseTool):
    name: str = "introduce_lesson"
    description: str = "Introduce the lesson material"
    
    def _run(self, lesson_introduction: str) -> str:
        introduction = f"Today's lesson is about: {lesson_introduction}"
        print(introduction)
        return introduction
    
    def _arun(self, lesson_introduction: str) -> str:
        raise NotImplementedError("Async not implemented")

class ReadExerciseTool(BaseTool):
    name: str = "read_exercise"
    description: str = "Read out the first exercise and await a response"
    
    def _run(self, exercise: str) -> str:
        print(f"Exercise: {exercise}")
        return exercise
    
    def _arun(self, exercise: str) -> str:
        raise NotImplementedError("Async not implemented")

class GiveFeedbackTool(BaseTool):
    name: str = "give_feedback"
    description: str = "Give feedback on the response and proceed to the next exercise"
    
    def _run(self, student_response: str) -> str:
        feedback = tutor_chain.run(input=student_response)
        print(f"Feedback: {feedback}")
        return feedback
    
    def _arun(self, student_response: str) -> str:
        raise NotImplementedError("Async not implemented")

class UpdateFeedbackMemoryTool(BaseTool):
    name: str = "update_feedback_memory"
    description: str = "Update the tutor_feedback memory buffer"
    
    def _run(self, feedback: str) -> str:
        tutor_feedback.add_message(feedback)
        print("Feedback memory updated.")
        return "Feedback memory updated."
    
    def _arun(self, feedback: str) -> str:
        raise NotImplementedError("Async not implemented")

# Create tools list for supervisor agent
tools = [
    TutorTool(),
    IntroduceLessonTool(),
    ReadExerciseTool(),
    GiveFeedbackTool(),
    UpdateFeedbackMemoryTool(),
]

# Example usage
# deploy_tool = DeployTutorTool()
# deploy_tool._run("")

# introduce_tool = IntroduceLessonTool()
# introduce_tool._run("Grammar Basics")

# read_tool = ReadExerciseTool()
# read_tool._run("Identify the subject and predicate in the following sentence: 'The cat sat on the mat.'")

# student_response = "The subject is 'The cat' and the predicate is 'sat on the mat.'"
# feedback_tool = GiveFeedbackTool()
# feedback = feedback_tool._run(student_response)

# update_tool = UpdateFeedbackMemoryTool()
# update_tool._run(feedback)


# Create supervisor agent memory
supervisor_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Initialize the supervisor agent
supervisor_agent = initialize_agent(
    tools,
    model,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=supervisor_memory,
    verbose=True
)

# Define the supervisor's system prompt
supervisor_system_prompt = """You are a supervisor agent responsible for managing an English tutoring session. Your role is to:
1. Assess the student's needs and current English level based on tutor_feedback memory.
2. Deploy the English tutor agent to teach the lesson materials.
3. Provide high-level guidance and structure to the session.
4. Once the lesson is complete, review the feedback memory and plan the next lesson accordingly.

Use the english_tutor tool to delegate actual teaching tasks and student interaction.
Always maintain a clear structure and learning objectives for the session."""
### Run the session
# Function to run a tutoring session
def run_tutoring_session(user_input: str) -> str:
    response = supervisor_agent.run(
        input=f"{supervisor_system_prompt}\n\nStudent input: {user_input}"
    )
    return response

# Example usage
if __name__ == "__main__":
    # Example interaction
    student_input = "Hi, I'm an intermediate English learner and I need help with phrasal verbs."
    response = run_tutoring_session(student_input)
    print(response)