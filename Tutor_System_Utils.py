import os
import requests
from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import BaseTool
# from tavily import TavilyClient


class SupervisorAgent:
    """
    The SupervisorAgent coordinates the learning experience by:
    - Deploying agents for lesson planning, tutoring, and feedback.
    - Updating the student's profile based on feedback.
    """

    def __init__(self, llm=None, model: str = "llama3.2:3b"):
        self.llm = llm or OllamaLLM(model=model)
        self.supervisor_template = ChatPromptTemplate.from_template(
            """Given the student's profile: {student_profile},\
            deploy the lesson planning agent to generate a detailed lesson plan."""
        )

    def update_student_profile(self, student_profile: dict, feedback: dict) -> dict:
        """
        Update the student's profile based on the feedback data.
        This is a placeholder for a more sophisticated update algorithm.
        """
        # For example, update strengths, weaknesses, and progress.
        student_profile["strengths"] = feedback.get("strengths", student_profile.get("strengths", ""))
        student_profile["weaknesses"] = feedback.get("weaknesses", student_profile.get("weaknesses", ""))
        student_profile["progress"] = feedback.get("progress", student_profile.get("progress", 0))
        return student_profile



class PlanningAgent:
    """
    The PlanningAgent generates a detailed lesson plan based on the student's profile.
    - It uses a language model to generate the lesson plan.
    - It can be extended to include more sophisticated planning algorithms.
    """
    def __init__(self, llm=None, model: str = "deepseek-r1:8b"):
        # Use a shared LLM instance if provided, otherwise initialize a default OpenAI instance.
        self.llm = llm or OllamaLLM(model=model)
        # self.memory = ConversationBufferMemory(memory_key="supervisor_memory", return_messages=True)
        self.lesson_template = ChatPromptTemplate.from_template(
            # """Based on the student's profile: {student_profile}, "
            # "generate a detailed lesson plan for English learning. "
            # "Include objectives, exercises, and progression suggestions."""
            
            """Based on the student's profile: {student_profile},
            Adjust the lesson to suit the student's needs. Return the lesson plan in the same JSON format as the input.
            {
            "lesson": {
                "context": "Welcome, Agent. You have been selected by our CEO for a special mission as a Quality Control Manager. A manufacturing line has been facing issues with the quality of smartphones, specifically with screen assembly and battery installation. Customers have reported issues such as screen flickering, unresponsive touchscreens, and the battery drains quickly. These problems have resulted in returns and negative reviews, risking financial loss. Your mission is to address quality concerns and initiate the path to a solution. The company's image and customer satisfaction rely on your success. Good luck, Agent!",
                "questions": [
                {
                    "question": "What is the main issue with the smartphone production line?",
                    "options": [
                    "a) The assembly speed is too slow.",
                    "b) There are problems with the screens and batteries.",
                    "c) The smartphones are too expensive."
                    ],
                    "answer_key": "b) There are problems with the screens and batteries.",
                    "feedback": {
                    "positive": "Great job! You identified the key issues affecting the smartphone quality. Keep it up, Agent!",
                    "constructive": "Not quite! Consider what specific smartphone parts were causing issues for customers. Review the mission details and try again."
                    }
                },
                {
                    "question": "How are smartphone problems affecting the company?",
                    "options": [
                    "a) Increased production costs.",
                    "b) A lack of materials.",
                    "c) Customer returns and negative reviews."
                    ],
                    "answer_key": "c) Customer returns and negative reviews.",
                    "feedback": {
                    "positive": "Excellent! You understand the impact of these issues on customer satisfaction and the company's reputation.",
                    "constructive": "Think about how the company's customers reacted to the smartphone problems. Revisit the mission briefing and give it another try."
                    }
                }
                ]
            }
            }
            """
        )
        self.lesson_chain = self.lesson_template | self.llm

    def generate_lesson_plan(self, student_profile: dict) -> str:
        """
        Generate a lesson plan based on the student's profile.
        """
        # Format the profile into a string for the LLM prompt.
        profile_str = ", ".join(f"{k}: {v}" for k, v in student_profile.items())
        lesson_plan = self.lesson_chain.invoke(input=profile_str)
        return lesson_plan


class TutorAgent:
    """
    The TutorAgent acts as an interactive tutor.
    - It takes a lesson plan and student responses.
    - It provides in-session feedback and suggestions.
    - It logs performance data for later analysis.
    """

    def __init__(self, llm=None, model: str = "llama3.2:3b"):
        self.llm = llm or OllamaLLM(model=model)
        self.tutoring_template = ChatPromptTemplate.from_template(
                """Given the lesson plan: {lesson_plan}\n"
                "and the student's response: {student_response}\n"
                "provide tutoring feedback, corrections, and suggestions."""
        )
        self.tutor_chain = self.tutoring_template | self.llm
        self.performance_logs = []

    def conduct_lesson(self, lesson_plan: str, student_responses: list) -> list:
        """
        Conduct the lesson by processing each student response.
        Returns a list of performance log entries.
        """
        for response in student_responses:
            feedback = self.tutor_chain.invoke(input={"lesson_plan": lesson_plan, "student_response": response})
            # Log each interaction
            log_entry = {
                "student_response": response,
                "tutor_feedback": feedback
            }
            self.performance_logs.append(log_entry)
        return self.performance_logs

    def get_performance_logs(self) -> list:
        """
        Retrieve the collected performance logs.
        """
        return self.performance_logs


class FeedbackAgent:
    """
    The FeedbackAgent generates detailed feedback after a tutoring session.
    - It uses the logged performance data.
    - It integrates with an external API (e.g., speechsuper) to generate detailed competency scores.
    """

    def __init__(self, llm=None, speechsuper_api_url: str = None, model: str = "llama3.2:3b"):
        self.llm = llm or OllamaLLM(model=model)
        self.feedback_template = ChatPromptTemplate.from_template(
                """Analyze the following performance data from the tutoring session: {performance_data}\n"
                "Generate detailed feedback including competency scores and recommendations."""
        )
        self.feedback_chain = self.feedback_template | self.llm
        self.speechsuper_api_url = speechsuper_api_url or "https://api.speechsuper.com/analyze"

    def generate_feedback(self, performance_logs: list) -> dict:
        """
        Generate detailed feedback using the performance logs and the speechsuper API.
        """
        # Create a summary string from the performance logs.
        performance_summary = "\n".join(
            f"Response: {log['student_response']}\nFeedback: {log['tutor_feedback']}"
            for log in performance_logs
        )
        # Generate initial feedback using the LLM.
        feedback_text = self.feedback_chain.invoke(input=performance_summary)
        # Optionally, get additional analysis from the speechsuper API.
        additional_scores = self.call_speechsuper_api(performance_summary)
        # Combine the outputs.
        feedback = {
            "detailed_feedback": feedback_text,
            "competency_scores": additional_scores
        }
        return feedback

    def call_speechsuper_api(self, performance_summary: str) -> dict:
        """
        Simulate a call to the speechsuper API for additional analysis.
        In a real-world scenario, proper authentication and error handling are necessary.
        """
        try:
            response = requests.post(
                self.speechsuper_api_url,
                json={"performance_summary": performance_summary},
                timeout=5,
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": "Speechsuper API error", "status_code": response.status_code}
        except Exception as e:
            return {"error": str(e)}


class LanguageLearningSystem:
    """
    Utility class to integrate all agents into a coherent language learning system.
    This class can be directly linked to FastAPI endpoints.
    """

    def __init__(self):
        self.supervisor = SupervisorAgent()
        self.tutor = TutorAgent()
        self.feedback = FeedbackAgent()

    def run_session(self, student_profile: dict, student_responses: list) -> dict:
        """
        Run a complete learning session:
        1. Generate a lesson plan.
        2. Conduct the tutoring session.
        3. Generate feedback.
        4. Update the student's profile.
        Returns a summary of the session.
        """
        # Generate the lesson plan.
        lesson_plan = self.supervisor.generate_lesson_plan(student_profile)
        # Conduct the lesson and capture performance logs.
        performance_logs = self.tutor.conduct_lesson(lesson_plan, student_responses)
        # Generate detailed feedback.
        detailed_feedback = self.feedback.generate_feedback(performance_logs)
        # Update the student's profile based on feedback.
        updated_profile = self.supervisor.update_student_profile(student_profile, detailed_feedback)
        return {
            "lesson_plan": lesson_plan,
            "performance_logs": performance_logs,
            "feedback": detailed_feedback,
            "updated_profile": updated_profile
        }


# Example usage (this block can be removed when integrating with FastAPI)
if __name__ == "__main__":
    # Example student profile and responses.
    student_profile = {
        "level": "Intermediate",
        "progress": 50,
        "strengths": "Vocabulary",
        "weaknesses": "Pronunciation"
    }
    student_responses = [
        "I am going to the store to buy groceries.",
        "Yesterday I read an interesting article about technology."
    ]

    # Instantiate the system and run a session.
    system = LanguageLearningSystem()
    session_result = system.run_session(student_profile, student_responses)
    
    print("Lesson Plan:\n", session_result["lesson_plan"])
    print("\nPerformance Logs:")
    for log in session_result["performance_logs"]:
        print(log)
    print("\nFeedback:\n", session_result["feedback"])
    print("\nUpdated Profile:\n", session_result["updated_profile"])