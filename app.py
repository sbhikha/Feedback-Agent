from langchain_ollama import ChatOllama
from Agent_Utils import FeedbackTools, TutorTools
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import tools_condition, ToolNode, create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph_supervisor import Supervisor
import matplotlib.pyplot as plt
import io
from PIL import Image as PILImage

# Define tools
search = TutorTools.TavilySearchTool
llm = TutorTools.LLMTool
multiply = FeedbackTools.multiply

tools = [multiply, 
         search,
        #  llm
        ]

llm = ChatOllama(model="llama3.2:3b")
llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Graph
builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node()
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
# builder.add_edge("assistant", END)
builder.add_edge("tools", "assistant")

## Memory
memory = MemorySaver()
react_graph_memory = builder.compile(checkpointer=memory)

## Show
def show_graph(graph):
    img_bytes = graph.get_graph(xray=True).draw_mermaid_png()
    img = PILImage.open(io.BytesIO(img_bytes))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def run_feedback_agent():
    # Specify a thread
    config = {"configurable": {"thread_id": "1"}}
    while True:
        user_input = input("Enter your message (or type 'exit' to quit): ")
        if user_input.strip().lower() in ("exit", "quit"):
            break
        response = react_graph_memory.invoke({"messages": [HumanMessage(content=user_input)]}, config)
        for msg in response["messages"]:
            msg.pretty_print()

if __name__ == "__main__":
    # show_graph(react_graph_memory)
    run_feedback_agent()