import sys
from pathlib import Path
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict, List
from langchain.messages import HumanMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph import MessagesState
from typing_extensions import Literal

load_dotenv()

# Allow running from project root (python src/simpleConversation.py) or from src/
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# ---- MODEL ----
from src.agents import frontendDeveloperAgent  # noqa: E402
# from src.agents import businessAnalystAgent  # noqa: E402


# ---- STATE ----
class ChatState(MessagesState):
    summary: str


# ---- NODE ----
def chatbot(state: ChatState):
    response = frontendDeveloperAgent.invoke(state["messages"])
    return {"messages": response}


def should_continue(state: ChatState) -> Literal["chatbot", END]:
    latest_message = state["messages"][-1]
    if latest_message.content == "stop":
        return END
    return "chatbot"


# ---- GRAPH ----
builder = StateGraph(ChatState)
builder.add_node("chatbot", chatbot)

# ---- EDGES ----
builder.add_edge(START, "chatbot")
builder.add_conditional_edges(
    "chatbot", should_continue, {"chatbot": "chatbot", END: END}
)
builder.add_edge("chatbot", END)

# ---- COMPILE ----
graph = builder.compile()

# ---- RUN ----
state: ChatState = {"messages": []}
state = graph.invoke(state)
print(state["summary"])
