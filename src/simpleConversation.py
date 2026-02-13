from langgraph.graph import StateGraph, START, END, MessagesState
from dotenv import load_dotenv
import sys
from pathlib import Path

load_dotenv()

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.agents import frontendDeveloperAgent  # noqa: E402


class ChatState(MessagesState):
    pass


def chatbot(state: ChatState):
    result = frontendDeveloperAgent.invoke({"messages": state["messages"]})
    return {"messages": result["messages"]}


builder = StateGraph(ChatState)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()
