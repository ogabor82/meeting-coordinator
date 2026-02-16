from langgraph.graph import StateGraph, START, END, MessagesState
from dotenv import load_dotenv
import sys
from pathlib import Path
from typing import Literal

load_dotenv()

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.agents import frontendDeveloperAgent  # noqa: E402
from src.orchestratoragents import roleSelectorAgent  # noqa: E402
from src.agents import businessAnalystAgent  # noqa: E402


class ChatState(MessagesState):
    role: Literal["frontenddeveloper", "businessanalyst", "customer"]


def frontendDeveloperBot(state: ChatState):
    result = frontendDeveloperAgent.invoke({"messages": state["messages"]})
    return {"messages": result["messages"]}


def businessAnalystBot(state: ChatState):
    result = businessAnalystAgent.invoke({"messages": state["messages"]})
    return {"messages": result["messages"]}


def roleSelector(
    state: ChatState,
) -> Literal["frontenddeveloper", "businessanalyst"]:
    response = roleSelectorAgent.invoke({"messages": state["messages"]})
    # return {"role": response["messages"][-1].content[0]["text"]}
    return response["messages"][-1].content[0]["text"]


builder = StateGraph(ChatState)
builder.add_node("frontenddeveloper", frontendDeveloperBot)
builder.add_node("roleSelector", roleSelector)
builder.add_node("businessanalyst", businessAnalystBot)

builder.add_conditional_edges(
    START,
    roleSelector,
    {
        "frontenddeveloper": "frontenddeveloper",
        "businessanalyst": "businessanalyst",
    },
)
builder.add_edge("frontenddeveloper", END)
builder.add_edge("businessanalyst", END)


graph = builder.compile()
