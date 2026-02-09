from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from dotenv import load_dotenv
from pathlib import Path
import sys

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()


# Allow running from project root (python src/graphConversation.py) or from src/
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def content_to_text(content):
    # string
    if isinstance(content, str):
        return content
    # Responses API: list of parts
    if isinstance(content, list):
        return "\n".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ).strip()
    return str(content)


def msg_role(m):
    # LangChain message objektumoknál ez a legbiztosabb
    if isinstance(m, HumanMessage):
        return "user"
    if isinstance(m, AIMessage):
        return "assistant"
    if isinstance(m, SystemMessage):
        return "system"
    # fallback: van ahol m.type = "human"/"ai"/"system"
    return getattr(m, "type", "unknown")


# --------- 1) State (a beszélgetés állapota) ---------
class State(TypedDict):
    messages: Annotated[
        list, add_messages
    ]  # LangGraph helper: hozzáfűzi az új üzeneteket
    turn: int  # hány "váltás" történt


memory = InMemorySaver()


# --------- 2) Modellek (egyik lehet local is) ---------
frontend_llm = ChatOpenAI(model="gpt-5-nano", temperature=0, use_responses_api=True)
ba_llm = ChatOpenAI(model="gpt-5-nano", temperature=0, use_responses_api=True)

# Local példa (ha kell)
# frontend_llm = ChatOpenAI(
#     model="local-model",
#     temperature=0,
#     base_url="http://localhost:1234/v1",
#     api_key="lm-studio",
#     use_responses_api=False,
# )


# --------- 3) Agentek ---------
frontend_agent = create_agent(
    model=frontend_llm,
    tools=[],
    system_prompt=(
        "You are a frontend software developer.\nAnswer shortly one or two sentences."
    ),
)

ba_agent = create_agent(
    model=ba_llm,
    tools=[],
    system_prompt=(
        "You are a business analyst.\n"
        "Ask EXACTLY ONE clarifying question per turn.\n"
        "Do NOT repeat earlier questions.\n"
        "Follow this order: (1) MVP scope, (2) storage choice, (3) edge cases.\n"
        "If scope and storage are decided, summarize requirements in 2 bullets and stop."
    ),
)


# --------- 4) Node-ok (egy node = egy agent lépése) ---------
def fe_node(state: State):
    result = frontend_agent.invoke({"messages": state["messages"]})
    # result["messages"] tartalmazza az egész addigi beszélgetést + új AI üzenetet
    return {"messages": result["messages"], "turn": state["turn"] + 1}


def ba_node(state: State):
    result = ba_agent.invoke({"messages": state["messages"]})
    return {"messages": result["messages"], "turn": state["turn"] + 1}


# --------- 5) Router (eldönti merre megy tovább / vége) ---------
def route(state: State):
    # 4 váltás = FE, BA, FE, BA (2 kör)
    if state["turn"] >= 4:
        return END
    # páros turn után FE, páratlan után BA (vagy fordítva)
    return "ba" if state["turn"] % 2 == 1 else "fe"


# --------- 6) Graph összerakása ---------
g = StateGraph(State)
g.add_node("fe", fe_node)
g.add_node("ba", ba_node)

g.set_entry_point("fe")  # FE kezd
g.add_conditional_edges("fe", route, {"fe": "fe", "ba": "ba", END: END})
g.add_conditional_edges("ba", route, {"fe": "fe", "ba": "ba", END: END})

app = g.compile(checkpointer=memory)


# --------- 7) Futtatás ---------
thread_id = "demo-thread-1"


initial = {
    "messages": [
        {
            "role": "user",
            "content": "Egy egyszerű todo listát szeretnék készíteni. Hogyan kezdjünk neki?",
        }
    ],
    "turn": 0,
}

for update in app.stream(
    initial,
    config={"configurable": {"thread_id": thread_id}},
    stream_mode="updates",
):
    # update pl: {"fe": {"messages": [...], "turn": 1}} vagy {"ba": {...}}
    for node_name, partial_state in update.items():
        msgs = partial_state.get("messages", [])
        if not msgs:
            continue
        last = msgs[-1]
        if isinstance(last, AIMessage):
            print(f"\n🤖 {node_name.upper()}:\n{content_to_text(last.content)}")
        elif isinstance(last, HumanMessage):
            print(f"\n👤 USER:\n{content_to_text(last.content)}")
