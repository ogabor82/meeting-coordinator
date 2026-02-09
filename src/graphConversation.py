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
from pydantic_core import Url

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
# frontend_llm = ChatOpenAI(model="gpt-5-nano", temperature=0, use_responses_api=True)
# ba_llm = ChatOpenAI(model="gpt-5-nano", temperature=0, use_responses_api=True)

frontend_llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0, use_responses_api=True)
ba_llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0, use_responses_api=True)


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
    system_prompt="""
        You are a senior frontend engineer working in a product discovery meeting.

        GOAL:
        Move the implementation forward step-by-step toward a minimal working MVP.

        RULES:
        - Answer in max 5 short sentences.
        - Do NOT repeat previous questions.
        - If the BA asks a question, ANSWER it with a concrete choice (pick one).
        - Then propose the next technical step as a single actionable instruction.
        - Avoid generic phrasing like "let's create an HTML page" unless you include the exact elements/IDs to create.
        - Always reply in Hungarian.
        - Never ask questions; only make decisions and give next steps.        


        STOP CONDITION:
        If MVP scope and storage are decided, output:
        "MVP agreed. Ready to implement."
        and stop asking anything.
        """,
)

ba_agent = create_agent(
    model=ba_llm,
    tools=[],
    system_prompt=(
        "You are a business analyst.\n"
        "Always reply in Hungarian.\n"
        "Ask EXACTLY ONE clarifying question per turn.\n"
        "Do NOT repeat earlier questions.\n"
        "Follow this order: (1) MVP scope, (2) storage choice, (3) edge cases.\n"
        "If scope and storage are decided, summarize requirements in 2 bullets and stop."
        "If the FE did not answer your last question, repeat the SAME question once, shorter, and stop."
    ),
)

customer_llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0, use_responses_api=True)
# customer_llm = ChatOpenAI(
#     model="gpt-5-nano",
#     base_url="http://localhost:1234/v1",
#     api_key="lm-studio",
#     use_responses_api=False,
# )
customer_agent = create_agent(
    model=customer_llm,
    tools=[],
    system_prompt="""
        You are a the customer of the product discovery meeting.
        Always reply in Hungarian.
        Always answer the business analyst's questions.
        You would like ttom implement a Duolingo like learning platform for children.
        The platform should have the following features:
        - A login page
        - A registration page
        - A dashboard page
        - A profile page
        - A settings page
        - A logout page
    """,
)


# --------- 4) Node-ok (egy node = egy agent lépése) ---------
def fe_node(state: State):
    result = frontend_agent.invoke({"messages": state["messages"]})
    last = result["messages"][-1]  # AIMessage
    return {
        "messages": [AIMessage(content=last.content, name="frontend engineer")],
        "turn": state["turn"] + 1,
    }


def ba_node(state: State):
    result = ba_agent.invoke({"messages": state["messages"]})
    last = result["messages"][-1]
    return {
        "messages": [AIMessage(content=last.content, name="business analyst")],
        "turn": state["turn"] + 1,
    }


def customer_node(state: State):
    result = customer_agent.invoke({"messages": state["messages"]})
    last = result["messages"][-1]
    return {
        "messages": [AIMessage(content=last.content, name="customer")],
        "turn": state["turn"] + 1,
    }


# --------- 5) Router (eldönti merre megy tovább / vége) ---------
def route(state: State):
    # 4 váltás = CUSTOMER, FE, BA, FE, BA (2 kör)
    if state["turn"] >= 12:
        return END

    queue = [
        "customer",
        "fe",
        "ba",
        "fe",
        "ba",
        "customer",
        "fe",
        "ba",
        "fe",
        "ba",
        "customer",
        "fe",
        "ba",
        "fe",
        "ba",
    ]
    return queue[state["turn"]]


# --------- 6) Graph összerakása ---------
g = StateGraph(State)
g.add_node("fe", fe_node)
g.add_node("ba", ba_node)
g.add_node("customer", customer_node)

g.set_entry_point("customer")  # Customer kezd
g.add_conditional_edges(
    "fe", route, {"fe": "fe", "ba": "ba", "customer": "customer", END: END}
)
g.add_conditional_edges(
    "ba", route, {"fe": "fe", "ba": "ba", "customer": "customer", END: END}
)
g.add_conditional_edges(
    "customer", route, {"fe": "fe", "ba": "ba", "customer": "customer", END: END}
)


app = g.compile(checkpointer=memory)


# --------- 7) Futtatás ---------
thread_id = "demo-thread-1"


initial = {
    "messages": [
        {
            "role": "user",
            # "content": "Egy egyszerű todo listát szeretnék készíteni. Hogyan kezdjünk neki?",
            # "content": "Egy egyszerű kereshető és sorrendezhető táblázat kellene a user-ek oldalra. Hogyan kezdjünk neki?",
            "content": "Egy Duolingo szerű oktatóoldalt szeretnénk gyerekeknek készíteni. Hogyan kezdjünk neki?",
        }
    ],
    "turn": 0,
}

last_printed_id = None

for state in app.stream(
    initial,
    config={"configurable": {"thread_id": thread_id}},
    stream_mode="values",
):
    last = state["messages"][-1]

    # duplikált kiírás ellen (ugyanazt a messaget többször is megkaphatod)
    mid = getattr(last, "id", None)
    if mid and mid == last_printed_id:
        continue
    last_printed_id = mid

    if isinstance(last, AIMessage):
        who = (getattr(last, "name", None) or "assistant").upper()
        print(f"\n🤖 {who}:\n{content_to_text(last.content)}")
    elif isinstance(last, HumanMessage):
        print(f"\n👤 USER:\n{content_to_text(last.content)}")
