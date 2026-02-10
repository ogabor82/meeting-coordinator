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
import hashlib

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


def build_customer_agent(customer_llm, product_brief: str):
    return create_agent(
        model=customer_llm,
        tools=[],
        system_prompt=f"""
    You are the customer of the product discovery meeting.
    Always reply in Hungarian.
    Always answer the business analyst's questions.

    PRODUCT BRIEF:
    {product_brief}
    """.strip(),
    )


class State(TypedDict):
    messages: Annotated[
        list, add_messages
    ]  # LangGraph helper: hozzáfűzi az új üzeneteket
    turn: int  # hány "váltás" történt


def route(state: State):
    # 4 váltás = CUSTOMER, FE, BA, FE, BA (2 kör)
    if state["turn"] >= 6:
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


memory = InMemorySaver()

SYSTEM_PROMPT_FRONTEND_DEVELOPER = """
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
        """

SYSTEM_PROMPT_BUSINESS_ANALYST = """
        You are a business analyst.
        Always reply in Hungarian.
        Ask EXACTLY ONE clarifying question per turn.
        Do NOT repeat earlier questions.
        Follow this order: (1) MVP scope, (2) storage choice, (3) edge cases.
        If scope and storage are decided, summarize requirements in 2 bullets and stop.
        If the FE did not answer your last question, repeat the SAME question once, shorter, and stop.
"""


def build_app(product_brief: str):
    memory = InMemorySaver()

    frontend_llm = ChatOpenAI(
        model="gpt-4.1-nano", temperature=0, use_responses_api=True
    )
    ba_llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0, use_responses_api=True)
    customer_llm = ChatOpenAI(
        model="gpt-4.1-nano", temperature=0, use_responses_api=True
    )

    frontend_agent = create_agent(
        model=frontend_llm,
        tools=[],
        system_prompt=SYSTEM_PROMPT_FRONTEND_DEVELOPER,
    )

    ba_agent = create_agent(
        model=ba_llm,
        tools=[],
        system_prompt=SYSTEM_PROMPT_BUSINESS_ANALYST,
    )

    customer_agent = build_customer_agent(customer_llm, product_brief)

    def fe_node(state: State):
        result = frontend_agent.invoke({"messages": state["messages"]})
        last = result["messages"][-1]
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

    g = StateGraph(State)
    g.add_node("fe", fe_node)
    g.add_node("ba", ba_node)
    g.add_node("customer", customer_node)

    g.set_entry_point("customer")
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
    return app


PRODUCT_BRIEF = """
    Goal: single-product landing page + checkout flow.
    MVP features:
    - Landing page: benefits, FAQ, reviews (mock), price
    - Checkout: shipping info + payment (can be Stripe test)
    - Order confirmation page + email receipt (can be mocked)
    Constraints:
    - Must be mobile-first and fast
    Non-goals:
    - Full catalog, user accounts (later)
"""


def run_conversation(
    app, user_message: str, thread_id: str, *, is_new_thread: bool = False
):
    initial = {
        "messages": [{"role": "user", "content": user_message}],
    }
    if is_new_thread:
        initial["turn"] = 0

    last_printed_id = None
    final_state = None

    for state in app.stream(
        initial,
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="values",
    ):
        final_state = state
        last = state["messages"][-1]

        mid = getattr(last, "id", None)
        if mid and mid == last_printed_id:
            continue
        last_printed_id = mid

        if isinstance(last, AIMessage):
            who = (getattr(last, "name", None) or "assistant").upper()
            print(f"\n🤖 {who}:\n{content_to_text(last.content)}")
        elif isinstance(last, HumanMessage):
            print(f"\n👤 USER:\n{content_to_text(last.content)}")

    return final_state


_APP_CACHE = {}


def get_app(product_brief: str):
    brief_hash = hashlib.sha256(product_brief.encode("utf-8")).hexdigest()[:12]

    if brief_hash not in _APP_CACHE:
        _APP_CACHE[brief_hash] = build_app(product_brief)

    return _APP_CACHE[brief_hash]


if __name__ == "__main__":
    thread_id = "demo-thread-1"

    app = get_app(PRODUCT_BRIEF)

    run_conversation(
        app,
        "Egy egyszerű dropshipping termék landing + checkout flow-t szeretnék. Hogyan kezdjünk neki?",
        thread_id,
        is_new_thread=True,
    )

    # második user üzenet ugyanabba a beszélgetésbe
    run_conversation(
        app,
        "Oké. Legyen Stripe test. Milyen adatokat kérjünk be checkoutnál?",
        thread_id,
        is_new_thread=False,
    )
