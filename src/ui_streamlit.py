import streamlit as st
from uuid import uuid4
import hashlib

from langchain_core.messages import AIMessage

from conversationBuilder import build_app, content_to_text


def brief_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


@st.cache_resource
def get_cached_app(product_brief: str):
    return build_app(product_brief)


st.set_page_config(page_title="Meeting Coordinator MVP", layout="wide")
st.title("🧩 Meeting Coordinator — MVP")

# --- Session state init ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())

if "history" not in st.session_state:
    # history: {"speaker": "USER/FE/BA/CUSTOMER", "text": "..."}
    st.session_state.history = []

if "last_brief_hash" not in st.session_state:
    st.session_state.last_brief_hash = None

# --- Sidebar: product brief ---
st.sidebar.header("Product brief")
default_brief = """Goal: single-product landing page + checkout flow.
MVP features:
- Landing page: benefits, FAQ, reviews (mock), price
- Checkout: shipping info + payment (can be Stripe test)
- Order confirmation page + email receipt (can be mocked)
Constraints:
- Must be mobile-first and fast
Non-goals:
- Full catalog, user accounts (later)
"""

product_brief = st.sidebar.text_area("PRODUCT_BRIEF", value=default_brief, height=260)

# reset history
current_hash = brief_hash(product_brief)
if st.session_state.last_brief_hash != current_hash:
    st.session_state.last_brief_hash = current_hash
    st.session_state.thread_id = str(uuid4())
    st.session_state.history = []

app = get_cached_app(product_brief)

# --- Render history ---
for item in st.session_state.history:
    role = "user" if item["speaker"] == "USER" else "assistant"
    with st.chat_message(role):
        st.markdown(f"**{item['speaker']}**\n\n{item['text']}")

# --- Chat input ---
prompt = st.chat_input("Írd ide a user üzenetet…")
if prompt:
    # 1) show user message
    st.session_state.history.append({"speaker": "USER", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2) stream graph updates, node based on who is speaking
    placeholders = {}  # node_name -> st.empty() (live updating)

    initial = {
        "messages": [{"role": "user", "content": prompt}],
        "turn": 0,  # new turn=0; thread_id handles the continuation
    }

    for update in app.stream(
        initial,
        config={"configurable": {"thread_id": st.session_state.thread_id}},
        stream_mode="updates",
    ):
        # update: {"customer": {"messages": [...], "turn": 1}}
        for node_name, partial_state in update.items():
            msgs = partial_state.get("messages", [])
            if not msgs:
                continue

            last = msgs[-1]
            if not isinstance(last, AIMessage):
                continue

            speaker = node_name.upper()
            text = content_to_text(last.content)

            # 2/a) create the bubble for the first time
            if node_name not in placeholders:
                with st.chat_message("assistant"):
                    placeholders[node_name] = st.empty()

            # 2/b) live update in the same bubble
            placeholders[node_name].markdown(f"**{speaker}**\n\n{text}")

    # 3) at the end of the stream, save the last displayed text to history
    for node_name, ph in placeholders.items():
        # unfortunately we can't read the text from the placeholder,
        # so in MVP we only save as much as you just saw:
        # -> simple solution: run again with stream_mode="values" and extract the final state
        pass

    st.rerun()
