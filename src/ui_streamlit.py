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
    # 1) user message mentése + megjelenítés
    st.session_state.history.append({"speaker": "USER", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    placeholders = {}  # node_name -> st.empty()
    last_text_by_node = {}  # node_name -> last text (persisthez)

    initial = {
        "messages": [{"role": "user", "content": prompt}],
        "turn": 0,  # MVP: mindig 0; thread_id a checkpointerben viszi a folytonosságot
    }

    for update in app.stream(
        initial,
        config={"configurable": {"thread_id": st.session_state.thread_id}},
        stream_mode="updates",
    ):
        for node_name, partial_state in update.items():
            msgs = partial_state.get("messages", [])
            if not msgs:
                continue

            last = msgs[-1]
            if not isinstance(last, AIMessage):
                continue

            speaker = node_name.upper()
            text = content_to_text(last.content)

            # live bubble
            if node_name not in placeholders:
                with st.chat_message("assistant"):
                    placeholders[node_name] = st.empty()

            placeholders[node_name].markdown(f"**{speaker}**\n\n{text}")

            # ✅ ez a lényeg: elmentjük a legutolsó szöveget node-onként
            last_text_by_node[node_name] = text

    # 3) ✅ stream végén persist a history-ba (fix sorrendben)
    order = ["customer", "fe", "ba"]
    for node in order:
        if node in last_text_by_node:
            st.session_state.history.append(
                {"speaker": node.upper(), "text": last_text_by_node[node]}
            )

    st.rerun()
