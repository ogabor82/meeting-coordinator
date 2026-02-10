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

if "seen_ids" not in st.session_state:
    st.session_state.seen_ids = set()


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
    st.session_state.seen_ids = set()

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

    initial = {
        "messages": [{"role": "user", "content": prompt}],
        "turn": 0,
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

            # 2/a) live bubble frissítés node-onként
            if node_name not in placeholders:
                with st.chat_message("assistant"):
                    placeholders[node_name] = st.empty()
            placeholders[node_name].markdown(f"**{speaker}**\n\n{text}")

            # 2/b) ✅ transcript mentése dupe nélkül
            mid = getattr(last, "id", None)
            key = mid or f"{node_name}:{hash(text)}"  # fallback, ha nincs id

            if key not in st.session_state.seen_ids:
                st.session_state.seen_ids.add(key)
                st.session_state.history.append({"speaker": speaker, "text": text})

    st.rerun()
