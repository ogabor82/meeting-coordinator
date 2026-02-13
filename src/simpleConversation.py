import sys
from pathlib import Path
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# Allow running from project root (python src/simpleConversation.py) or from src/
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from langchain.messages import HumanMessage  # noqa: E402
from src.agents import frontendDeveloperAgent  # noqa: E402
# from src.agents import businessAnalystAgent  # noqa: E402


memory = InMemorySaver()
question = ""

while question != "exit":
    question = HumanMessage(content=input("Enter a question: "))
    config = {"configurable": {"thread_id": "1"}, "checkpoint_saver": memory}

    for chunk in frontendDeveloperAgent.stream(question, config):
        # print(chunk)
        print(chunk.get("model").get("messages")[-1].text)
