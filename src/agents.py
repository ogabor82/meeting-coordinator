from langchain_openai import ChatOpenAI
from langchain.agents import create_agent


SYSTEM_PROMPT = """
You are a frontend software developer. 
Anwser the user's question in a friendly and helpful manner.

"""


frontendDeveloper = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0,
    use_responses_api=True,
)

frontendDeveloperLocal = ChatOpenAI(
    model="local-model",
    temperature=0,
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    use_responses_api=False,
)


frontendDeveloperAgent = create_agent(
    model=frontendDeveloper,
    system_prompt=SYSTEM_PROMPT,
)

frontendDeveloperAgentLocal = create_agent(
    model=frontendDeveloperLocal,
    system_prompt=SYSTEM_PROMPT,
)
