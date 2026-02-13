from langchain_openai import ChatOpenAI
from langchain.agents import create_agent


SYSTEM_PROMPT_FRONTEND_DEVELOPER = """
You are a frontend software developer. 
Anwser the user's question in a friendly and helpful manner.

"""

SYSTEM_PROMPT_BUSINESS_ANALYST = """
You are a business analyst.
Anwser the user's question in a friendly and helpful manner.

"""


frontendDeveloper = ChatOpenAI(
    model="gpt-4.1-mini",
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
    system_prompt=SYSTEM_PROMPT_FRONTEND_DEVELOPER,
)

frontendDeveloperAgentLocal = create_agent(
    model=frontendDeveloperLocal,
    system_prompt=SYSTEM_PROMPT_FRONTEND_DEVELOPER,
)

businessAnalyst = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    use_responses_api=True,
)

businessAnalystAgent = create_agent(
    model=businessAnalyst,
    system_prompt=SYSTEM_PROMPT_BUSINESS_ANALYST,
)
