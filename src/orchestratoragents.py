from langchain_openai import ChatOpenAI
from langchain.agents import create_agent


SYSTEM_PROMPT_ROLE_SELECTOR = """
You are a role selector agent.
Choose the exact role based on the last message from the user.
The roles are:
- frontenddeveloper
- businessanalyst
- customer
- other

Return the role in the following format:
{role}
"""

roleSelector = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    use_responses_api=True,
)

roleSelectorAgent = create_agent(
    model=roleSelector,
    system_prompt=SYSTEM_PROMPT_ROLE_SELECTOR,
)
