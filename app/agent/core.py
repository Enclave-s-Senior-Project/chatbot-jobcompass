from app.tools import website_tool, db_tool
from langchain.agents import initialize_agent, AgentType
from app.llm import llm

# Agent
tools = [website_tool, db_tool]
agent = initialize_agent(
    tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)  # A zero shot agent that does a reasoning step before acting. verbose=True to see the agent's thought process (logging).
