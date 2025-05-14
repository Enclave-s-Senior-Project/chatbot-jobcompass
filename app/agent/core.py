from app.tools import website_tool, db_tool, job_tool
from langchain.agents import initialize_agent, AgentType
from app.llm import llm
from .prompt import agent_prompt

# Agent
tools = [website_tool, job_tool, db_tool]
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={"prompt": agent_prompt},
)  # A zero shot agent that does a reasoning step before acting. verbose=True to see the agent's thought process (logging).
