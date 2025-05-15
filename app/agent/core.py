from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from app.tools import website_tool, db_tool, job_tool
from app.llm import llm
from .prompt import agent_prompt

# Define a structured job search function
from pydantic import BaseModel, Field
from typing import List, Optional


class JobSearch(BaseModel):
    """Search for jobs based on different criteria."""

    query: str = Field(..., description="Search query for job search")


# Define tools
tools = [website_tool, job_tool, db_tool]

# Create the function-calling agent using the imported agent_prompt
agent = create_openai_functions_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
