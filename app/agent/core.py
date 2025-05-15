from langchain.agents import AgentExecutor, create_openai_functions_agent
from app.tools import website_tool, db_tool, job_tool
from app.llm import llm
from .prompt import (
    agent_prompt,
    job_search_prompt,
    website_content_prompt,
    enterprise_search_prompt,
)
from pydantic import BaseModel, Field
from typing import List, Dict, Any


class JobSearch(BaseModel):
    """Search for jobs based on different criteria."""

    query: str = Field(..., description="Search query for job search")


# Create specialized agents
job_search_agent = create_openai_functions_agent(llm, [job_tool], job_search_prompt)
job_search_executor = AgentExecutor(
    agent=job_search_agent, tools=[job_tool], verbose=True
)

enterprise_search_agent = create_openai_functions_agent(
    llm, [db_tool], enterprise_search_prompt
)
enterprise_search_executor = AgentExecutor(
    agent=enterprise_search_agent, tools=[db_tool], verbose=True
)

website_content_agent = create_openai_functions_agent(
    llm, [website_tool], website_content_prompt
)
website_content_executor = AgentExecutor(
    agent=website_content_agent, tools=[website_tool], verbose=True
)

# Keep the original general agent for backward compatibility
tools = [website_tool, job_tool, db_tool]
agent = create_openai_functions_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# Agent router function to direct queries to the appropriate specialized agent
def route_to_agent(query: str, chat_history: List = None) -> Dict[str, Any]:
    """
    Routes the query to the appropriate specialized agent based on the content.

    Args:
        query: The user's query string
        chat_history: Optional chat history

    Returns:
        The response from the appropriate agent
    """
    # Simple keyword-based routing
    query_lower = query.lower()

    chat_history = chat_history or []

    # Route to job search agent
    if any(
        keyword in query_lower
        for keyword in [
            "job",
            "career",
            "position",
            "salary",
            "apply",
            "hiring",
            "developer",
            "engineer",
        ]
    ):
        return job_search_executor.invoke(
            {"input": query, "chat_history": chat_history}
        )

    # Route to enterprise search agent
    elif any(
        keyword in query_lower
        for keyword in [
            "company",
            "enterprise",
            "organization",
            "firm",
            "employer",
            "business",
        ]
    ):
        return enterprise_search_executor.invoke(
            {"input": query, "chat_history": chat_history}
        )

    # Route to website content agent
    elif any(
        keyword in query_lower
        for keyword in [
            "website",
            "page",
            "about",
            "contact",
            "help",
            "support",
            "terms",
            "policy",
        ]
    ):
        return website_content_executor.invoke(
            {"input": query, "chat_history": chat_history}
        )

    # Default to the general agent for other queries
    else:
        return agent_executor.invoke({"input": query, "chat_history": chat_history})
