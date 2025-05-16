from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from app.tools import website_tool, db_tool, job_tool, enterprise_tool
from app.llm import llm
from .prompt import (
    agent_prompt,
    job_search_prompt,
    website_content_prompt,
    enterprise_search_prompt,
)
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage


class JobSearch(BaseModel):
    """Search for jobs based on different criteria."""

    query: str = Field(..., description="Search query for job search")


# Create specialized agents
job_search_agent = create_openai_functions_agent(llm, [job_tool], job_search_prompt)
job_search_executor = AgentExecutor(
    agent=job_search_agent, tools=[job_tool], verbose=True
)

enterprise_search_agent = create_openai_functions_agent(
    llm, [enterprise_tool], enterprise_search_prompt
)
enterprise_search_executor = AgentExecutor(
    agent=enterprise_search_agent, tools=[enterprise_tool], verbose=True
)

website_content_agent = create_openai_functions_agent(
    llm, [website_tool], website_content_prompt
)
website_content_executor = AgentExecutor(
    agent=website_content_agent, tools=[website_tool], verbose=True
)

# General agent for fallback or uncertain cases
tools = [website_tool, job_tool, db_tool]
agent = create_openai_functions_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# Router function to direct queries to the appropriate specialized agent
def route_to_agent(query: str, chat_history: List = None) -> Dict[str, Any]:
    """
    Routes the query to the appropriate specialized agent based on the content.

    Args:
        query: The user's query string
        chat_history: Optional chat history

    Returns:
        The response from the appropriate agent
    """
    # Use a classification approach instead of simple keyword matching
    # Here we'll use the LLM to determine the most appropriate agent
    prompt = f"""
    Analyze this user query: "{query}"
    
    Based on the query content, determine which category it best fits into:
    1. "job_search" - for queries about finding jobs, job descriptions, requirements, salary information, etc.
    2. "enterprise_search" - for queries about companies, organizations, employers, etc.
    3. "website_content" - for queries about the website itself, help pages, terms, etc.
    4. "general" - for queries that don't clearly fit the above categories or span multiple categories
    
    Return only one of these four options, no explanation: job_search, enterprise_search, website_content, or general
    """

    # Get classification from LLM
    classification_response = llm.invoke(prompt)
    classification = classification_response.content.strip().lower()

    chat_history = chat_history or []

    # Route based on classification
    if "job_search" in classification:
        return job_search_executor.invoke(
            {"input": query, "chat_history": chat_history}
        )
    elif "enterprise_search" in classification:
        return enterprise_search_executor.invoke(
            {"input": query, "chat_history": chat_history}
        )
    elif "website_content" in classification:
        return website_content_executor.invoke(
            {"input": query, "chat_history": chat_history}
        )
    else:
        # Default to the general agent for uncertain cases
        return agent_executor.invoke({"input": query, "chat_history": chat_history})
