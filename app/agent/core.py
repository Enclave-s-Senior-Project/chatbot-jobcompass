from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from app.tools import website_tool, db_tool, job_tool, enterprise_tool
from app.llm import llm
from app.utils import clean_html
from app.utils.api_client import get_enterprise_details, get_profile_details
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
job_search_agent = create_openai_functions_agent(
    llm, [job_tool, db_tool], job_search_prompt
)
job_search_executor = AgentExecutor(
    agent=job_search_agent, tools=[job_tool, db_tool], verbose=True
)

enterprise_search_agent = create_openai_functions_agent(
    llm, [enterprise_tool, db_tool], enterprise_search_prompt
)
enterprise_search_executor = AgentExecutor(
    agent=enterprise_search_agent, tools=[enterprise_tool, db_tool], verbose=True
)

website_content_agent = create_openai_functions_agent(
    llm, [website_tool, db_tool], website_content_prompt
)
website_content_executor = AgentExecutor(
    agent=website_content_agent, tools=[website_tool, db_tool], verbose=True
)

# General agent for fallback or uncertain cases
tools = [website_tool, job_tool, db_tool]
agent = create_openai_functions_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


def summarize_profile_info(profileId: Optional[str] = None) -> str:
    """
    Summarizes the additional content to be used in the chat.
    """
    additional_content = ""
    if profileId:
        profile_details = get_profile_details(profileId)
        if profile_details:
            roles = profile_details.get("roles", [])
            additional_content = f"""
[User Info]: Name: {profile_details.get('fullName', 'N/A')}
Email: {profile_details.get('email', 'N/A')}
Gender: {profile_details.get('gender', 'N/A')}
Nationality: {profile_details.get('nationality', 'N/A')} (Prefer job/enterprise in this country)
Date of Birth: {profile_details.get('dateOfBirth', 'N/A')}
Marital Status: {profile_details.get('maritalStatus', 'N/A')}
Education: {clean_html(profile_details.get('education', 'N/A'))}
Experience: {clean_html(profile_details.get('experience', 'N/A'))}
Roles: {", ".join(roles)}
Industry [Working Field]: {profile_details.get('industry', {}).get('categoryName', 'N/A') if profile_details.get('industry', {}) else 'N/A'} (Refer job/enterprise in this field)
Major: {profile_details.get('majority', {}).get('categoryName', 'N/A') if profile_details.get('majority', {}) else 'N/A'} (Refer job/enterprise in this major)\n
"""

    return additional_content


def summarize_enterprise_info(enterpriseId: Optional[str] = None) -> str:
    additional_content = ""
    if enterpriseId:
        enterprise_details = get_enterprise_details(enterpriseId)
        if enterprise_details:
            industries = enterprise_details.get("categories", [])
            addresses = enterprise_details.get("addresses", [])
            additional_content = f"""
[Enterprise Info]: Name: {enterprise_details.get('name', 'N/A')}
Email: {enterprise_details.get('email', 'N/A')}
Phone: {enterprise_details.get('phone', 'N/A')}
Description: {clean_html(enterprise_details.get('description', 'N/A'))}
Benefits: {clean_html(enterprise_details.get('benefit', 'N/A'))}
Founded In: {enterprise_details.get('foundedIn', 'N/A')}
Team Size: {enterprise_details.get('teamSize', 'N/A')}
Organization Type: {enterprise_details.get('organizationType', 'N/A')}
Industries [Working Fields]: {"; ".join([i.get('categoryName') for i in industries]) if len(industries) > 0 else 'N/A'}
Addresses: {", ".join([a.get('mixedAddress') for a in addresses]) if len(addresses) > 0 else 'N/A'}
"""

    return additional_content


# Router function to direct queries to the appropriate specialized agent
def route_to_agent(
    query: str,
    chat_history: List = None,
    profileId: Optional[str] = None,
    enterpriseId: Optional[str] = None,
) -> Dict[str, Any]:
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
        profile_content = summarize_profile_info(profileId=profileId)
        full_query = (
            query
            + "\nYou have to follow these additional information for best response"
            + profile_content
            if profile_content
            else query
        )
        return job_search_executor.invoke(
            {"input": full_query, "chat_history": chat_history}
        )
    elif "enterprise_search" in classification:
        profile_content = summarize_profile_info(profileId=profileId)
        full_query = (
            query
            + "\n[ADDITIONAL INFO] Use these information for best response"
            + profile_content
            if profile_content
            else query
        )
        return enterprise_search_executor.invoke(
            {"input": full_query, "chat_history": chat_history}
        )
    elif "website_content" in classification:
        return website_content_executor.invoke(
            {"input": query, "chat_history": chat_history}
        )
    else:
        # Default to the general agent for uncertain cases
        return agent_executor.invoke({"input": query, "chat_history": chat_history})
