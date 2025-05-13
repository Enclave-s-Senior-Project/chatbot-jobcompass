from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.utilities import SQLDatabase
import os
from dotenv import load_dotenv
from constants import database_url

load_dotenv()

app = FastAPI()

# LLM (GPT-4o-mini)
llm = ChatOpenAI(
    model_name="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.5
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = PGVector(
    collection_name="website_content", connection=database_url, embeddings=embeddings
)

# Website search tool
website_tool = Tool(
    name="WebsiteSearch",
    func=lambda q: "\n".join(
        [doc.page_content for doc in vector_store.similarity_search(q, k=3)]
    ),
    description="Search website content for relevant information.",
)

# Database tool
db = SQLDatabase.from_uri(database_url)
db_tool = Tool(
    name="Database",
    func=lambda q: db.run(q),
    description="Query the database for information.",
)

# Agent
tools = [website_tool]
agent = initialize_agent(
    tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)  # A zero shot agent that does a reasoning step before acting. verbose=True to see the agent's thought process (logging).


class ChatRequest(BaseModel):
    query: str


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = agent.run(
            f"Answer based on website or database: {request.query}. If no data, say 'I couldn’t find that. Can you clarify?'"
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# This route is used to test the server
@app.get("/test")
async def get_test():
    """Handle chat requests by processing user queries against the website database.

    Args:
        request (ChatRequest): The chat request containing the user's query

    Returns:
        dict: Response containing the agent's answer

    Raises:
        HTTPException: If there's an error processing the request
    """
    return {"response": "Hello, world!"}
