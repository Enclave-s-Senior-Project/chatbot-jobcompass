from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from app.agent.core import agent

load_dotenv()

app = FastAPI()


class ChatRequest(BaseModel):
    query: str


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = agent.run(
            f"Answer based on website or database: {request.query}. If no data, ask more information"
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
