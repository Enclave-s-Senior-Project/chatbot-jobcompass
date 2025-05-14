from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from app.agent.core import agent
from typing import List, Optional
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

app = FastAPI()


class Message(BaseModel):
    type: str  # "ai" or "human"
    content: str


class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[List[Message]] = None


class ChatResponse(BaseModel):
    response: str
    chat_history: List[Message]


@app.post("/chat")
async def chat(request: ChatRequest):
    # Initialize memory for this request
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Load chat history from request, if provided
    if request.chat_history:
        for msg in request.chat_history:
            if msg.type == "human":
                memory.save_context({"input": msg.content}, {"output": ""})
            elif msg.type == "ai":
                memory.save_context({"input": ""}, {"output": msg.content})

    # Run agent
    agent.memory = memory
    response = agent.run(request.query)

    # Get updated chat history
    history = []
    for msg in memory.load_memory_variables({})["chat_history"]:
        if isinstance(msg, HumanMessage):
            history.append({"type": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            history.append({"type": "ai", "content": msg.content})

    return ChatResponse(response=response, chat_history=history)


# This route is used to test the server
@app.get("/test")
async def get_test():
    """Handle chat requests by processing user queries against the website database.

        Args:
            request (ChatRequest): The chat request containing the user's query
    `
        Returns:
            dict: Response containing the agent's answer

        Raises:
            HTTPException: If there's an error processing the request
    """
    return {"response": "Hello, world!"}
