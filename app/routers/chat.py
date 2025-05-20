from typing import List, Optional
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from app.agent.core import route_to_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage

from app.utils import get_enterprise_details


chat_router = APIRouter(prefix="/conversation")

templates = Jinja2Templates(directory="app/static")


class Message(BaseModel):
    type: str  # "ai" or "human"
    content: str


class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[List[Message]] = None


class ChatResponse(BaseModel):
    response: str
    chat_history: List[Message]


@chat_router.post(
    "/ask",
    response_model=ChatResponse,
    tags=["chat"],
    summary="Ask a question to the chatbot",
)
async def chat(request: ChatRequest):
    # Initialize memory for this request
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        ai_prefix="ai",
        human_prefix="human",
    )

    # Load chat history from request, if provided
    chat_history = []
    if request.chat_history:
        for msg in request.chat_history:
            if msg.type == "human":
                chat_history.append(HumanMessage(content=msg.content))
                memory.save_context({"input": msg.content}, {"output": ""})
            elif msg.type == "ai":
                chat_history.append(AIMessage(content=msg.content))
                memory.save_context({"input": ""}, {"output": msg.content})

    # Use the agent router to direct to the appropriate specialized agent
    response = route_to_agent(request.query, chat_history)

    # Get updated chat history
    history = []
    for msg in memory.load_memory_variables({})["chat_history"]:
        if isinstance(msg, HumanMessage):
            history.append({"type": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            history.append({"type": "ai", "content": msg.content})

    # Add the latest interaction
    history.append({"type": "human", "content": request.query})
    history.append({"type": "ai", "content": response["output"]})

    return ChatResponse(response=response["output"], chat_history=history)


@chat_router.get("/app", tags=["chat"], response_class=HTMLResponse)
async def get_app(request: Request):
    """
    This endpoint is used to test the server.
    """
    return templates.TemplateResponse("views/index.html", {"request": request})
