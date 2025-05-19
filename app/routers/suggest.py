from typing import List, Optional
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from app.agent.core import agent_executor
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage


chat_router = APIRouter(prefix="/suggest")

@chat_router.

