from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from app.agent.core import agent, agent_executor
from typing import List, Optional, AsyncGenerator
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from fastapi.responses import StreamingResponse
import json

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

    async def generate_response() -> AsyncGenerator[str, None]:
        try:
            # Stream the response using agent_executor
            async for chunk in agent_executor.astream(
                {
                    "input": request.query,
                    "chat_history": memory.load_memory_variables({})["chat_history"],
                    "agent_scratchpad": "",
                }
            ):
                # Extract content from different types of responses
                content = None
                if hasattr(chunk, "content"):
                    content = chunk.content
                elif isinstance(chunk, dict):
                    if "output" in chunk:
                        content = chunk["output"]
                    elif "messages" in chunk and chunk["messages"]:
                        for msg in chunk["messages"]:
                            if hasattr(msg, "content") and msg.content:
                                content = msg.content
                                break

                if content:
                    response_data = {"response": content}
                    yield f"data: {json.dumps(response_data)}\n\n"
        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            error_message = f"Error: {str(e)}"
            response_data = {"response": error_message}
            yield f"data: {json.dumps(response_data)}\n\n"

    return StreamingResponse(generate_response(), media_type="text/event-stream")


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
