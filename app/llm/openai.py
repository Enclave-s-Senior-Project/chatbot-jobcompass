from langchain_openai import ChatOpenAI
import os


# LLM (GPT-4o-mini)
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.3,
    max_retries=5,
)
