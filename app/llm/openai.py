from langchain_openai import ChatOpenAI
import os


# LLM (GPT-4o-mini)
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.5,
    max_retries=3,
)
