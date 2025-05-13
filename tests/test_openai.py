# test_openai.py
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.1, base_url='https://api.llmapi.com/')
response = llm.invoke("What can you do for me?")
print(response.content)