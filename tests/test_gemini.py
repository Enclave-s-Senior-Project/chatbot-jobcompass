from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
llm = ChatGoogleGenerativeAI(model="learnlm-2.0-flash-experimental", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.1)
response = llm.invoke("What can you do for me?")
print(response.content)