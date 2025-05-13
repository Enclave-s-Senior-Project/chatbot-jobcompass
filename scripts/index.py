# index.py
from langchain_huggingface import HuggingFaceEmbeddings
import json
from dotenv import load_dotenv
from app.vectorstore import vector_store

load_dotenv()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


with open("website_chunks.json", "r") as f:
    website_chunks = json.load(f)

texts = [chunk["content"] for chunk in website_chunks]
metadata = [{"url": chunk["url"]} for chunk in website_chunks]

# Add texts with their embeddings
vector_store.add_texts(
    texts=texts,
    metadatas=metadata
)