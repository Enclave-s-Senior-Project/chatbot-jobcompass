# index.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from constants import database_url
import json
from dotenv import load_dotenv

load_dotenv()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = PGVector(
    collection_name="website_content",
    connection=database_url,
    embeddings=embeddings
)

with open("website_chunks.json", "r") as f:
    website_chunks = json.load(f)

texts = [chunk["content"] for chunk in website_chunks]
metadata = [{"url": chunk["url"]} for chunk in website_chunks]

# Add texts with their embeddings
vector_store.add_texts(
    texts=texts,
    metadatas=metadata
)