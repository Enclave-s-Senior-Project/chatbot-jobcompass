from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from constants import database_url


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = PGVector(
    collection_name="website_content", connection=database_url, embeddings=embeddings
)
