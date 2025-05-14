from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from constants import vector_database_url


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

website_content_vector_store = PGVector(
    collection_name="website_content",
    connection=vector_database_url,
    embeddings=embeddings,
)

job_vector_store = PGVector(
    collection_name="job_listings",
    connection=vector_database_url,
    embeddings=embeddings,
)
