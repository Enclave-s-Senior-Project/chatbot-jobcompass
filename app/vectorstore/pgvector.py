from langchain_postgres import PGVector
from constants import vector_database_url
from app.llm import embeddings_model

website_content_vector_store = PGVector(
    collection_name="website_content",
    connection=vector_database_url,
    embeddings=embeddings_model,
    use_jsonb=True,
)

job_vector_store = PGVector(
    collection_name="job_listings",
    connection=vector_database_url,
    embeddings=embeddings_model,
    use_jsonb=True,
)

enterprise_vector_store = PGVector(
    collection_name="enterprise_listings",
    connection=vector_database_url,
    embeddings=embeddings_model,
    use_jsonb=True,
)
