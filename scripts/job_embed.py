from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from app.vectorstore import job_vector_store
from constants import main_database_url
from contextlib import contextmanager
from bs4 import BeautifulSoup
import psycopg2
import json


# Database connection
@contextmanager
def get_db_connection():
    conn = psycopg2.connect(main_database_url)
    try:
        yield conn
    finally:
        conn.close()


# Embedding model
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Clean HTML and summarize text
def clean_html(text):
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def summarize_requirements(text):
    if not text:
        return ""
    lines = text.split(". ")
    key_phrases = []
    for line in lines[:10]:  # Limit to first 10 sentences
        if any(
            keyword in line.lower()
            for keyword in ["degree", "experience", "certification", "fluency", "years"]
        ):
            key_phrases.append(line.strip())
    return ". ".join(key_phrases[:5])[:300]  # Limit to 5 phrases, 300 chars


# Fetch jobs with related data
def fetch_jobs():
    print("Fetching jobs...")
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """ 
                SELECT
                    jb.job_id, 
                    jb.name as job_name, 
                    jb.type as job_type, 
                    jb.deadline, 
                    jb.education, 
                    jb.experience,
                    jb.highest_wage,
                    jb.lowest_wage,
                    jb.status as job_status,
                    jb.requirement,
                    en.enterprise_id,
                    en.name as enterprise_name,
                    en.is_premium,
                    en.is_trial,
                    en.organization_type,
                    en.status as enterprise_status,
                    JSON_AGG(DISTINCT JSONB_BUILD_OBJECT(
                        'id', ct1.category_id,
                        'name', ct1.category_name
                    )) FILTER (WHERE ct1.category_id IS NOT NULL) AS job_categories,
                    JSON_AGG(DISTINCT JSONB_BUILD_OBJECT(
                        'id', ct2.category_id,
                        'name', ct2.category_name
                    )) FILTER (WHERE ct2.category_id IS NOT NULL) AS job_specializations,
                    JSON_AGG(DISTINCT JSONB_BUILD_OBJECT(
                        'id', tg.tag_id,
                        'name', tg."name",
                        'color', tg.color,
                        'background_color', tg.background_color
                    )) FILTER (WHERE tg.tag_id IS NOT NULL) AS job_tags,
                    JSON_AGG(DISTINCT JSONB_BUILD_OBJECT(
                        'id', ad.address_id,
                        'country', ad.country,
                        'city', ad.city,
                        'street', ad.street,
                        'zip_code', ad.zip_code
                    )) FILTER (WHERE ad.address_id IS NOT NULL) AS job_addresses
                FROM jobs jb
                LEFT JOIN enterprises en ON jb.enterprise_id = en.enterprise_id
                LEFT JOIN job_categories jbc ON jbc.job_id = jb.job_id
                LEFT JOIN categories ct1 ON ct1.category_id = jbc.category_id
                LEFT JOIN job_specializations jbs ON jbs.job_id = jb.job_id
                LEFT JOIN categories ct2 ON ct2.category_id = jbs.category_id
                LEFT JOIN job_tags jbt ON jbt.job_id = jb.job_id
                LEFT JOIN tags tg ON tg.tag_id = jbt.tag_id
                LEFT JOIN job_addresses jba ON jba.job_id = jb.job_id
                LEFT JOIN addresses ad ON ad.address_id = jba.address_id
                WHERE jb.is_active = TRUE
                GROUP BY 
                    jb.job_id, 
                    jb.name, 
                    jb.type, 
                    jb.deadline, 
                    jb.education, 
                    jb.experience,
                    jb.highest_wage,
                    jb.lowest_wage,
                    jb.status,
                    jb.requirement,
                    en.enterprise_id,
                    en.name,
                    en.is_premium,
                    en.is_trial,
                    en.organization_type,
                    en.status
                """
            )
            results = cursor.fetchall()
            return results


# Create documents for embedding
jobs = fetch_jobs()
documents = []
print("Creating documents...")
for job in jobs:
    (
        job_id,
        job_name,
        job_type,
        deadline,
        education,
        experience,
        highest_wage,
        lowest_wage,
        job_status,
        requirement,
        enterprise_id,
        enterprise_name,
        is_premium,
        is_trial,
        organization_type,
        enterprise_status,
        job_categories,
        job_specializations,
        job_tags,
        job_addresses,
    ) = job

    # Clean and summarize fields
    clean_requirement = summarize_requirements(clean_html(requirement))

    # Extract category, tag, specialization names
    categories = [cat["name"] for cat in job_categories] if job_categories else []
    tags = [tag["name"] for tag in job_tags] if job_tags else []
    specializations = (
        [spec["name"] for spec in job_specializations] if job_specializations else []
    )

    # Combine fields for embedding
    content = (
        f"Job Title: {job_name}\n"
        f"Type: {job_type}\n"
        f"Company: {enterprise_name}\n"
        f"Requirements: {clean_requirement}\n"
    )
    if categories:
        content += f"Categories: {', '.join(categories)}\n"
    if tags:
        content += f"Tags: {', '.join(tags)}\n"
    if specializations:
        content += f"Specializations: {', '.join(specializations)}\n"

    # Metadata
    city = job_addresses[0]["city"] if job_addresses and len(job_addresses) > 0 else ""
    country = (
        job_addresses[0]["country"] if job_addresses and len(job_addresses) > 0 else ""
    )
    metadata = {
        "job_id": str(job_id),
        "company": enterprise_name or "",
        "city": city,
        "country": country,
        "experience": experience or 0,
        "education": education or "",
        "status": job_status or "",
        "lowest_wage": lowest_wage or 0,
        "highest_wage": highest_wage or 0,
        "categories": job_categories or [],
        "tags": job_tags or [],
        "specializations": job_specializations or [],
        "deadline": str(deadline) if deadline else "",
        "is_premium": is_premium or False,
        "is_trial": is_trial or False,
        "organization_type": organization_type or "",
        "enterprise_status": enterprise_status or "",
    }

    documents.append(Document(page_content=content, metadata=metadata))

# Add documents to vector store
print("Adding documents to vector store...")
# job_vector_store.delete_collection()  # Optional: clear existing collection
job_vector_store.add_documents(documents)

# Create IVFFlat index
with get_db_connection() as conn:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_job_listings_embedding
            ON langchain_pg_embedding
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 4)
            WHERE collection_id = (
                SELECT id FROM langchain_pg_collection WHERE name = 'job_listings'
            );
            """
        )
        conn.commit()

print(f"Indexed {len(documents)} jobs into pgvector collection 'job_listings'.")
