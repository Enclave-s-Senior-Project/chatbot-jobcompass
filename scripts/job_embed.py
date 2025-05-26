from langchain_core.documents import Document
from app.services.preprocess import preprocess_text
from app.vectorstore import job_vector_store
from constants import main_database_url
from contextlib import contextmanager
from app.utils import clean_html
import psycopg2


# Database connection
@contextmanager
def get_db_connection():
    conn = psycopg2.connect(main_database_url)
    try:
        yield conn
    finally:
        conn.close()


def create_job_document(job_data: tuple) -> Document:
    """Create a well-structured document for job embedding."""
    # Unpack job data
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
        description,
        responsibility,
        job_benefits,
        enterprise_name,
        organization_type,
        is_premium,
        is_trial,
        enterprise_status,
        points_used,
        job_categories,
        job_specializations,
        job_tags,
        job_addresses,
    ) = job_data

    # Extract and format metadata
    # Handle parse categories from queries
    try:
        categories = (
            [cat["name"] for cat in job_categories]
            if job_categories and isinstance(job_categories, list)
            else []
        )
    except (TypeError, AttributeError):
        categories = []

    # Handle parse tags from queries
    try:
        tags = (
            [tag["name"] for tag in job_tags]
            if job_tags and isinstance(job_tags, list)
            else []
        )
    except (TypeError, AttributeError):
        tags = []

    # Handle parse specializations from queries
    try:
        specializations = (
            [spec["name"] for spec in job_specializations]
            if job_specializations and isinstance(job_specializations, list)
            else []
        )
    except (TypeError, AttributeError):
        specializations = []

    # Handle parse addresses from queries
    locations = []
    if job_addresses and isinstance(job_addresses, list) and len(job_addresses) > 0:
        try:
            for address in job_addresses:
                city = address.get("city", "")
                country = address.get("country", "")

                location = ""
                if city and country:
                    location = f"In {country}, {city}"
                elif city:
                    location = "In " + city
                elif country:
                    location = "In " + country
                locations.append(location)
        except (TypeError, AttributeError):
            locations = []

    all_keywords = (
        [preprocess_text(job_name)]
        + [preprocess_text(cat) for cat in categories]
        + [preprocess_text(spec) for spec in specializations]
        + [preprocess_text(tag) for tag in tags]
    )
    all_keywords += [preprocess_text(requirement), preprocess_text(description)]
    all_keywords += [preprocess_text(loc) for loc in locations]
    keyword_blob = " ".join([w for w in all_keywords if w])

    # Repeat important fields for weighting
    repeated_title = (job_name + " ") * 5
    repeated_categories = (", ".join(categories) + " ") * 4 if categories else ""
    repeated_specializations = (
        (", ".join(specializations) + " ") * 4 if specializations else ""
    )
    repeated_tags = (", ".join(tags) + " ") * 3 if tags else ""
    repeated_location = (" ;".join(locations) + "; ") * 2 if locations else "Remote "
    # Compose a skills line if possible
    skills_from_tags = ", ".join(tags)
    skills_from_requirements = clean_html(requirement)
    skills_line = f"Skills: {skills_from_tags} {skills_from_requirements}".strip()
    # Compose the content
    content = f"""
    Priority Points: {points_used};
    Title: {repeated_title};
    Industries: {repeated_categories};
    Majorities/Major: {repeated_specializations};
    Related Keywords: {repeated_tags};
    Location: {repeated_location};
    {keyword_blob}
    Company: {enterprise_name}
    Type: {job_type}
    {skills_line}
    Requirements: {clean_html(requirement)}
    Experience: {experience} years
    Education: {education}
    Deadline: {deadline}
    Salary Range: {lowest_wage} - {highest_wage} (USD)
    Company Type: {organization_type}
    """

    # Enhanced metadata
    metadata = {
        "job_id": str(job_id),
        "job_name": job_name,
        "company": enterprise_name or "",
        "experience": experience or 0,
        "education": education or "",
        "status": job_status or "",
        "salary_range": {"min": lowest_wage or 0, "max": highest_wage or 0},
        "categories": categories,
        "tags": tags,
        "specializations": specializations,
        "deadline": str(deadline) if deadline else "",
        "is_premium": is_premium or False,
        "is_trial": is_trial or False,
        "organization_type": organization_type or "",
        "enterprise_status": enterprise_status or "",
        "locations": locations,
        "job_type": job_type or "",
        "description": description or "",
        "responsibility": responsibility or "",
        "requirement": requirement or "",
        "job_benefits": job_benefits or "",
        "points_used": points_used or 0,
    }

    return Document(page_content=content, metadata=metadata)


def fetch_jobs():
    """Fetch jobs with related data from database."""
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
                    jb.description,
                    jb.responsibility,
                    jb.enterprise_benefits as job_benefits,
                    en.name as enterprise_name,
                    en.organization_type,
                    en.is_premium,
                    en.is_trial,
                    en.status as enterprise_status,
                    COALESCE(bjb.points_used, 0) as points_used,
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
                LEFT JOIN boosted_jobs bjb ON bjb.job_id = jb.job_id
                WHERE jb.status = 'OPEN' and en.status = 'ACTIVE'
                GROUP BY 
                    jb.job_id, 
                    jb.name, 
                    en.name,
                    en.is_premium,
                    en.is_trial,
                    en.organization_type,
                    en.status,
                    bjb.points_used
                """
            )
            return cursor.fetchall()


def main():
    """Main function to process and embed jobs."""
    # Fetch jobs
    jobs = fetch_jobs()

    # Create documents
    print("Creating documents...")
    documents = [create_job_document(job) for job in jobs]

    # Add to vector store
    print("Adding documents to vector store...")
    job_vector_store.add_documents(
        documents, ids=[f"job-{doc.metadata['job_id']}" for doc in documents]
    )

    print(
        f"Successfully indexed {len(documents)} jobs into pgvector collection 'job_listings'."
    )


if __name__ == "__main__":
    main()
