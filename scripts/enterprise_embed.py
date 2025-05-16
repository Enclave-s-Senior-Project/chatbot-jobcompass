# Database connection
from contextlib import contextmanager
from langchain_core.documents import Document
from constants import main_database_url
from app.vectorstore import enterprise_vector_store

import psycopg2


@contextmanager
def get_db_connection():
    conn = psycopg2.connect(main_database_url)
    try:
        yield conn
    finally:
        conn.close()


def fetch_enterprises() -> list:
    """Fetch enterprise data from the database."""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                en.enterprise_id,
                en.name, 
                en.description,
                en.company_vision,
                en.logo_url, 
                en.founded_in,
                en.organization_type, 
                en.team_size, 
                en.status, 
                en.is_premium,
                en.is_trial,
                JSON_AGG(JSONB_BUILD_OBJECT(
                    'category_id', ca.category_id,
                    'category_name', ca.category_name
                )) as enterprise_categories,
                JSON_AGG(JSONB_BUILD_OBJECT(
                    'address_id', addr.address_id,
                    'mixed_address', addr.mixed_address
                )) as enterprise_addresses
            FROM enterprises en
            LEFT JOIN enterprise_addresses enaddr ON enaddr.enterprise_id = en.enterprise_id
            LEFT JOIN addresses addr ON addr.address_id = enaddr.address_id
            LEFT JOIN websites we ON we.enterprise_id = en.enterprise_id
            LEFT JOIN categories ca ON text(ca.category_id) = ANY(en.categories)
            WHERE en.status = 'ACTIVE'
            GROUP BY en.enterprise_id
            """
            )
            return cursor.fetchall()


def create_enterprise_document(enterprise_data: tuple) -> Document:
    """Create a well-structured document for enterprise embedding."""
    # Unpack enterprise data
    (
        enterprise_id,
        name,
        description,
        company_vision,
        logo_url,
        founded_in,
        organization_type,
        team_size,
        status,
        is_premium,
        is_trial,
        enterprise_categories,
        enterprise_addresses,
    ) = enterprise_data

    # Parse enterprise categories from queries
    categories = (
        [
            {"category_id": cat["category_id"], "category_name": cat["category_name"]}
            for cat in enterprise_categories
        ]
        if isinstance(enterprise_categories, list)
        else []
    )

    addresses = (
        [
            {"address_id": addr["address_id"], "mixed_address": addr["mixed_address"]}
            for addr in enterprise_addresses
        ]
        if isinstance(enterprise_addresses, list)
        else []
    )

    # Create content for embedding
    content = f"""
        Company Name: {name}
        Company Description: {description}
        Company Vision: {company_vision}
        Founded In: {founded_in}
        Organization Type: {organization_type}
        Team Size: {team_size}
        Status: {status}
        Is Premium: {is_premium or is_trial}
        Categories: {", ".join([cat["category_name"] for cat in categories]) if categories else "Not specified"}
        Addresses: {"; ".join([addr["mixed_address"] for addr in addresses]) if addresses else "Not specified"}
    """

    # Create metadata
    metadata = {
        "enterprise_id": enterprise_id,
        "name": name,
        "description": description,
        "company_vision": company_vision,
        "logo_url": logo_url,
        "founded_in": str(founded_in) if founded_in else "",
        "organization_type": organization_type,
        "team_size": team_size,
        "status": status,
        "is_premium": is_premium or is_trial,
        "categories": categories,
        "addresses": addresses,
    }

    return Document(page_content=content, metadata=metadata)


if __name__ == "__main__":
    # Fetch enterprise data
    print("Fetching enterprise data from the database...")
    enterprises = fetch_enterprises()

    # Create documents for each enterprise
    print("Creating documents for enterprise data...")
    documents = [create_enterprise_document(enterprise) for enterprise in enterprises]

    # Add embeddings to vector store
    print("Adding embeddings to vector store...")
    enterprise_vector_store.add_documents(documents)

    print("Enterprise embeddings added successfully.")
