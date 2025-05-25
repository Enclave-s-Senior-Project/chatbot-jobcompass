import csv
import sys
import os

from app.services.preprocess import preprocess_text

# Add the parent directory to the path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.documents import Document
from app.vectorstore import website_content_vector_store


def load_website_content_from_csv():
    """Load website content from CSV file."""
    website_content = []

    # Get the correct path to the CSV file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "..", "app", "data", "website_content.csv")
    csv_path = os.path.normpath(csv_path)

    try:
        with open(csv_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                website_content.append(row)
        print(
            f"Website content loaded successfully. Total entries: {len(website_content)}"
        )
        return website_content
    except FileNotFoundError:
        print(f"Error: website_content.csv file not found at {csv_path}")
        return []
    except Exception as e:
        print(f"Error loading website content: {str(e)}")
        return []


def create_website_content_document(content_data: dict) -> Document:
    """Create a document for website content embedding."""
    question = content_data.get("question", "")
    answer = content_data.get("answer", "")
    content_type = content_data.get("type", "General")

    # Combine question and answer for better searchability
    page_content = f"Question: {question}\nAnswer: {answer}"

    # Create metadata for the document
    metadata = {
        "question": preprocess_text(question),
        "answer": preprocess_text(answer),
        "type": content_type,
        "url": f"/faq#{question.lower().replace(' ', '-').replace('?', '')}",
        "source": "website_content_csv",
    }

    return Document(page_content=page_content, metadata=metadata)


def embed_website_content():
    """Embed all website content into the vector store."""
    print("Starting website content embedding process...")

    # Load website content from CSV
    website_content = load_website_content_from_csv()

    if not website_content:
        print("No website content to embed.")
        return

    # Create documents for embedding
    documents = []
    document_ids = []

    for i, content in enumerate(website_content):
        try:
            document = create_website_content_document(content)
            documents.append(document)
            # Create unique ID for each document
            doc_id = f"website-content-{i+1}"
            document_ids.append(doc_id)
        except Exception as e:
            print(f"Error creating document for content {i+1}: {str(e)}")
            continue

    if not documents:
        print("No valid documents created for embedding.")
        return

    try:
        # Clear existing website content embeddings
        print("Clearing existing website content embeddings...")
        # Note: This will clear all existing website content embeddings
        # If you want to add incrementally, remove this step
        existing_ids = []
        try:
            # Get existing document IDs (if any)
            existing_docs = website_content_vector_store.similarity_search("", k=1000)
            existing_ids = [
                doc.metadata.get("id", f"unknown-{i}")
                for i, doc in enumerate(existing_docs)
            ]
            if existing_ids:
                website_content_vector_store.delete(existing_ids)
                print(f"Cleared {len(existing_ids)} existing embeddings.")
        except Exception as e:
            print(f"Warning: Could not clear existing embeddings: {str(e)}")

        # Add new documents to vector store
        print(f"Adding {len(documents)} documents to vector store...")
        website_content_vector_store.add_documents(documents, ids=document_ids)

        print(f"Successfully embedded {len(documents)} website content entries.")
        print("Website content embedding process completed successfully!")

    except Exception as e:
        print(f"Error during embedding process: {str(e)}")
        raise


if __name__ == "__main__":
    embed_website_content()
