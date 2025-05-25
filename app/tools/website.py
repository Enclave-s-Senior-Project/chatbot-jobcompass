from langchain.agents import Tool
from app.vectorstore import website_content_vector_store
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from app.llm import llm
import html


# Website search tool
def website_search(query):
    try:
        docs = website_content_vector_store.similarity_search(query, k=5)

        if not docs:
            return "No relevant website content found."

        formatted_results = []
        for doc in docs:
            formatted_results.append(doc.metadata)

        return formatted_results
    except Exception as e:
        return f"Error searching website: {str(e)}"


website_tool = Tool(
    name="WebsiteSearch",
    func=website_search,
    description="Search website content for relevant information. Returns up to 3 snippets with URLs.",
)
