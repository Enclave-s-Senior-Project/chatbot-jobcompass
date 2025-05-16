from langchain.agents import Tool
from app.vectorstore import website_content_vector_store
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from app.llm import llm
import html


# Website search tool
def website_search(query):
    try:
        # First stage: Initial retrieval with more documents
        docs = website_content_vector_store.similarity_search(query, k=5)

        if not docs:
            return "No relevant website content found."

        # Second stage: Apply contextual compression/reranking
        # This helps filter out less relevant information
        embeddings = website_content_vector_store.embeddings
        compressor = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.7)
        compression_retriever = ContextualCompressionRetriever(
            base_retriever=website_content_vector_store.as_retriever(
                search_kwargs={"k": 5}
            ),
            doc_compressor=compressor,
        )
        compressed_docs = compression_retriever.get_relevant_documents(query)

        # Use the compressed/reranked docs if available, otherwise fall back to original
        final_docs = compressed_docs if compressed_docs else docs

        # Format response as HTML for better presentation
        formatted_results = []
        for doc in final_docs[:3]:  # Limit to top 3 for concise response
            content = (
                doc.page_content[:200] + "..."
                if len(doc.page_content) > 200
                else doc.page_content
            )
            formatted_results.append(
                f'<div class="content-card">\n'
                f'  <h4><a href="{html.escape(doc.metadata["url"])}">{html.escape(doc.metadata["url"].split("/")[-1] or "Home Page")}</a></h4>\n'
                f"  <p>{html.escape(content)}</p>\n"
                f"</div>"
            )

        return "\n".join(formatted_results)
    except Exception as e:
        return f"Error searching website: {str(e)}"


website_tool = Tool(
    name="WebsiteSearch",
    func=website_search,
    description="Search website content for relevant information. Returns up to 3 snippets with URLs.",
)
