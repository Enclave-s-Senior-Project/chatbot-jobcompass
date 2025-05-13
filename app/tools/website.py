from langchain.agents import Tool
from ..vectorstore import vector_store


# Website search tool
def website_search(query):
    try:
        docs = vector_store.similarity_search(query, k=3)
        if not docs:
            return "No relevant website content found."
        return "\n\n".join(
            [
                f"From {doc.metadata['url']}:\n{doc.page_content[:200]}..."
                for doc in docs
            ]
        )
    except Exception as e:
        return f"Error searching website: {str(e)}"


website_tool = Tool(
    name="WebsiteSearch",
    func=website_search,
    description="Search website content for relevant information. Returns up to 3 snippets with URLs.",
)
