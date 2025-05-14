from langchain.agents import Tool
from app.vectorstore import job_vector_store


def job_vector_search(query):
    try:
        docs = job_vector_store.similarity_search(query, k=3)
        if not docs:
            return "No jobs found matching the criteria."
        formatted = []
        for doc in docs:
            meta = doc.metadata
            # Extract names from JSON metadata
            categories = (
                ", ".join([cat["name"] for cat in meta["categories"]])
                if meta["categories"]
                else "None"
            )
            tags = (
                ", ".join([tag["name"] for tag in meta["tags"]])
                if meta["tags"]
                else "None"
            )
            specializations = (
                ", ".join([spec["name"] for spec in meta["specializations"]])
                if meta["specializations"]
                else "None"
            )
            # Extract requirements safely
            requirements = ""
            content_lines = doc.page_content.split("\n")
            for line in content_lines:
                if line.startswith("Requirements: "):
                    requirements = line.replace("Requirements: ", "")[:100]
                    break
            formatted.append(
                f"Job Title: {content_lines[0].replace('Job Title: ', '')}\n"
                f"Company: {meta['company']}\n"
                f"Location: {meta['city']}, {meta['country']}\n"
                f"Categories: {categories}\n"
                f"Tags: {tags}\n"
                f"Specializations: {specializations}\n"
                f"Experience: {meta['experience']} years\n"
                f"Education: {meta['education']}\n"
                f"Salary: ${meta['lowest_wage']:,.0f} - ${meta['highest_wage']:,.0f}\n"
                f"Status: {meta['status']}\n"
                f"Deadline: {meta['deadline']}\n"
                f"Requirements: {requirements or 'Not provided'}..."
            )
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Error searching jobs: {str(e)}"


job_tool = Tool(
    name="JobSearch",
    description="Search for jobs using a natural language query.",
    func=job_vector_search,
)
