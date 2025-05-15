from langchain.agents import Tool
from app.vectorstore import job_vector_store
from dotenv import load_dotenv
from os import getenv
from utils.api_client import get_job_details
from utils.format_salary import format_salary
import html

load_dotenv()


def job_vector_search(query):
    try:
        docs = job_vector_store.similarity_search(query, k=3)
        if not docs:
            return "<p>No jobs found matching the criteria.</p>"

        formatted = []
        for doc in docs:
            meta = doc.metadata
            job_id = meta.get("job_id", "Unknown")

            # First try to get detailed job information from API
            job_details = get_job_details(job_id)

            if job_details:
                # Format tags, categories and specializations for HTML
                # Process tags - now they're objects with name property
                tags = job_details.get("tags", meta.get("tags", []))
                tags_html = (
                    ", ".join([html.escape(tag.get("name", "")) for tag in tags])
                    if tags and isinstance(tags, list)
                    else "Not specified"
                )

                # Process categories - now they're objects with categoryName property
                categories = job_details.get("categories", meta.get("categories", []))
                categories_html = (
                    ", ".join(
                        [html.escape(cat.get("categoryName", "")) for cat in categories]
                    )
                    if categories and isinstance(categories, list)
                    else "Not specified"
                )

                # Process specializations - now they're objects with categoryName property
                specializations = job_details.get(
                    "specializations", meta.get("specializations", [])
                )
                specializations_html = (
                    ", ".join(
                        [
                            html.escape(spec.get("categoryName", ""))
                            for spec in specializations
                        ]
                    )
                    if specializations and isinstance(specializations, list)
                    else "Not specified"
                )

                # Process location from addresses array
                location = "Not specified"
                if (
                    job_details.get("addresses")
                    and isinstance(job_details["addresses"], list)
                    and len(job_details["addresses"]) > 0
                ):
                    address = job_details["addresses"][0]
                    city = address.get("city", "")
                    country = address.get("country", "")
                    location = (
                        f"{city}, {country}" if city and country else city or country
                    )

                # Format requirements for HTML list - requirements field contains HTML
                requirements = job_details.get("requirements", "Not specified")
                requirements_html = (
                    requirements
                    if requirements and requirements != "Not specified"
                    else "<p>Not specified</p>"
                )

                # Format job description - description field contains HTML
                description = job_details.get("description", "Not specified")
                description_html = (
                    description
                    if description and description != "Not specified"
                    else "<p>Not specified</p>"
                )

                # Format responsibility section if available
                responsibility = job_details.get("responsibility", "")
                responsibility_html = ""
                if responsibility and responsibility != "Not specified":
                    responsibility_html = f"  <p><strong>Job Responsibilities:</strong></p>\n  {responsibility}\n"

                # Get enterprise details
                enterprise = job_details.get("enterprise", {})
                company_name = enterprise.get(
                    "name", meta.get("company", "Unknown Company")
                )
                organization_type = enterprise.get(
                    "organizationType", meta.get("organization_type", "Not specified")
                )

                # Create formatted HTML for job listing with all available information
                formatted.append(
                    f'<div class="job-card">\n'
                    f'  <h3><a href="{getenv("DETAILS_SINGLE_JOB_FRONTEND_LINK")}/{job_id}">{html.escape(job_details.get("name", meta.get("job_name", "Unknown Job")))}</a></h3>\n'
                    f'  <p><strong>Boosted Points:</strong> {1 if job_details.get("isBoost", False) else 0}</p>\n'
                    f"  <p><strong>Company:</strong> {html.escape(company_name)}</p>\n"
                    f"  <p><strong>Company Type:</strong> {html.escape(organization_type)}</p>\n"
                    f"  <p><strong>Location:</strong> {html.escape(location)}</p>\n"
                    f"  <p><strong>Industries/Categories:</strong> {categories_html}</p>\n"
                    f"  <p><strong>Keywords:</strong> {tags_html}</p>\n"
                    f"  <p><strong>Specializations:</strong> {specializations_html}</p>\n"
                    f'  <p><strong>Experience:</strong> {job_details.get("experience", meta.get("experience", "Not specified"))} years</p>\n'
                    f'  <p><strong>Education:</strong> {html.escape(str(job_details.get("education", meta.get("education", "Not specified"))))}</p>\n'
                    f'  <p><strong>Salary:</strong> {format_salary(job_details.get("lowestWage", meta.get("salary_range", {}).get("min", 0)), job_details.get("highestWage", meta.get("salary_range", {}).get("max", 0)))}</p>\n'
                    f'  <p><strong>Status:</strong> {html.escape(job_details.get("status", meta.get("status", "Not specified")))}</p>\n'
                    f'  <p><strong>Job Type:</strong> {html.escape(job_details.get("type", meta.get("type", "Not specified")))}</p>\n'
                    f'  <p><strong>Deadline:</strong> {html.escape(str(job_details.get("deadline", meta.get("deadline", "Not specified"))))}</p>\n'
                    f"  <p><strong>Job Description:</strong></p>\n"
                    f"  {description_html}\n"
                    f"{responsibility_html}"
                    f"  <p><strong>Job Requirements:</strong></p>\n"
                    f"  {requirements_html}\n"
                )

                # Add benefits section separately to avoid nested f-string
                if job_details.get("enterpriseBenefits"):
                    formatted[
                        -1
                    ] += f"  <p><strong>Benefits:</strong></p>\n  {job_details.get('enterpriseBenefits', '')}\n"

                # Add closing div and hr separately
                formatted[-1] += f"</div>\n<hr>\n"
            else:
                # Fall back to vector store metadata if API call fails
                # Format data for HTML
                categories = meta.get("categories", [])
                categories_html = (
                    ", ".join([html.escape(cat) for cat in categories])
                    if categories
                    else "Not specified"
                )

                tags = meta.get("tags", [])
                tags_html = (
                    ", ".join([html.escape(tag) for tag in tags])
                    if tags
                    else "Not specified"
                )

                specializations = meta.get("specializations", [])
                specializations_html = (
                    ", ".join([html.escape(spec) for spec in specializations])
                    if specializations
                    else "Not specified"
                )

                formatted.append(
                    f'<div class="job-card">\n'
                    f'  <h3><a href="{getenv("DETAILS_SINGLE_JOB_FRONTEND_LINK")}/{job_id}">{html.escape(meta.get("job_name", "Unknown Job"))}</a></h3>\n'
                    f'  <p><strong>Boosted Points:</strong> {meta.get("points_used", 0)}</p>\n'
                    f'  <p><strong>Company:</strong> {html.escape(meta.get("company", "Unknown Company"))}</p>\n'
                    f'  <p><strong>Company Type:</strong> {html.escape(meta.get("organization_type", "Not specified"))}</p>\n'
                    f'  <p><strong>Location:</strong> {html.escape(meta.get("location", "Location not specified"))}</p>\n'
                    f"  <p><strong>Industries/Categories:</strong> {categories_html}</p>\n"
                    f"  <p><strong>Keywords:</strong> {tags_html}</p>\n"
                    f"  <p><strong>Specializations:</strong> {specializations_html}</p>\n"
                    f'  <p><strong>Experience:</strong> {meta.get("experience", "Not specified")} years</p>\n'
                    f'  <p><strong>Education:</strong> {html.escape(str(meta.get("education", "Not specified")))}</p>\n'
                    f'  <p><strong>Salary:</strong> {format_salary(meta.get("salary_range", {}).get("min", 0), meta.get("salary_range", {}).get("max", 0))}</p>\n'
                    f'  <p><strong>Status:</strong> {html.escape(meta.get("status", "Not specified"))}</p>\n'
                    f'  <p><strong>Deadline:</strong> {html.escape(str(meta.get("deadline", "Not specified")))}</p>\n'
                    f"</div>\n"
                    f"<hr>\n"
                )

        return "\n".join(formatted)
    except Exception as e:
        return f"<p>Error searching jobs: {html.escape(str(e))}</p>"


job_tool = Tool(
    name="JobSearch",
    description="Search for jobs using a natural language query.",
    func=job_vector_search,
)
