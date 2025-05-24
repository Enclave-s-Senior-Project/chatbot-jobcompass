from langchain.agents import Tool
from app.vectorstore import job_vector_store
from dotenv import load_dotenv
from os import getenv
from app.utils import get_job_details, format_salary
import re

load_dotenv()


def format_list_items(items, default="Not specified"):
    """Format a list of items into a comma-separated string"""
    if not items:
        return default

    if isinstance(items, list):
        item_names = []
        for item in items:
            if isinstance(item, dict) and "name" in item:
                item_names.append(item["name"])
            elif isinstance(item, dict) and "categoryName" in item:
                item_names.append(item["categoryName"])
            elif isinstance(item, str):
                item_names.append(item)
        if item_names:
            return ", ".join(item_names)
    elif isinstance(items, str):
        return items

    return default


def extract_list_items(items):
    """Extract text from list of dicts or strings"""
    if not items:
        return ""
    if isinstance(items, list):
        extracted = []
        for item in items:
            if isinstance(item, dict):
                name = (
                    item.get("name")
                    or item.get("categoryName")
                    or item.get("category_name", "")
                    or item.get("text", "")
                )
                if name:
                    extracted.append(str(name))
            elif isinstance(item, str):
                extracted.append(item)
        return " ".join(extracted)
    return str(items)


def extract_job_data(job_details, metadata):
    """Extract and structure job data from job_details or fallback to metadata"""
    url = f"{getenv('DETAILS_SINGLE_JOB_FRONTEND_LINK')}/{job_details.get('jobId', metadata.get('job_id'))}"
    if job_details:
        return {
            "priority_points": (
                job_details.get("boostedJob", {}).get("pointsUsed", 0)
                if job_details.get("boostedJob", {})
                else 0
            ),
            "job_name": job_details.get("name", ""),
            "description": job_details.get("description", ""),
            "categories_text": extract_list_items(job_details.get("categories", [])),
            "specializations_text": extract_list_items(
                job_details.get("specializations", [])
            ),
            "tags_text": extract_list_items(job_details.get("tags", [])),
            "requirement": job_details.get("requirement", ""),
            "company_name": (
                job_details.get("enterprise", {}).get("name", "")
                if job_details.get("enterprise", {})
                else metadata.get("company", "")
            ),
            "location_text": _extract_location_from_addresses(
                job_details.get("addresses", [])
            ),
            "url": url,
            "job_details": job_details,
        }
    else:
        return {
            "priority_points": job_details.get("points_used", 0),
            "job_name": metadata.get("job_name", ""),
            "description": metadata.get("description", ""),
            "categories_text": extract_list_items(metadata.get("categories", [])),
            "specializations_text": extract_list_items(
                metadata.get("specializations", [])
            ),
            "tags_text": extract_list_items(metadata.get("tags", [])),
            "requirement": metadata.get("requirement", ""),
            "company_name": metadata.get("company", ""),
            "location_text": extract_list_items(metadata.get("locations", [])),
            url: url,
            "job_details": None,
        }


def _extract_location_from_addresses(addresses):
    """Extract location text from addresses array"""
    if not addresses:
        return ""

    locations = []
    for addr in addresses:
        if isinstance(addr, dict):
            city = addr.get("city", "")
            country = addr.get("country", "")
            if city:
                locations.append(city)
            if country:
                locations.append(country)
        elif isinstance(addr, str):
            locations.append(addr)
    return " ".join(locations)


def format_job_result(doc, job_data, index):
    """Format a single job result for display"""
    meta = doc.metadata
    job_id = meta.get("job_id", "Unknown")

    if job_data["job_details"]:
        return _format_detailed_job(job_data, job_id, index)
    else:
        return _format_metadata_job(meta, job_data, job_id, index)


def _format_detailed_job(job_data, job_id, index):
    """Format job with full details"""
    job_details = job_data["job_details"]

    # Process location from addresses
    addresses = job_details.get("addresses", [])
    locations_text = "Not specified"

    if addresses and isinstance(addresses, list):
        locations = []
        for address in addresses:
            if isinstance(address, dict) and "city" in address:
                city = address.get("city", "")
                country = address.get("country", "")
                location = f"{city}, {country}" if city and country else city or country
                if location:
                    locations.append(location)
            elif isinstance(address, str):
                locations.append(address)
        if locations:
            locations_text = ", ".join(locations)

    # Get enterprise details
    enterprise = job_details.get("enterprise", {})
    company_name = enterprise.get("name", "Unknown Company")
    organization_type = enterprise.get("organizationType", "Not specified")

    # Extract job details
    job_info = {
        "name": job_details.get("name", "Unknown Job"),
        "status": job_details.get("status", "Not specified"),
        "type": job_details.get("type", "Not specified"),
        "lowest_wage": job_details.get("lowestWage", 0),
        "highest_wage": job_details.get("highestWage", 0),
        "deadline": job_details.get("deadline", "Not specified"),
        "experience": job_details.get("experience", "Not specified"),
        "education": job_details.get("education", "Not specified"),
        "points_used": (
            job_details.get("boostedJob", {}).get("pointsUsed", 0)
            if job_details.get("boostedJob", {})
            else 0
        ),
    }

    return f"""
Priority Points: {job_info['points_used']}
JOB {index}: {job_info['name']}
Company: {company_name}
Status: {job_info['status']} | Type: {job_info['type']}
Salary: {format_salary(job_info['lowest_wage'], job_info['highest_wage'])}
Deadline: {job_info['deadline']}
Location: {locations_text}
Company Type: {organization_type}
Experience Required: {job_info['experience']} years
Education: {job_info['education']}
Industries [Working Fields]: {format_list_items(job_details.get("categories", []))}
Majorities: {format_list_items(job_details.get("specializations", []))}
Keywords: {format_list_items(job_details.get("tags", []))}
View Details: {getenv("DETAILS_SINGLE_JOB_FRONTEND_LINK")}/{job_id}
""".strip()


def _format_metadata_job(meta, job_data, job_id, index):
    """Format job using metadata only"""
    locations = meta.get("locations", [])
    location_text = (
        ", ".join(locations)
        if isinstance(locations, list)
        else meta.get("location", "Not specified")
    )

    salary_range = meta.get("salary_range", {})
    lowest_wage = salary_range.get("min", 0)
    highest_wage = salary_range.get("max", 0)

    return f"""
JOB {index}: {job_data['job_name']}
Company: {job_data['company_name']}
Status: {meta.get('status', 'Not specified')}
Type: {meta.get('job_type', 'Not specified')}
Salary: {format_salary(lowest_wage, highest_wage)}
Deadline: {meta.get('deadline', 'Not specified')}
Location: {location_text}
Company Type: {meta.get('organization_type', 'Not specified')}
Experience Required: {meta.get('experience', 'Not specified')} years
Education: {meta.get('education', 'Not specified')}
Industries [Working Fields]: {job_data['categories_text']}
Majorities: {job_data['specializations_text']}
Keywords: {job_data['tags_text']}
View Details: {getenv("DETAILS_SINGLE_JOB_FRONTEND_LINK")}/{job_id}
""".strip()


def job_vector_search(query):
    """Search for jobs using natural language query with enhanced relevance scoring"""
    try:
        # Get initial results with higher k for better filtering
        docs = job_vector_store.similarity_search(query, k=15)
        print(f"Found {len(docs)} jobs matching the query.")
        if not docs:
            return "No jobs found matching your criteria."

        formatted_jobs = []
        for i, doc in enumerate(docs):
            meta = doc.metadata
            job_id = meta.get("job_id", "Unknown")
            job_details = get_job_details(job_id)

            job_data = extract_job_data(job_details, meta)

            final = format_job_result(doc, job_data, i)

            formatted_jobs.append(final)

        return formatted_jobs

    except Exception as e:
        return f"Error searching jobs: {str(e)}"


job_tool = Tool(
    name="JobSearch",
    description="Search for jobs using a natural language query.",
    func=job_vector_search,
)
