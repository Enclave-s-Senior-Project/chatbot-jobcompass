from langchain.agents import Tool
from app.vectorstore import job_vector_store
from dotenv import load_dotenv
from os import getenv
from utils.api_client import get_job_details
from utils.format_salary import format_salary
import html

load_dotenv()


# HTML Component Helper Functions
def create_job_header(job_id, job_name, company_name, status, job_type):
    """Generate HTML for job header section with title, company name and status badges"""
    return f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 15px;">
        <div>
            <a href="{getenv("DETAILS_SINGLE_JOB_FRONTEND_LINK")}/{job_id}" target="_blank" style="text-decoration: none;">
                <h3 style="margin: 0; color: #2563eb; font-size: 1.5rem; font-weight: 600; display: flex; align-items: center;">
                    {html.escape(job_name)}
                    <span style="margin-left: 8px; font-size: 0.8rem; color: #6b7280;">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                            <path d="M8.636 3.5a.5.5 0 0 0-.5-.5H1.5A1.5 1.5 0 0 0 0 4.5v10A1.5 1.5 0 0 0 1.5 16h10a1.5 1.5 0 0 0 1.5-1.5V7.864a.5.5 0 0 0-1 0V14.5a.5.5 0 0 1-.5.5h-10a.5.5 0 0 1-.5-.5v-10a.5.5 0 0 1 .5-.5h6.636a.5.5 0 0 0 .5-.5z"/>
                            <path d="M16 .5a.5.5 0 0 0-.5-.5h-5a.5.5 0 0 0 0 1h3.793L6.146 9.146a.5.5 0 1 0 .708.708L15 1.707V5.5a.5.5 0 0 0 1 0v-5z"/>
                        </svg>
                    </span>
                </h3>
            </a>
            <h4 style="margin: 5px 0; color: #4b5563; font-weight: 500; font-size: 1.1rem;">{html.escape(company_name)}</h4>
            <div style="display: flex; flex-wrap: wrap; margin-top: 5px;">
                <span style="font-size: 0.85rem; background-color: #f3f4f6; border-radius: 20px; padding: 3px 12px; margin-right: 8px; margin-bottom: 5px; color: #4b5563; border: 1px solid #e5e7eb;">{html.escape(status)}</span>
                <span style="font-size: 0.85rem; background-color: #f3f4f6; border-radius: 20px; padding: 3px 12px; margin-right: 8px; margin-bottom: 5px; color: #4b5563; border: 1px solid #e5e7eb;">{html.escape(job_type)}</span>
            </div>
        </div>
    """


def create_salary_info(lowest_wage, highest_wage, deadline):
    """Generate HTML for salary and deadline information"""
    return f"""
        <div style="text-align: right;">
            <p style="margin: 0; font-weight: 600; color: #059669; font-size: 1.1rem; background-color: #ecfdf5; padding: 6px 12px; border-radius: 6px; display: inline-block;">{format_salary(lowest_wage, highest_wage)}</p>
            <p style="margin: 5px 0 0 0; color: #6b7280; font-size: 0.9rem;">
                <span style="background-color: #fee2e2; color: #ef4444; padding: 3px 8px; border-radius: 4px; font-weight: 500;">Deadline: {html.escape(str(deadline))}</span>
            </p>
        </div>
    </div>
    """


def create_info_grid_item(label, value):
    """Generate HTML for a single grid item in the job details section"""
    return f"""
    <div style="background-color: #f9fafb; border-radius: 8px; padding: 12px; border: 1px solid #f3f4f6; transition: transform 0.2s; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
        <p style="margin: 0 0 5px 0; font-weight: 600; color: #374151; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em;">{label}</p>
        <p style="margin: 0; padding: 8px; background-color: white; border-radius: 4px; color: #4b5563; font-weight: 500; border-left: 3px solid #6366f1;">{html.escape(str(value))}</p>
    </div>
    """


def create_job_info_section(label, content):
    """Generate HTML for a job information section (categories, specializations, etc.)"""
    return f"""
    <div style="margin-bottom: 15px; background-color: #f9fafb; border-radius: 8px; padding: 15px; border: 1px solid #f3f4f6;">
        <p style="margin: 0 0 5px 0; font-weight: 600; color: #374151; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em;">{label}</p>
        <p style="margin: 0; padding: 10px; background-color: white; border-radius: 6px; color: #4b5563; line-height: 1.6; border-left: 3px solid #6366f1;">{content}</p>
    </div>
    """


def create_job_footer(job_id):
    """Generate HTML for job footer with action button"""
    return f"""
    <div style="text-align: right; margin-top: 15px; padding-top: 15px; border-top: 1px solid #e5e7eb;">
        <a href="{getenv("DETAILS_SINGLE_JOB_FRONTEND_LINK")}/{job_id}" target="_blank" 
           style="display: inline-block; background-color: #2563eb; color: white; padding: 10px 20px; 
                  border-radius: 8px; text-decoration: none; font-weight: 500; transition: background-color 0.3s;
                  box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2);">
           View Complete Job Details
        </a>
    </div>
    """


def create_job_card_wrapper(content):
    """Wrap job content in a styled card container"""
    return f"""
    <div class="job-card" style="border-radius: 12px; border: 1px solid #e0e0e0; box-shadow: 0 4px 8px rgba(0,0,0,0.05); 
                                  padding: 25px; margin: 20px 0; background-color: #ffffff; transition: transform 0.2s, box-shadow 0.2s;
                                  overflow: hidden; position: relative;">
        <div style="position: absolute; top: 0; left: 0; width: 5px; height: 100%; background: linear-gradient(to bottom, #3b82f6, #6366f1);"></div>
        <div style="margin-left: 10px;">
            {content}
        </div>
    </div>
    """


def format_list_items(items, default="Not specified"):
    """Format a list of items into a comma-separated string with proper HTML escaping"""
    if not items:
        return default

    if isinstance(items, list):
        item_names = []
        for item in items:
            if isinstance(item, dict) and "name" in item:
                item_names.append(html.escape(item["name"]))
            elif isinstance(item, dict) and "categoryName" in item:
                item_names.append(html.escape(item["categoryName"]))
            elif isinstance(item, str):
                item_names.append(html.escape(item))
        if item_names:
            return ", ".join(item_names)
    elif isinstance(items, str):
        return html.escape(items)

    return default


def job_vector_search(query):
    try:
        docs = job_vector_store.similarity_search(query, k=3)
        if not docs:
            return '<p style="padding: 15px; background-color: #fee2e2; border-radius: 8px; color: #b91c1c; font-weight: 500; border-left: 4px solid #ef4444;">No jobs found matching your criteria.</p>'

        formatted = []
        for doc in docs:
            meta = doc.metadata
            job_id = meta.get("job_id", "Unknown")

            # get detailed job information from API
            job_details = get_job_details(job_id)

            if job_details:
                # Format tags
                tags = job_details.get("tags", meta.get("tags", []))
                tags_html = format_list_items(tags)

                # Format categories
                categories = job_details.get("categories", meta.get("categories", []))
                categories_html = format_list_items(categories)

                # Format specializations
                specializations = job_details.get(
                    "specializations", meta.get("specializations", [])
                )
                specializations_html = format_list_items(specializations)

                # Process location from addresses array
                addresses = job_details.get("addresses", meta.get("locations", []))
                locations_html = "Not specified"

                if addresses:
                    if isinstance(addresses, list):
                        locations = []
                        for address in addresses:
                            if isinstance(address, dict) and "city" in address:
                                city = address.get("city", "")
                                country = address.get("country", "")
                                location = ""
                                if city and country:
                                    location = f"{city}, {country}"
                                elif city:
                                    location = city
                                elif country:
                                    location = country
                                locations.append(location)
                            elif isinstance(address, str):
                                locations.append(html.escape(address))
                        if locations:
                            locations_html = ", ".join(locations)

                # Get enterprise details
                enterprise = job_details.get("enterprise", {})
                company_name = enterprise.get(
                    "name", meta.get("company", "Unknown Company")
                )
                organization_type = enterprise.get(
                    "organizationType", meta.get("organization_type", "Not specified")
                )

                # Build job card content using components
                job_name = job_details.get("name", meta.get("job_name", "Unknown Job"))
                job_status = job_details.get(
                    "status", meta.get("status", "Not specified")
                )
                job_type = job_details.get("type", meta.get("type", "Not specified"))
                lowest_wage = job_details.get(
                    "lowestWage", meta.get("salary_range", {}).get("min", 0)
                )
                highest_wage = job_details.get(
                    "highestWage", meta.get("salary_range", {}).get("max", 0)
                )
                deadline = job_details.get(
                    "deadline", meta.get("deadline", "Not specified")
                )
                experience = job_details.get(
                    "experience", meta.get("experience", "Not specified")
                )
                education = job_details.get(
                    "education", meta.get("education", "Not specified")
                )

                # Construct job card from components
                job_header = create_job_header(
                    job_id, job_name, company_name, job_status, job_type
                )
                salary_info = create_salary_info(lowest_wage, highest_wage, deadline)

                # Info grid items
                info_grid = f"""
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0;">
                    {create_info_grid_item("Location", locations_html)}
                    {create_info_grid_item("Company Type", organization_type)}
                    {create_info_grid_item("Experience", f"{experience} years")}
                    {create_info_grid_item("Education", education)}
                </div>
                """

                # Information sections
                info_sections = (
                    create_job_info_section("Industries/Categories", categories_html)
                    + create_job_info_section("Specializations", specializations_html)
                    + create_job_info_section("Keywords", tags_html)
                )

                # Add a prominent "View Details" button in the middle of the card
                details_button = f"""
                <div style="text-align: center; margin: 20px 0;">
                    <a href="{getenv("DETAILS_SINGLE_JOB_FRONTEND_LINK")}/{job_id}" target="_blank" 
                       style="display: inline-flex; align-items: center; justify-content: center; background-color: #f0f9ff; color: #0369a1; 
                              padding: 12px 24px; border-radius: 8px; text-decoration: none; font-weight: 600; transition: all 0.3s;
                              border: 1px solid #bae6fd; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                        <span style="margin-right: 8px;">
                            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="currentColor" viewBox="0 0 16 16">
                                <path d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0zM4.5 7.5a.5.5 0 0 0 0 1h5.793l-2.147 2.146a.5.5 0 0 0 .708.708l3-3a.5.5 0 0 0 0-.708l-3-3a.5.5 0 1 0-.708.708L10.293 7.5H4.5z"/>
                            </svg>
                        </span>
                        View Full Job Details
                    </a>
                </div>
                """

                job_footer = create_job_footer(job_id)

                # Combine all sections
                job_content = (
                    job_header
                    + salary_info
                    + info_grid
                    + info_sections
                    + details_button
                    + job_footer
                )
                formatted_job = create_job_card_wrapper(job_content)

                formatted.append(formatted_job)

            else:
                # Create a better fallback display for when API details aren't available
                categories = meta.get("categories", [])
                categories_html = format_list_items(categories)

                tags = meta.get("tags", [])
                tags_html = format_list_items(tags)

                specializations = meta.get("specializations", [])
                specializations_html = format_list_items(specializations)

                # Process locations
                locations = meta.get("locations", [])
                location_text = (
                    "; ".join(locations)
                    if isinstance(locations, list)
                    else meta.get("location", "Not specified")
                )

                # Build fallback job card using our components
                job_name = meta.get("job_name", "Unknown Job")
                company_name = meta.get("company", "Unknown Company")
                job_status = meta.get("status", "Not specified")
                job_type = meta.get("job_type", "Not specified")

                job_header = create_job_header(
                    job_id, job_name, company_name, job_status, job_type
                )

                lowest_wage = meta.get("salary_range", {}).get("min", 0)
                highest_wage = meta.get("salary_range", {}).get("max", 0)
                deadline = meta.get("deadline", "Not specified")

                salary_info = create_salary_info(lowest_wage, highest_wage, deadline)

                info_grid = f"""
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0;">
                    {create_info_grid_item("Location", location_text)}
                    {create_info_grid_item("Company Type", meta.get("organization_type", "Not specified"))}
                    {create_info_grid_item("Experience", f"{meta.get('experience', 'Not specified')} years")}
                    {create_info_grid_item("Education", meta.get("education", "Not specified"))}
                </div>
                """

                info_sections = (
                    create_job_info_section("Industries/Categories", categories_html)
                    + create_job_info_section("Specializations", specializations_html)
                    + create_job_info_section("Keywords", tags_html)
                )

                # Add a prominent "View Details" button in the middle of the card
                details_button = f"""
                <div style="text-align: center; margin: 20px 0;">
                    <a href="{getenv("DETAILS_SINGLE_JOB_FRONTEND_LINK")}/{job_id}" target="_blank" 
                       style="display: inline-flex; align-items: center; justify-content: center; background-color: #f0f9ff; color: #0369a1; 
                              padding: 12px 24px; border-radius: 8px; text-decoration: none; font-weight: 600; transition: all 0.3s;
                              border: 1px solid #bae6fd; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                        <span style="margin-right: 8px;">
                            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="currentColor" viewBox="0 0 16 16">
                                <path d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0zM4.5 7.5a.5.5 0 0 0 0 1h5.793l-2.147 2.146a.5.5 0 0 0 .708.708l3-3a.5.5 0 0 0 0-.708l-3-3a.5.5 0 1 0-.708.708L10.293 7.5H4.5z"/>
                            </svg>
                        </span>
                        View Full Job Details
                    </a>
                </div>
                """

                job_footer = create_job_footer(job_id)

                job_content = (
                    job_header
                    + salary_info
                    + info_grid
                    + info_sections
                    + details_button
                    + job_footer
                )
                formatted_job = create_job_card_wrapper(job_content)

                formatted.append(formatted_job)

        return "\n".join(formatted)
    except Exception as e:
        return f'<p style="padding: 15px; background-color: #fee2e2; border-radius: 8px; color: #b91c1c; font-weight: 500; border-left: 4px solid #ef4444;">Error searching jobs: {html.escape(str(e))}</p>'


job_tool = Tool(
    name="JobSearch",
    description="Search for jobs using a natural language query.",
    func=job_vector_search,
)
