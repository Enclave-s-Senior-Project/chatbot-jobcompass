from langchain.agents import Tool

from app.utils import get_enterprise_details
from ..vectorstore import enterprise_vector_store
from dotenv import load_dotenv
import os

load_dotenv()


# Text Component Helper Functions
def create_enterprise_header(
    enterprise_id, name, logo_url, status, founded_in, team_size
):
    """Generate plain text for enterprise header section"""
    link = f"{os.getenv('DETAILS_ENTERPRISE_FRONTEND_LINK')}/{enterprise_id}"
    return f"""Company: {name}
Logo Url: {logo_url or 'No logo available'}
Status: {status or 'Not specified'}
Founded: {founded_in or 'N/A'}
Team Size: {team_size or 'N/A'}
Profile Link: {link}
"""


def create_description_section(description):
    """Generate plain text for enterprise description section"""
    return f"Description: {description or 'No description available'}\n"


def create_enterprise_info_grid(items):
    """Generate plain text for a grid of information items"""
    grid_text = ""
    for item in items:
        label, value = item
        grid_text += f"{label}: {value}\n"
    return grid_text


def create_enterprise_info_section(label, content):
    """Generate plain text for an enterprise information section"""
    return f"{label}: {content}\n"


def create_enterprise_footer(enterprise_id):
    """Generate plain text for enterprise footer with action link"""
    return f"Complete Profile: {os.getenv('DETAILS_ENTERPRISE_FRONTEND_LINK')}/{enterprise_id}\n"


def format_list_items(items, default="Not specified"):
    """Format a list of items into a comma-separated string"""
    if not items:
        return default

    if isinstance(items, list):
        item_names = []
        for item in items:
            if isinstance(item, dict) and "categoryName" in item:
                item_names.append(item["categoryName"])
            elif isinstance(item, dict) and "name" in item:
                item_names.append(item["name"])
            elif isinstance(item, str):
                item_names.append(item)
        if item_names:
            return ", ".join(item_names)
    elif isinstance(items, str):
        return items

    return default


def enterprise_vector_search(query):
    try:
        docs = enterprise_vector_store.similarity_search(query, k=10)
        if not docs:
            return "No results found."

        formatted_results = []
        for doc in docs:
            meta = doc.metadata
            enterprise_id = meta.get("enterprise_id")
            enterprise_details = get_enterprise_details(enterprise_id)

            if enterprise_details:
                # Format categories/industries
                categories_text = format_list_items(
                    enterprise_details.get("categories")
                )

                # Format addresses
                addresses_text = "Not specified"
                if enterprise_details.get("addresses"):
                    address_list = []
                    for addr in enterprise_details.get("addresses", []):
                        if isinstance(addr, dict) and "mixedAddress" in addr:
                            address_list.append(addr["mixedAddress"])
                    if address_list:
                        addresses_text = "; ".join(address_list)

                # Format websites/contact
                websites_text = "Not specified"
                if enterprise_details.get("websites"):
                    website_list = []
                    for website in enterprise_details.get("websites", []):
                        if isinstance(website, dict) and "url" in website:
                            website_list.append(website["url"])
                    if website_list:
                        websites_text = "; ".join(website_list)

                # Build enterprise card using components
                enterprise_header = create_enterprise_header(
                    enterprise_details.get("enterpriseId"),
                    enterprise_details.get("name", "Unknown Enterprise"),
                    enterprise_details.get("logoUrl", ""),
                    enterprise_details.get("status"),
                    enterprise_details.get("foundedIn"),
                    enterprise_details.get("teamSize"),
                )

                description_section = create_description_section(
                    enterprise_details.get("description")
                )

                # Info grid
                info_grid = create_enterprise_info_grid(
                    [
                        ("Industries", categories_text),
                        (
                            "Organization Type",
                            enterprise_details.get("organizationType")
                            or "Not specified",
                        ),
                    ]
                )

                # Info sections
                addresses_section = create_enterprise_info_section(
                    "Addresses", addresses_text
                )
                contact_section = create_enterprise_info_section(
                    "Contact", websites_text
                )

                # Footer
                enterprise_footer = create_enterprise_footer(
                    enterprise_details.get("enterpriseId")
                )

                # Combine all sections
                enterprise_content = (
                    enterprise_header
                    + description_section
                    + info_grid
                    + addresses_section
                    + contact_section
                    + enterprise_footer
                )

                formatted_results.append(enterprise_content)
            else:
                # Format categories
                categories_text = (
                    [cat["category_name"] for cat in meta.get("categories")]
                    if meta.get("categories")
                    else []
                )

                # Format addresses
                addresses_text = "Not specified"
                if meta.get("addresses"):
                    address_list = []
                    for addr in meta.get("addresses", []):
                        if isinstance(addr, dict) and "mixed_address" in addr:
                            address_list.append(addr["mixed_address"])
                    if address_list:
                        addresses_text = "; ".join(address_list)

                # Build fallback enterprise card using components
                enterprise_header = create_enterprise_header(
                    meta.get("enterprise_id"),
                    meta.get("name", "Unknown Enterprise"),
                    meta.get("logo_url", ""),
                    meta.get("status"),
                    meta.get("founded_in"),
                    meta.get("team_size"),
                )

                description_section = create_description_section(
                    meta.get("description")
                )

                # Info grid
                info_grid = create_enterprise_info_grid(
                    [
                        ("Industries", categories_text),
                        (
                            "Organization Type",
                            meta.get("organization_type") or "Not specified",
                        ),
                    ]
                )

                # Info sections
                addresses_section = create_enterprise_info_section(
                    "Addresses", addresses_text
                )

                # Footer
                enterprise_footer = create_enterprise_footer(meta.get("enterprise_id"))

                # Combine all sections
                enterprise_content = (
                    enterprise_header
                    + description_section
                    + info_grid
                    + addresses_section
                    + enterprise_footer
                )

                formatted_results.append(enterprise_content)

        return "\n".join(formatted_results)
    except Exception as e:
        return f"Error searching enterprises: {str(e)}"


enterprise_tool = Tool(
    name="EnterpriseSearch",
    func=enterprise_vector_search,
    description="Use this tool to search for enterprise data. The input should be a string of text.",
)
