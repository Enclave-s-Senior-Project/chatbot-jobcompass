from langchain.agents import Tool

from app.utils import get_enterprise_details
from ..vectorstore import enterprise_vector_store
from dotenv import load_dotenv
import html
import os

load_dotenv()


# HTML Component Helper Functions
def create_enterprise_header(
    enterprise_id, name, logo_url, status, founded_in, team_size
):
    """Generate HTML for enterprise header section with logo, name and badges"""
    return f"""
    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <div style="flex-shrink: 0; margin-right: 20px;">
            <img src="{html.escape(logo_url)}" alt="Logo" style="width: 80px; height: 80px; object-fit: contain; border-radius: 5px;"/>
        </div>
        <div>
            <a href="{os.getenv('DETAILS_ENTERPRISE_FRONTEND_LINK')}/{enterprise_id}" target="_blank" style="text-decoration: none;">
                <h3 style="margin: 0; color: #2563eb; font-size: 1.5rem;">{html.escape(name)}</h3>
            </a>
            <div style="display: flex; flex-wrap: wrap; margin-top: 5px;">
                <span style="font-size: 0.85rem; background-color: #f3f4f6; border-radius: 20px; padding: 3px 12px; margin-right: 8px; margin-bottom: 5px; color: #4b5563;">{html.escape(status or 'Not specified')}</span>
                <span style="font-size: 0.85rem; background-color: #f3f4f6; border-radius: 20px; padding: 3px 12px; margin-right: 8px; margin-bottom: 5px; color: #4b5563;">Founded: {html.escape(str(founded_in or 'N/A'))}</span>
                <span style="font-size: 0.85rem; background-color: #f3f4f6; border-radius: 20px; padding: 3px 12px; margin-right: 8px; margin-bottom: 5px; color: #4b5563;">Team: {html.escape(str(team_size or 'N/A'))}</span>
            </div>
        </div>
    </div>
    """


def create_description_section(description):
    """Generate HTML for enterprise description section"""
    return f"""
    <div style="padding: 10px; background-color: #f9fafb; border-radius: 6px; margin-bottom: 15px;">
        <p style="margin: 0; line-height: 1.6; color: #4b5563;">{html.escape(description or 'No description available')}</p>
    </div>
    """


def create_enterprise_info_grid(items):
    """Generate HTML for a grid of information items"""
    grid_items = ""
    for item in items:
        label, value = item
        grid_items += f"""
        <div>
            <p style="margin: 0 0 5px 0; font-weight: 600; color: #374151;">{label}</p>
            <p style="margin: 0; padding: 8px; background-color: #f3f4f6; border-radius: 4px; color: #4b5563;">{value}</p>
        </div>
        """

    return f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin-bottom: 15px;">
        {grid_items}
    </div>
    """


def create_enterprise_info_section(label, content):
    """Generate HTML for an enterprise information section"""
    return f"""
    <div style="margin-bottom: 15px;">
        <p style="margin: 0 0 5px 0; font-weight: 600; color: #374151;">{label}</p>
        <p style="margin: 0; padding: 8px; background-color: #f3f4f6; border-radius: 4px; color: #4b5563;">{content}</p>
    </div>
    """


def create_enterprise_footer(enterprise_id):
    """Generate HTML for enterprise footer with action button"""
    return f"""
    <div style="text-align: right; margin-top: 10px;">
        <a href="{os.getenv('DETAILS_ENTERPRISE_FRONTEND_LINK')}/{enterprise_id}" target="_blank" style="display: inline-block; background-color: #2563eb; color: white; padding: 8px 16px; border-radius: 6px; text-decoration: none; font-weight: 500;">View Complete Profile</a>
    </div>
    """


def create_enterprise_card_wrapper(content):
    """Wrap enterprise content in a styled card container"""
    return f"""
    <div class="enterprise-card" style="border-radius: 8px; border: 1px solid #e0e0e0; box-shadow: 0 2px 5px rgba(0,0,0,0.1); padding: 20px; margin: 15px 0; background-color: #ffffff;">
        {content}
    </div>
    """


def format_list_items(items, default="Not specified"):
    """Format a list of items into a comma-separated string with proper HTML escaping"""
    if not items:
        return default

    if isinstance(items, list):
        item_names = []
        for item in items:
            if isinstance(item, dict) and "categoryName" in item:
                item_names.append(html.escape(item["categoryName"]))
            elif isinstance(item, dict) and "name" in item:
                item_names.append(html.escape(item["name"]))
            elif isinstance(item, str):
                item_names.append(html.escape(item))
        if item_names:
            return ", ".join(item_names)
    elif isinstance(items, str):
        return html.escape(items)

    return default


def enterprise_vector_search(query):
    try:
        docs = enterprise_vector_store.similarity_search(query, k=4)
        if not docs:
            return "<p>No results found.</p>"

        formatted_results = []
        for doc in docs:
            meta = doc.metadata
            enterprise_id = meta.get("enterprise_id")
            enterprise_details = get_enterprise_details(enterprise_id)

            if enterprise_details:
                # Format categories/industries
                categories_html = format_list_items(
                    enterprise_details.get("categories")
                )

                # Format addresses
                addresses_html = "Not specified"
                if enterprise_details.get("addresses"):
                    address_list = []
                    for addr in enterprise_details.get("addresses", []):
                        if isinstance(addr, dict) and "mixedAddress" in addr:
                            address_list.append(html.escape(addr["mixedAddress"]))
                    if address_list:
                        addresses_html = "; ".join(address_list)

                # Format websites/contact
                websites_html = "Not specified"
                if enterprise_details.get("websites"):
                    website_list = []
                    for website in enterprise_details.get("websites", []):
                        if isinstance(website, dict) and "url" in website:
                            website_list.append(html.escape(website["url"]))
                    if website_list:
                        websites_html = "; ".join(website_list)

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
                        ("Industries", categories_html),
                        (
                            "Organization Type",
                            html.escape(
                                enterprise_details.get("organizationType")
                                or "Not specified"
                            ),
                        ),
                    ]
                )

                # Info sections
                addresses_section = create_enterprise_info_section(
                    "Addresses", addresses_html
                )
                contact_section = create_enterprise_info_section(
                    "Contact", websites_html
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

                formatted_enterprise = create_enterprise_card_wrapper(
                    enterprise_content
                )
                formatted_results.append(formatted_enterprise)
            else:
                # Format categories
                categories_html = (
                    [cat["category_name"] for cat in meta.get("categories")]
                    if meta.get("categories")
                    else []
                )

                # Format addresses
                addresses_html = "Not specified"
                if meta.get("addresses"):
                    address_list = []
                    for addr in meta.get("addresses", []):
                        if isinstance(addr, dict) and "mixed_address" in addr:
                            address_list.append(html.escape(addr["mixed_address"]))
                    if address_list:
                        addresses_html = "; ".join(address_list)

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
                        ("Industries", categories_html),
                        (
                            "Organization Type",
                            html.escape(
                                meta.get("organization_type") or "Not specified"
                            ),
                        ),
                    ]
                )

                # Info sections
                addresses_section = create_enterprise_info_section(
                    "Addresses", addresses_html
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

                formatted_enterprise = create_enterprise_card_wrapper(
                    enterprise_content
                )
                formatted_results.append(formatted_enterprise)

        return "\n".join(formatted_results)
    except Exception as e:
        return f"<p>Error searching enterprises: {html.escape(str(e))}</p>"


enterprise_tool = Tool(
    name="enterprise_tool",
    func=enterprise_vector_search,
    description="Use this tool to search for enterprise data. The input should be a string of text.",
)
