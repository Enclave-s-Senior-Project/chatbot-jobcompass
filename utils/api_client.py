import requests
import os
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


def get_job_details(job_id):
    """
    Fetch detailed job information from the JobCompass API.

    Args:
        job_id: The ID of the job to fetch details for

    Returns:
        dict: Full job details or None if request failed
    """
    api_url = os.getenv("JOB_API_URL")
    if not api_url:
        logger.error("JOB_API_URL environment variable not set")
        return None

    try:
        url = f"{api_url}/job/{job_id}"
        headers = {
            "Content-Type": "application/json",
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching job details for job ID {job_id}: {str(e)}")
        return None
