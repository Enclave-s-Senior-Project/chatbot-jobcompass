from pydantic import BaseModel, Field
import requests
import os
from dotenv import load_dotenv
import logging
from typing import List, Optional

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

        results = response.json()
        data = results.get("payload", {}).get("value", {})
        return data
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching job details for job ID {job_id}: {str(e)}")
        return None


class JobCategory:
    isActive: bool
    categoryId: str
    categoryName: str


class EnterpriseAddress:
    createdAt: str
    updatedAt: str
    isActive: bool
    addressId: str
    country: str
    city: str
    street: str
    zipCode: str
    mixedAddress: str


class Website:
    websiteId: str
    url: str
    type: str
    isActive: bool


class Job:
    createdAt: str
    updatedAt: str
    isActive: bool
    jobId: str
    name: str
    lowestWage: Optional[float] = None
    highestWage: Optional[float] = None
    description: Optional[str] = None
    responsibility: Optional[str] = None
    type: Optional[str] = None
    experience: Optional[int] = None
    deadline: Optional[str] = None
    introImg: Optional[str] = None
    status: str
    education: Optional[str] = None
    isBoost: bool = False
    enterpriseBenefits: Optional[str] = None
    requirements: Optional[str] = None


class EnterpriseResponse:
    """Enterprise data response model from JobCompass API"""

    createdAt: str
    updatedAt: str
    isActive: bool
    enterpriseId: str
    name: str
    email: str
    phone: Optional[str]
    description: Optional[str]
    benefit: Optional[str]
    companyVision: Optional[str]
    logoUrl: Optional[str]
    boostLimit: int = Field(0)
    backgroundImageUrl: Optional[str]
    foundedIn: Optional[str]
    organizationType: Optional[str]
    teamSize: Optional[str]
    status: str
    categories: List[JobCategory] = []
    bio: Optional[str]
    isPremium: bool = Field(False)
    isTrial: bool = Field(False)
    websites: List[Website] = []
    addresses: List[EnterpriseAddress] = []
    jobs: List[Job] = []


def get_enterprise_details(enterprise_id) -> Optional[EnterpriseResponse]:
    """
    Fetch detailed enterprise information from the JobCompass API.

    Args:
        enterprise_id: The ID of the enterprise to fetch details for

    Returns:
        dict: Full enterprise details or None if request failed
    """
    api_url = os.getenv("JOB_API_URL")
    if not api_url:
        logger.error("JOB_API_URL environment variable not set")
        return None

    try:
        url = f"{api_url}/enterprise/{enterprise_id}"
        headers = {
            "Content-Type": "application/json",
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        results = response.json()
        data = results.get("payload", {}).get("value", {})

        return data
    except requests.exceptions.RequestException as e:
        logger.error(
            f"Error fetching enterprise details for enterprise ID {enterprise_id}: {str(e)}"
        )
        return None


def get_profile_details(profile_id):
    """
    Fetch detailed profile information from the JobCompass API.

    Args:
        profile_id: The ID of the profile to fetch details for

    Returns:
        dict: Full profile details or None if request failed
    """
    api_url = os.getenv("JOB_API_URL")
    if not api_url:
        logger.error("JOB_API_URL environment variable not set")
        return None

    try:
        url = f"{api_url}/user/{profile_id}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('JOB_API_TOKEN')}",
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        results = response.json()
        data = results.get("payload", {}).get("value", {})

        return data
    except requests.exceptions.RequestException as e:
        logger.error(
            f"Error fetching profile details for profile ID {profile_id}: {str(e)}"
        )
        return None
