from langchain_core.documents import Document
from fastapi import APIRouter
from app.models import Job, Enterprise
from utils import clean_html, format_salary
from app.vectorstore import job_vector_store, enterprise_vector_store

embedding_router = APIRouter(prefix="/embedding", tags=["embedding"])


def create_job_content(job_info: Job) -> str:
    content = f"""
        Job Title: {job_info.name or 'Unknown'}
        Type: {job_info.type or 'Unknown'}
        Company: {job_info.enterprise.name or 'Unknown enterprise'}
        Company Type: {job_info.enterprise.organizationType or 'Unknown'}
        Company Status: {job_info.enterprise.status or 'Unknown'}
        Location: {"; ".join([addr.mixedAddress for addr in job_info.addresses]) if len(job_info.addresses) > 0 else 'Unknown'}
        Salary Range: {format_salary(job_info.lowestWage, job_info.highestWage)}
        Education: {job_info.education or 'Unknown'}
        Experience: {job_info.experience or 'Unknown'} years
        Deadline: {str(job_info.deadline)}
        Status: {job_info.status}
        Categories: {", ".join([cat.categoryName for cat in job_info.categories]) if len(job_info.categories) > 0 else 'Unknown'}
        Specializations: {", ".join([cat.categoryName for cat in job_info.specializations]) if len(job_info.specializations) > 0 else 'Unknown'}
        Tags: {", ".join([tag.name for tag in job_info.tags]) if len(job_info.tags) > 0 else 'Unknown'}
        Description: {clean_html(job_info.description)}
        Responsibilities: {clean_html(job_info.responsibility)}
        Requirements: {clean_html(job_info.requirements)}
        Benefits: {clean_html(job_info.enterpriseBenefits)}
        Priority: {job_info.enterprise.isPremium if job_info.enterprise.isPremium else job_info.enterprise.isTrial}
        Boosted Points Used: {job_info.boostedJob.pointsUsed if job_info.boostedJob else 0}
        """
    return content


def create_job_metadata(job_info: Job) -> dict:
    metadata = {
        "job_id": str(job_info.jobId),
        "job_name": job_info.name or "Unknown",
        "company": job_info.enterprise.name or "Unknown enterprise",
        "experience": job_info.experience or 0,
        "education": job_info.education or "",
        "status": job_info.status or "",
        "salary_range": {
            "min": job_info.lowestWage or 0,
            "max": job_info.highestWage or 0,
        },
        "categories": [cat.categoryName for cat in job_info.categories],
        "tags": [tag.name for tag in job_info.tags],
        "specializations": [cat.categoryName for cat in job_info.specializations],
        "deadline": str(job_info.deadline) if job_info.deadline else "",
        "is_premium": (
            job_info.enterprise.isPremium if job_info.enterprise.isPremium else False
        ),
        "is_trial": (
            job_info.enterprise.isTrial if job_info.enterprise.isTrial else False
        ),
        "organization_type": job_info.enterprise.organizationType or "",
        "enterprise_status": job_info.enterprise.status or "",
        "locations": [addr.mixedAddress for addr in job_info.addresses],
        "job_type": job_info.type or "",
        "description": job_info.description or "",
        "responsibility": job_info.responsibility or "",
        "requirement": job_info.requirements or "",
        "job_benefits": job_info.enterpriseBenefits or "",
        "points_used": job_info.boostedJob.pointsUsed if job_info.boostedJob else 0,
    }

    return metadata


def create_job_document(job_info: Job) -> Document:
    # Create content for embedding
    content = create_job_content(job_info)

    # Create metadata
    metadata = create_job_metadata(job_info)

    # Create and return the document
    return Document(page_content=content, metadata=metadata)


def create_enterprise_content(enterprise_info: Enterprise) -> str:
    content = f"""
    Company Name: {enterprise_info.name or 'Unknown'}
    Company Description: {enterprise_info.description or 'Unknown'}
    Company Vision: {enterprise_info.companyVision or 'Unknown'}
    Founded In: {str(enterprise_info.foundedIn) if enterprise_info.foundedIn else 'Unknown'}
    Organization Type: {enterprise_info.organizationType or 'Unknown'}
    Team Size: {enterprise_info.teamSize or 'Unknown'}
    Status: {enterprise_info.status or 'Unknown'}
    Is Premium: {enterprise_info.isPremium or enterprise_info.isTrial}
    Categories: {", ".join([cat.categoryName for cat in enterprise_info.categories]) if len(enterprise_info.categories) > 0 else "Not specified"}
    Addresses: {"; ".join([addr.mixedAddress for addr in enterprise_info.addresses]) if len(enterprise_info.addresses) > 0  else "Not specified"}
    """
    return content


def create_enterprise_metadata(enterprise_info: Enterprise) -> dict:
    metadata = {
        "enterprise_id": str(enterprise_info.enterpriseId),
        "name": enterprise_info.name or "Unknown",
        "description": enterprise_info.description or "Unknown",
        "company_vision": enterprise_info.companyVision or "Unknown",
        "logo_url": enterprise_info.logoUrl or "Unknown",
        "founded_in": (
            str(enterprise_info.foundedIn) if enterprise_info.foundedIn else ""
        ),
        "organization_type": enterprise_info.organizationType or "Unknown",
        "team_size": enterprise_info.teamSize or "Unknown",
        "status": enterprise_info.status or "Unknown",
        "is_premium": enterprise_info.isPremium or enterprise_info.isTrial,
        "categories": (
            [
                {
                    "category_id": str(cat.categoryId),
                    "category_name": cat.categoryName,
                }
                for cat in enterprise_info.categories
            ]
            if len(enterprise_info.categories) > 0
            else []
        ),
        "addresses": (
            [
                {
                    "address_id": str(addr.addressId),
                    "mixed_address": addr.mixedAddress,
                }
                for addr in enterprise_info.addresses
            ]
            if isinstance(enterprise_info.addresses, list)
            else []
        ),
    }

    return metadata


def create_enterprise_document(enterprise_info: Enterprise) -> Document:
    # Create content for embedding
    content = create_enterprise_content(enterprise_info)

    # Create metadata
    metadata = create_enterprise_metadata(enterprise_info)

    # Create and return the document
    return Document(page_content=content, metadata=metadata)


@embedding_router.post("/job")
def create_embedding_job(job_info: Job):
    try:
        document = create_job_document(job_info)
        job_vector_store.add_documents([document], ids=[f"job-{job_info.jobId}"])
        return {"message": "Job embedding updated successfully"}
    except Exception as e:
        return {"error": str(e)}


@embedding_router.put("/job/{job_id}")
def update_embedding_job(job_id: str, job_info: Job):
    try:
        job_info.jobId = job_id
        document = create_job_document(job_info)
        job_vector_store.add_documents([document], ids=[f"job-{job_id}"])

        return {"message": "Job embedding updated successfully"}
    except Exception as e:
        return {"error": str(e)}


@embedding_router.put("/enterprise/{enterprise_id}")
def update_embedding_enterprise(enterprise_id: str, enterprise_info: Enterprise):
    try:
        enterprise_info.enterpriseId = enterprise_id
        document = create_enterprise_document(enterprise_info)

        enterprise_vector_store.add_documents(
            [document], ids=[f"enterprise-{enterprise_id}"]
        )
        return {"message": "Enterprise embedding updated successfully"}
    except Exception as e:
        return {"error": str(e)}


@embedding_router.post("/enterprise")
def create_embedding_enterprise(enterprise_info: Enterprise):
    try:
        document = create_enterprise_document(enterprise_info)

        enterprise_vector_store.add_documents(
            [document], ids=[f"enterprise-{enterprise_info.enterpriseId}"]
        )
        return {"message": "Enterprise embedding updated successfully"}
    except Exception as e:
        return {"error": str(e)}
