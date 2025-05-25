from typing import List
from langchain_core.documents import Document
from fastapi import APIRouter
from app.models import Job, Enterprise
from app.utils import clean_html, format_salary
from app.services.preprocess import preprocess_text
from app.vectorstore import job_vector_store, enterprise_vector_store

embedding_router = APIRouter(prefix="/embedding", tags=["embedding"])


def create_job_content(job_info: Job) -> str:
    try:
        categories = (
            [cat.categoryName for cat in job_info.categories]
            if hasattr(job_info, "categories") and isinstance(job_info.categories, list)
            else []
        )
        tags = (
            [tag.name for tag in job_info.tags]
            if hasattr(job_info, "tags") and isinstance(job_info.tags, list)
            else []
        )
        specializations = (
            [cat.categoryName for cat in job_info.specializations]
            if hasattr(job_info, "specializations")
            and isinstance(job_info.specializations, list)
            else []
        )
        locations = (
            ["In " + addr.country + " " + addr.city for addr in job_info.addresses]
            if job_info.addresses
            and isinstance(job_info.addresses, list)
            and len(job_info.addresses) > 0
            else []
        )

        skills_from_tags = ", ".join(tags)
        skills_from_requirements = clean_html(
            job_info.requirements if job_info.requirements else ""
        )
        skills_line = f"Skills: {skills_from_tags} {skills_from_requirements}".strip()
        all_keywords = (
            [preprocess_text(job_info.name)]
            + [preprocess_text(cat) for cat in categories]
            + [preprocess_text(spec) for spec in specializations]
            + [preprocess_text(tag) for tag in tags]
        )
        all_keywords += [
            preprocess_text(skills_from_requirements),
            preprocess_text(job_info.description if job_info.description else ""),
        ]
        all_keywords += [preprocess_text(loc) for loc in locations]
        keyword_blob = " ".join([w for w in all_keywords if w])
        repeated_categories = (
            (", ".join(categories) + " ") * 4 if len(categories) > 0 else ""
        )
        repeated_specializations = (
            (", ".join(specializations) + " ") * 4 if len(specializations) > 0 else ""
        )
        repeated_tags = (", ".join(tags) + " ") * 3 if len(tags) > 0 else ""
        repeated_location = (
            (" ;".join(locations) + "; ") * 2 if locations else "Remote "
        )
        content = f"""
        Priority Points: {job_info.boostedJob.pointsUsed if job_info.boostedJob else 0};
        Title: {job_info.name or ''}
        Industries: {repeated_categories}
        Majorities/Major: {repeated_specializations}
        Related Keywords: {repeated_tags};
        Location: {repeated_location};
        {keyword_blob}
        Company: {job_info.enterprise.name if job_info.enterprise else ''}
        Type: {job_info.type}
        {skills_line}
        Requirements: {clean_html(job_info.requirements) or ""}
        Experience: {job_info.experience} years
        Education: {job_info.education}
        Deadline: {job_info.deadline}
        Salary Range: {job_info.lowestWage} - {job_info.highestWage} (USD)
        Company Type: {job_info.enterprise.organizationType if job_info.enterprise else ''}
        """
        return content
    except Exception as e:
        print(f"Error creating job content: {e}")
        return f"Title: {getattr(job_info, 'name', 'Unknown Job')}"


def create_job_metadata(job_info: Job) -> dict:
    try:
        categories = (
            [cat.categoryName for cat in job_info.categories]
            if hasattr(job_info, "categories") and isinstance(job_info.categories, list)
            else []
        )
        tags = (
            [tag.name for tag in job_info.tags]
            if hasattr(job_info, "tags") and isinstance(job_info.tags, list)
            else []
        )
        specializations = (
            [cat.categoryName for cat in job_info.specializations]
            if hasattr(job_info, "specializations")
            and isinstance(job_info.specializations, list)
            else []
        )
        locations = (
            [addr.mixedAddress for addr in job_info.addresses]
            if job_info.addresses
            and isinstance(job_info.addresses, list)
            and len(job_info.addresses) > 0
            else []
        )

        metadata = {
            "job_id": str(job_info.jobId),
            "job_name": job_info.name if hasattr(job_info, "name") else "",
            "company": (
                job_info.enterprise.name
                if job_info.enterprise and hasattr(job_info.enterprise, "name")
                else ""
            ),
            "experience": job_info.experience if hasattr(job_info, "experience") else 0,
            "education": job_info.education if hasattr(job_info, "education") else "",
            "status": job_info.status if hasattr(job_info, "status") else "",
            "salary_range": {
                "min": job_info.lowestWage if hasattr(job_info, "lowestWage") else 0,
                "max": job_info.highestWage if hasattr(job_info, "highestWage") else 0,
            },
            "categories": categories,
            "tags": tags,
            "specializations": specializations,
            "deadline": (
                str(job_info.deadline)
                if hasattr(job_info, "deadline") and job_info.deadline
                else ""
            ),
            "is_premium": (
                job_info.enterprise.isPremium
                if job_info.enterprise
                and hasattr(job_info.enterprise, "isPremium")
                and job_info.enterprise.isPremium
                else False
            ),
            "is_trial": (
                job_info.enterprise.isTrial
                if job_info.enterprise
                and hasattr(job_info.enterprise, "isTrial")
                and job_info.enterprise.isTrial
                else False
            ),
            "organization_type": (
                job_info.enterprise.organizationType
                if job_info.enterprise
                and hasattr(job_info.enterprise, "organizationType")
                else ""
            ),
            "enterprise_status": (
                job_info.enterprise.status
                if job_info.enterprise and hasattr(job_info.enterprise, "status")
                else ""
            ),
            "locations": locations,
            "job_type": job_info.type if hasattr(job_info, "type") else "",
            "description": (
                job_info.description if hasattr(job_info, "description") else ""
            ),
            "responsibility": (
                job_info.responsibility if hasattr(job_info, "responsibility") else ""
            ),
            "requirement": (
                job_info.requirements if hasattr(job_info, "requirements") else ""
            ),
            "job_benefits": (
                job_info.enterpriseBenefits
                if hasattr(job_info, "enterpriseBenefits")
                else ""
            ),
            "points_used": (
                job_info.boostedJob.pointsUsed
                if hasattr(job_info, "boostedJob") and job_info.boostedJob
                else 0
            ),
        }
        return metadata
    except Exception as e:
        print(f"Error creating job metadata: {e}")
        return {
            "job_id": str(getattr(job_info, "jobId", "unknown")),
            "job_name": getattr(job_info, "name", "Unknown Job"),
            "error": str(e),
        }


def create_job_document(job_info: Job) -> Document:
    try:
        # Create content for embedding
        content = create_job_content(job_info)

        # Create metadata
        metadata = create_job_metadata(job_info)

        # Create and return the document
        return Document(page_content=content, metadata=metadata)
    except Exception as e:
        print(f"Error creating job document: {e}")
        raise e


def create_enterprise_content(enterprise_info: Enterprise) -> str:
    try:
        locations = (
            [
                "In " + addr.country + ", " + addr.city
                for addr in enterprise_info.addresses
            ]
            if enterprise_info.addresses
            and isinstance(enterprise_info.addresses, list)
            and len(enterprise_info.addresses) > 0
            else []
        )

        categories = (
            [cat.categoryName for cat in enterprise_info.categories]
            if enterprise_info.categories
            and isinstance(enterprise_info.categories, list)
            and len(enterprise_info.categories) > 0
            else []
        )

        repeated_addresses = ("; ".join(locations) + " ") * 3
        repeated_industries = (", ".join(categories) + " ") * 3

        content = f"""
        Company Name: {enterprise_info.name or 'Unknown'}
        Company Description: {clean_html(enterprise_info.description) or 'Unknown'}
        Company Vision: {clean_html(enterprise_info.companyVision) or 'Unknown'}
        Founded In: {str(enterprise_info.foundedIn) if enterprise_info.foundedIn else 'Unknown'}
        Organization Type: {enterprise_info.organizationType or 'Unknown'}
        Team Size: {enterprise_info.teamSize or 'Unknown'}
        Status: {enterprise_info.status or 'Unknown'}
        Is Premium: {enterprise_info.isPremium or enterprise_info.isTrial}
        Industries/Fields: {repeated_industries}
        Addresses: {repeated_addresses}
        """
        return content
    except Exception as e:
        print(f"Error creating enterprise content: {e}")
        return f"Company Name: {getattr(enterprise_info, 'name', 'Unknown Company')}"


def create_enterprise_metadata(enterprise_info: Enterprise) -> dict:
    try:
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
                        "country": addr.country,
                        "city": addr.city,
                        "mixed_address": addr.mixedAddress,
                    }
                    for addr in enterprise_info.addresses
                ]
                if isinstance(enterprise_info.addresses, list)
                and len(enterprise_info.addresses) > 0
                else []
            ),
        }

        return metadata
    except Exception as e:
        print(f"Error creating enterprise metadata: {e}")
        return {
            "enterprise_id": str(getattr(enterprise_info, "enterpriseId", "unknown")),
            "name": getattr(enterprise_info, "name", "Unknown Company"),
            "error": str(e),
        }


def create_enterprise_document(enterprise_info: Enterprise) -> Document:
    try:
        # Create content for embedding
        content = create_enterprise_content(enterprise_info)

        # Create metadata
        metadata = create_enterprise_metadata(enterprise_info)

        # Create and return the document
        return Document(page_content=content, metadata=metadata)
    except Exception as e:
        print(f"Error creating enterprise document: {e}")
        raise e


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


@embedding_router.delete("/job/{job_id}")
def delete_embedding_job(job_id: str):
    try:
        job_vector_store.delete([f"job-{job_id}"])
        return {"message": "Job embedding deleted successfully"}
    except Exception as e:
        return {"error": str(e)}


@embedding_router.delete("/job")
def delete_embedding_jobs(job_ids: List[str]):
    try:
        job_vector_store.delete([f"job-{job_id}" for job_id in job_ids])
        return {"message": "Job embedding deleted successfully"}
    except Exception as e:
        return {"error": str(e)}


@embedding_router.delete("/enterprise/{enterprise_id}")
def delete_embedding_enterprise(enterprise_id: str):
    try:
        job_vector_store.delete([f"enterprise-{enterprise_id}"])
        return {"message": "Enterprise embedding deleted successfully"}
    except Exception as e:
        return {"error": str(e)}


@embedding_router.delete("/enterprise")
def delete_embedding_enterprises(enterprise_ids: List[str]):
    try:
        job_vector_store.delete(
            [f"enterprise-{enterprise_id}" for enterprise_id in enterprise_ids]
        )
        return {"message": "Enterprise embedding deleted successfully"}
    except Exception as e:
        return {"error": str(e)}
