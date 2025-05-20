from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import requests
import logging
import ast
from app.config.config import OUTPUT_PATH,JOB_API_URL
from app.services.job_service import get_related_jobs_for_ids, get_related_jobs_for_multiple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

suggest_router = APIRouter(prefix="/suggest", tags=["suggest"])

@suggest_router.get("/related-jobs/{job_id}")
async def get_top_related_jobs_for_user(job_id: str):
    try:
        # Read the CSV file
        logger.info(f"Reading related jobs from {OUTPUT_PATH}")
        try:
            df = pd.read_csv(OUTPUT_PATH)
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail=f"Related jobs CSV file not found at {OUTPUT_PATH}")
        
        # Find the job with the given job_id
        job_row = df[df['JobID'] == job_id]
        
        if job_row.empty:
            raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found")
        
        # Parse the related_jobs list (stored as a string in CSV)
        related_jobs_ids_str = job_row.iloc[0]['related_jobs']
        try:
            related_jobs_ids = ast.literal_eval(related_jobs_ids_str)
            if not isinstance(related_jobs_ids, list):
                raise ValueError("related_jobs is not a list")
        except (ValueError, SyntaxError) as e:
            logger.error(f"Error parsing related_jobs for job_id {job_id}: {str(e)}")
            raise HTTPException(status_code=500, detail="Invalid related_jobs data in CSV")
        
        if not related_jobs_ids:
            return {"related_jobs": []}
        
        # Make request to external service
        try:
            response = requests.post(
                "http://localhost:3001/api/v1/job/related-jobs",
                json={"related_jobs": related_jobs_ids},
                headers={"Accept": "*/*", "Content-Type": "application/json"}
            )
            response.raise_for_status()  # Raise exception for bad status codes
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error calling related jobs service for job_id {job_id}: {str(e)}")
            # Fallback to CSV data if external service fails
            return {"related_jobs": related_jobs_ids}
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error fetching related jobs for job_id {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching related jobs: {str(e)}")

class JobSuggestionsInput(BaseModel):
    job_ids: list[str]
    num_suggestions: int = 3  # Default to 3 suggestions

@suggest_router.post("/job-suggestions")
async def get_job_suggestions(input_data: JobSuggestionsInput):
    """
    Get job suggestions for a list of job IDs.
    - For 2 or more job IDs, returns a list of job details most similar to the combined set.
    - For 1 job ID, returns a dictionary with suggestions for that job.
    
    Args:
        input_data: Object containing a list of job IDs and optional number of suggestions.
    
    Returns:
        Dictionary with job suggestions and a message.
    """
    try:
        logger.info(f"Fetching suggestions for job IDs: {input_data.job_ids}")
        if not input_data.job_ids:
            raise HTTPException(status_code=400, detail="Job IDs list cannot be empty")
        
        # For 2 or more job IDs, get combined suggestions
        if len(input_data.job_ids) >= 2:
            # Get combined job IDs
            related_job_ids = get_related_jobs_for_multiple(
                job_ids=input_data.job_ids,
                num_related=input_data.num_suggestions
            )
            
            if not related_job_ids:
                logger.warning("No related jobs found for the provided job IDs")
                return {
                    "job_suggestions": [],
                    "message": f"No suggestions found for {len(input_data.job_ids)} job IDs"
                }
            
            # Fetch job details from external API
            try:
                response = requests.post(
                    "http://localhost:3001/api/v1/job/related-jobs",
                    json={"related_jobs": related_job_ids},
                    headers={"Accept": "*/*", "Content-Type": "application/json"}
                )
                response.raise_for_status()
                job_details = response.json()
                # Ensure job_details is a list; adjust based on actual API response
                if isinstance(job_details, dict) and "related_jobs" in job_details:
                    job_details = job_details["related_jobs"]
                return {
                    "job_suggestions": job_details,
                    "message": f"Suggestions retrieved for {len(input_data.job_ids)} job IDs"
                }
            except requests.RequestException as e:
                logger.error(f"Error calling related jobs service: {str(e)}")
                # Fallback to returning job IDs
                return {
                    "job_suggestions": related_job_ids,
                    "message": f"Suggestions retrieved for {len(input_data.job_ids)} job IDs (API fallback)"
                }
        
        # For 1 job ID, get individual suggestions
        else:
            related_jobs = get_related_jobs_for_ids(
                job_ids=input_data.job_ids,
                num_related=input_data.num_suggestions
            )
            
            # Check if the job ID was found
            found_jobs = [job_id for job_id in input_data.job_ids if related_jobs.get(job_id)]
            if not found_jobs:
                raise HTTPException(status_code=404, detail="No valid job IDs found")
            
            # Fetch job details for each job ID's suggestions
            result = {}
            for job_id, related_ids in related_jobs.items():
                if not related_ids:
                    result[job_id] = []
                    continue
                try:
                    response = requests.post(
                        "http://localhost:3001/api/v1/job/related-jobs",
                        json={"related_jobs": related_ids},
                        headers={"Accept": "*/*", "Content-Type": "application/json"}
                    )
                    response.raise_for_status()
                    job_details = response.json()
                    if isinstance(job_details, dict) and "related_jobs" in job_details:
                        job_details = job_details["related_jobs"]
                    result[job_id] = job_details
                except requests.RequestException as e:
                    logger.error(f"Error calling related jobs service for job_id {job_id}: {str(e)}")
                    result[job_id] = related_ids  # Fallback to IDs
            
            return {
                "job_suggestions": result,
                "message": f"Suggestions retrieved for {len(found_jobs)}/{len(input_data.job_ids)} job ID"
            }
    
    except ValueError as ve:
        logger.error(f"Validation error in /job-suggestions: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except FileNotFoundError as fnfe:
        logger.error(f"Model file error in /job-suggestions: {str(fnfe)}")
        raise HTTPException(status_code=500, detail="Model file not found")
    except Exception as e:
        logger.error(f"Error in /job-suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
