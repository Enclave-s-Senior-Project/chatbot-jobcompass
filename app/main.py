import nltk
import logging
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from .routers import chat_router, embedding_router, suggest_router
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import pandas as pd
from pydantic import BaseModel
from .routers import chat_router
from app.services.preprocess import preprocess_text
from app.config.config import DB_CONFIG_PRIMARY, DATASET_PATH, OUTPUT_PATH
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import psycopg2
from contextlib import asynccontextmanager
from app.services.job_service import compute_related_jobs
import os
import pickle
from app.config.config import MODEL_PATH

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NLTK Setup (called before imports)
def setup_nltk_data():
    """Download NLTK data only if not already present."""
    nltk_data = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
        ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng')
    ]
    for resource_path, resource_name in nltk_data:
        try:
            nltk.data.find(resource_path)
            logger.info(f"NLTK resource {resource_name} already exists.")
        except LookupError:
            logger.info(f"Downloading NLTK resource {resource_name}...")
            nltk.download(resource_name, quiet=True)

# Initialize NLTK resources before imports
logger.info("Initializing NLTK resources...")
setup_nltk_data()

load_dotenv()

# Pydantic model for user input
class UserInput(BaseModel):
    text: str
    Industry: str = ""
    Career_Level: str = ""
    Job_Type: str = ""

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("Running job_training once on startup...")
    await run_job_training()
    logger.info("Initial job_training run completed.")

    logger.info("Starting scheduler...")
    scheduler.add_job(
        run_job_training,
        trigger=CronTrigger(hour=1, minute=0),
        id="job_training",
        replace_existing=True
    )
    scheduler.start()
    logger.info("Scheduler started.")
    
    try:
        yield
    finally:
        # Shutdown logic
        logger.info("Shutting down scheduler...")
        scheduler.shutdown()
        logger.info("Scheduler shut down.")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.include_router(chat_router)
app.include_router(embedding_router)
app.include_router(suggest_router)

# Fetch Jobs Function
def fetch_jobs():
    logger.info("Starting fetch_jobs: Connecting to primary database...")
    conn = psycopg2.connect(**DB_CONFIG_PRIMARY)
    cursor = conn.cursor()
    
    try:
        query = """
        SELECT 
            j.job_id,
            j.name AS job_title,
            j.description AS job_description,
            j.requirement AS job_requirements,
            j.enterprise_benefits AS benefits,
            j.type AS job_type,
            j.experience AS years_experience,
            j.education AS career_level,
            e.name AS name_company,
            e.description AS company_overview,
            e.team_size AS company_size,
            COALESCE((
                SELECT STRING_AGG(a.city || ', ' || a.country, '; ')
                FROM addresses a
                JOIN enterprise_addresses ea ON a.address_id = ea.address_id
                WHERE ea.enterprise_id = e.enterprise_id
            ), 'Not specified') AS company_address,
            COALESCE((
                SELECT STRING_AGG(a.city || ', ' || a.country, '; ')
                FROM addresses a
                JOIN job_addresses ja ON a.address_id = ja.address_id
                WHERE ja.job_id = j.job_id
            ), 'Not specified') AS job_address,
            COALESCE((
                SELECT STRING_AGG(c.category_name, ', ')
                FROM categories c
                JOIN job_categories jc ON c.category_id = jc.category_id
                WHERE jc.job_id = j.job_id
            ), '') AS categories,
            COALESCE((
                SELECT STRING_AGG(c.category_name, ', ')
                FROM categories c
                JOIN job_specializations js ON c.category_id = js.category_id
                WHERE js.job_id = j.job_id
            ), '') AS specializations
        FROM jobs j
        JOIN enterprises e ON j.enterprise_id = e.enterprise_id
        WHERE j.status = 'OPEN'
        """
        logger.info("Executing SQL query to fetch jobs...")
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        logger.info(f"Fetched {len(rows)} jobs from database")
        df = pd.DataFrame(rows, columns=columns)
        
        df['url_job'] = df['job_id'].apply(lambda x: f"https://job-compass.bunkid.online/single-job/{x}")
        df['industry'] = df.apply(
            lambda x: ", ".join(filter(None, [x['categories'], x['specializations']])),
            axis=1
        )
        df['industry'] = df['industry'].replace("", "Not specified")
        df = df.drop(columns=['categories', 'specializations'])
        
        df = df.rename(columns={
            "job_id": "JobID",
            "url_job": "URL Job",
            "job_title": "Job Title",
            "name_company": "Name Company",
            "company_overview": "Company Overview",
            "company_size": "Company Size",
            "company_address": "Company Address",
            "job_description": "Job Description",
            "job_requirements": "Job Requirements",
            "benefits": "Benefits",
            "job_address": "Job Address",
            "job_type": "Job Type",
            "career_level": "Career Level",
            "years_experience": "Years of Experience",
            "industry": "Industry"
        })
        
        output_columns = [
            "JobID", "URL Job", "Job Title", "Name Company", "Company Overview",
            "Company Size", "Company Address", "Job Description", "Job Requirements",
            "Benefits", "Job Address", "Job Type", "Career Level", "Years of Experience", "Industry"
        ]
        df = df[output_columns]
        
        df.to_csv(DATASET_PATH, index=False)
        logger.info(f"Data saved to {DATASET_PATH} with {len(df)} rows")
        return df
    finally:
        cursor.close()
        conn.close()
        logger.info("Database connection closed")


async def run_job_training():
    try:
        logger.info("Starting job training cron job...")
        setup_nltk_data()
        
        # Fetch dữ liệu từ database
        df = fetch_jobs()
        if df.empty:
            logger.warning("No jobs fetched from database")
            return
        
        logger.info(f"Fetched DataFrame with {len(df)} jobs")
        
        # Thực hiện train mô hình
        result = compute_related_jobs(df)
        
        # Kiểm tra kết quả trả về từ compute_related_jobs
        if isinstance(result, tuple):
            # Giả sử compute_related_jobs trả về một tuple
            if len(result) >= 3:
                df, vectorizer, tfidf_matrix = result[:3]
                hybrid_sim = result[3] if len(result) > 3 else None
            else:
                logger.error("compute_related_jobs returned insufficient values")
                raise ValueError("compute_related_jobs did not return expected values")
        else:
            # Nếu chỉ trả về df, vectorizer, tfidf_matrix mà không có hybrid_sim
            df, vectorizer, tfidf_matrix = result
            hybrid_sim = None
        
        # Lưu trữ các biến vào app.state
        app.state.vectorizer = vectorizer
        app.state.tfidf_matrix = tfidf_matrix
        app.state.df = df
        if hybrid_sim is not None:
            app.state.hybrid_sim = hybrid_sim
        
        # Lưu mô hình mới vào file
        model_file_path = os.path.join(MODEL_PATH, 'job_data.pkl')  # Sử dụng os.path.join để đảm bảo đường dẫn đúng
        logger.info(f"Saving new model to {model_file_path}...")
        data_to_save = {
            'df': df,
            'vectorizer': vectorizer,
            'tfidf_matrix': tfidf_matrix
        }
        if hybrid_sim is not None:
            data_to_save['hybrid_sim'] = hybrid_sim
        
        os.makedirs(MODEL_PATH, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại
        with open(model_file_path, 'wb') as f:
            pickle.dump(data_to_save, f)
        logger.info(f"New model saved successfully to {model_file_path}")
        
        logger.info("Job training cron job completed successfully.")
    except Exception as e:
        logger.error(f"Error in job training cron job: {str(e)}", exc_info=True)
        raise

scheduler = AsyncIOScheduler()
