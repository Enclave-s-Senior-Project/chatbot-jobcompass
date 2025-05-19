import nltk
import logging
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from .routers import chat_router, embedding_router
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

# Compute Related Jobs Function
# def compute_related_jobs(df):
#     logger.info("Starting compute_related_jobs: Preprocessing text data...")
#     df['processed_text'] = df.apply(lambda row: ' '.join([
#         preprocess_text(row['Job Title']) * 6,
#         preprocess_text(row['Job Description']),
#         preprocess_text(row['Job Requirements']),
#         preprocess_text(row['Industry']) * 8,
#         preprocess_text(row['Career Level']) * 2,
#         preprocess_text(row['Job Type'])
#     ]), axis=1)

#     logger.info("Computing TF-IDF vectors...")
#     vectorizer = TfidfVectorizer(
#         max_features=5000,
#         ngram_range=(1, 2),
#         min_df=3,
#         max_df=0.7,
#         sublinear_tf=True
#     )
#     tfidf_matrix = vectorizer.fit_transform(df['processed_text'])
#     logger.info(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")

#     logger.info("Calculating cosine similarity...")
#     cosine_sim_text = cosine_similarity(tfidf_matrix)
#     logger.info(f"Text Cosine Similarity Matrix Shape: {cosine_sim_text.shape}")

#     def categorical_similarity(row1, row2):
#         score = 0
#         if row1['Industry'] == row2['Industry']:
#             score += 0.5
#         if row1['Career Level'] == row2['Career Level']:
#             score += 0.3
#         if row1['Job Type'] == row2['Job Type']:
#             score += 0.2
#         return score

#     num_jobs = len(df)
#     hybrid_sim = np.zeros((num_jobs, num_jobs))
#     for i in range(num_jobs):
#         for j in range(num_jobs):
#             text_sim = cosine_sim_text[i, j]
#             cat_sim = categorical_similarity(df.iloc[i], df.iloc[j])
#             hybrid_sim[i, j] = 0.7 * text_sim + 0.3 * cat_sim
#     logger.info(f"Hybrid Similarity Matrix Shape: {hybrid_sim.shape}")

#     def get_related_job_ids(job_index, num_jobs=3):
#         sim_scores = hybrid_sim[job_index]
#         related_indices = np.argsort(sim_scores)[::-1][1:num_jobs+1]
#         related_job_ids = df.iloc[related_indices]['JobID'].tolist()
#         return related_job_ids

#     logger.info("Generating related jobs...")
#     df['related_jobs'] = [get_related_job_ids(idx, num_jobs=3) for idx in range(len(df))]
#     df.to_csv(OUTPUT_PATH, index=False)
#     logger.info(f"Related jobs data saved to {OUTPUT_PATH} with {len(df)} rows")
#     return df, vectorizer, tfidf_matrix

# Job to Run Fetch and Compute
async def run_job_training():
    try:
        logger.info("Starting job training cron job...")
        setup_nltk_data()
        df = fetch_jobs()
        if df.empty:
            logger.warning("No jobs fetched from database")
            return
        logger.info(f"Fetched DataFrame with {len(df)} jobs")
        df, vectorizer, tfidf_matrix = compute_related_jobs(df)
        # Store vectorizer and tfidf_matrix globally for endpoint use
        app.state.vectorizer = vectorizer
        app.state.tfidf_matrix = tfidf_matrix
        app.state.df = df
        logger.info("Job training cron job completed successfully.")
    except Exception as e:
        logger.error(f"Error in job training cron job: {str(e)}", exc_info=True)
        raise

# Scheduler Setup
scheduler = AsyncIOScheduler()

# Endpoint for user input
@app.post("/related-jobs")
async def get_top_related_jobs_for_user(user_input: UserInput):
    try:
        if not hasattr(app.state, 'vectorizer') or not hasattr(app.state, 'tfidf_matrix') or not hasattr(app.state, 'df'):
            logger.error("Vectorizer, TF-IDF matrix, or DataFrame not initialized. Run job_training first.")
            return {"error": "Job data not initialized"}
        
        df = app.state.df
        vectorizer = app.state.vectorizer
        tfidf_matrix = app.state.tfidf_matrix

        processed_input = preprocess_text(user_input.text)
        user_tfidf = vectorizer.transform([processed_input])
        text_sim_scores = cosine_similarity(user_tfidf, tfidf_matrix)[0]

        hybrid_scores = []
        for idx in range(len(df)):
            cat_sim = 0
            if user_input.Industry and user_input.Industry == df.iloc[idx]['Industry']:
                cat_sim += 0.5
            if user_input.Career_Level and user_input.Career_Level == df.iloc[idx]['Career Level']:
                cat_sim += 0.3
            if user_input.Job_Type and user_input.Job_Type == df.iloc[idx]['Job Type']:
                cat_sim += 0.2
            hybrid_score = 0.7 * text_sim_scores[idx] + 0.3 * cat_sim
            hybrid_scores.append(hybrid_score)

        top_indices = np.argsort(hybrid_scores)[::-1][:3]
        top_scores = np.array(hybrid_scores)[top_indices]
        related_jobs = df.iloc[top_indices][['JobID', 'Job Title', 'Name Company', 'Industry', 'Career Level', 'Job Type']].copy()
        related_jobs['Similarity Score'] = top_scores
        return related_jobs.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error in get_top_related_jobs_for_user: {str(e)}", exc_info=True)
        return {"error": str(e)}