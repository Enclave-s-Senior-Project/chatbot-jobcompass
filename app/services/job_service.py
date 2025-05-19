import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .preprocess import preprocess_text
from app.config.config import OUTPUT_PATH, MODEL_PATH
import logging
import pickle
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_related_jobs(df):
    """
    Compute TF-IDF features and hybrid similarity matrix for jobs, and generate related jobs.
    
    Args:
        df (pd.DataFrame): DataFrame containing job data with 'Job Title', 'Job Description', etc.
    
    Returns:
        tuple: (df, vectorizer, tfidf_matrix, hybrid_sim)
    """
    logger.info("Starting compute_related_jobs: Preprocessing text data...")
    df['processed_text'] = df.apply(lambda row: ' '.join([
        preprocess_text(row['Job Title']) * 6,
        preprocess_text(row['Job Description']),
        preprocess_text(row['Job Requirements']),
        preprocess_text(row['Industry']) * 8,
        preprocess_text(row['Career Level']) * 2,
        preprocess_text(row['Job Type'])
    ]), axis=1)

    logger.info("Computing TF-IDF vectors...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.7,
        sublinear_tf=True
    )
    tfidf_matrix = vectorizer.fit_transform(df['processed_text'])
    logger.info(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")

    logger.info("Calculating cosine similarity...")
    cosine_sim_text = cosine_similarity(tfidf_matrix)
    logger.info(f"Text Cosine Similarity Matrix Shape: {cosine_sim_text.shape}")

    def categorical_similarity(row1, row2):
        score = 0
        if row1['Industry'] == row2['Industry']:
            score += 0.5
        if row1['Career Level'] == row2['Career Level']:
            score += 0.3
        if row1['Job Type'] == row2['Job Type']:
            score += 0.2
        return score

    num_jobs = len(df)
    hybrid_sim = np.zeros((num_jobs, num_jobs))
    for i in range(num_jobs):
        for j in range(num_jobs):
            text_sim = cosine_sim_text[i, j]
            cat_sim = categorical_similarity(df.iloc[i], df.iloc[j])
            hybrid_sim[i, j] = 0.7 * text_sim + 0.3 * cat_sim
    logger.info(f"Hybrid Similarity Matrix Shape: {hybrid_sim.shape}")

    # Compute related jobs for each job
    logger.info("Generating related jobs...")
    df['related_jobs'] = [
        df.iloc[np.argsort(hybrid_sim[idx])[::-1][1:4]]['JobID'].tolist()
        for idx in range(len(df))
    ]

    # Save only JobID and related_jobs to CSV
    output_df = df[['JobID', 'related_jobs']]
    output_df.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Related jobs data saved to {OUTPUT_PATH} with {len(output_df)} rows")

    return df, vectorizer, tfidf_matrix, hybrid_sim

def get_related_jobs_for_ids(job_ids, num_related=3):
    """
    Retrieve related jobs for a list of job IDs individually.
    
    Args:
        job_ids (list): List of job IDs to find related jobs for.
        num_related (int): Number of related jobs to return per job ID.
    
    Returns:
        dict: Mapping of job IDs to their related job IDs.
    """
    try:
        # Load precomputed data
        model_file_path = os.path.join(MODEL_PATH, 'job_data.pkl')
        with open(model_file_path, 'rb') as f:
            data = pickle.load(f)
            df = data['df']
            hybrid_sim = data['hybrid_sim']
        
        result = {}
        for job_id in job_ids:
            # Find index of job_id in DataFrame
            job_index = df.index[df['JobID'] == job_id].tolist()
            if not job_index:
                logger.warning(f"Job ID {job_id} not found in DataFrame")
                result[job_id] = []
                continue
            
            job_index = job_index[0]
            sim_scores = hybrid_sim[job_index]
            related_indices = np.argsort(sim_scores)[::-1][1:num_related+1]
            related_job_ids = df.iloc[related_indices]['JobID'].tolist()
            result[job_id] = related_job_ids
        
        return result
    except Exception as e:
        logger.error(f"Error in get_related_jobs_for_ids: {str(e)}", exc_info=True)
        raise

def get_related_jobs_for_multiple(job_ids, num_related=3):
    """
    Retrieve jobs most similar to a set of job IDs combined.
    
    Args:
        job_ids (list): List of job IDs (2 or more).
        num_related (int): Number of related jobs to return.
    
    Returns:
        list: List of job IDs most similar to the combined set.
    """
    try:
        if len(job_ids) < 2:
            raise ValueError("At least 2 job IDs must be provided")
        
        # Load precomputed data
        model_file_path = os.path.join(MODEL_PATH, 'job_data.pkl')
        if not os.path.exists(model_file_path):
            raise FileNotFoundError(f"Model file not found at {model_file_path}")
        
        with open(model_file_path, 'rb') as f:
            data = pickle.load(f)
            df = data['df']
            hybrid_sim = data['hybrid_sim']
        
        # Find indices of the input job IDs
        indices = []
        for job_id in job_ids:
            job_index = df.index[df['JobID'] == job_id].tolist()
            if not job_index:
                logger.warning(f"Job ID {job_id} not found in DataFrame")
                return []
            indices.append(job_index[0])
        
        # Compute combined similarity by averaging the similarity scores
        combined_sim = np.mean([hybrid_sim[idx] for idx in indices], axis=0)
        
        # Exclude the input jobs from the results
        for idx in indices:
            combined_sim[idx] = -1  # Set to negative to exclude from sorting
        
        # Get the top num_related job IDs
        related_indices = np.argsort(combined_sim)[::-1][:num_related]
        related_job_ids = df.iloc[related_indices]['JobID'].tolist()
        
        logger.info(f"Retrieved {len(related_job_ids)} related jobs for {len(job_ids)} job IDs: {job_ids}")
        return related_job_ids
    
    except Exception as e:
        logger.error(f"Error in get_related_jobs_for_multiple: {str(e)}", exc_info=True)
        raise