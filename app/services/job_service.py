import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .preprocess import preprocess_text
from app.config.config import OUTPUT_PATH
import logging
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def compute_related_jobs(df):
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

    def get_related_job_ids(job_index, num_jobs=3):
        sim_scores = hybrid_sim[job_index]
        related_indices = np.argsort(sim_scores)[::-1][1:num_jobs+1]
        related_job_ids = df.iloc[related_indices]['JobID'].tolist()
        return related_job_ids

    logger.info("Generating related jobs...")
    df['related_jobs'] = [get_related_job_ids(idx, num_jobs=3) for idx in range(len(df))]
    df.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Related jobs data saved to {OUTPUT_PATH} with {len(df)} rows")
    return df, vectorizer, tfidf_matrix

    df['processed_text'] = df.apply(lambda row: ' '.join([
        preprocess_text(row['Job Title']) * 4,
        preprocess_text(row['Job Description']) * 2,
        preprocess_text(row['Job Requirements']),
        preprocess_text(row['Industry']) * 3,
        preprocess_text(row['Career Level']) * 2,
        preprocess_text(row['Job Type']) * 2
    ]), axis=1)

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.7,
        sublinear_tf=True
    )
    tfidf_matrix = vectorizer.fit_transform(df['processed_text'])

    cosine_sim_text = cosine_similarity(tfidf_matrix)

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

    def get_related_job_ids(job_index, num_jobs=3):
        sim_scores = hybrid_sim[job_index]
        related_indices = np.argsort(sim_scores)[::-1][1:num_jobs+1]
        related_job_ids = df.iloc[related_indices]['JobID'].tolist()
        return related_job_ids

    df['related_jobs'] = [get_related_job_ids(idx, num_jobs=3) for idx in range(len(df))]
    df.to_csv(OUTPUT_PATH, index=False)
    return df