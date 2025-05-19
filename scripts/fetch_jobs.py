import pandas as pd
import psycopg2
from config.config import DB_CONFIG, DATASET_PATH

def fetch_jobs():
    conn = psycopg2.connect(**DB_CONFIG)
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
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        df = pd.DataFrame(rows, columns=columns)
        
        # Generate URL Job
        df['url_job'] = df['job_id'].apply(lambda x: f"https://job-compass.bunkid.online/single-job/{x}")
        
        # Combine categories and specializations into Industry
        df['industry'] = df.apply(
            lambda x: ", ".join(filter(None, [x['categories'], x['specializations']])),
            axis=1
        )
        df['industry'] = df['industry'].replace("", "Not specified")
        
        # Drop temporary columns used for industry
        df = df.drop(columns=['categories', 'specializations'])
        
        # Rename columns to match desired output
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
        
        # Define output columns
        output_columns = [
            "JobID", "URL Job", "Job Title", "Name Company", "Company Overview",
            "Company Size", "Company Address", "Job Description", "Job Requirements",
            "Benefits", "Job Address", "Job Type", "Career Level", "Years of Experience", "Industry"
        ]
        df = df[output_columns]
        
        # Save to CSV
        df.to_csv(DATASET_PATH, index=False)
        print(f"Data saved to {DATASET_PATH}")
    finally:
        cursor.close()
        conn.close()