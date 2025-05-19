import os
from pathlib import Path

# Base directory for file paths
BASE_DIR = Path(__file__).resolve().parent.parent

# Database configuration
DB_CONFIG_PRIMARY = {
    "dbname": os.getenv("MAIN_DB_DATABASE"),
    "user": os.getenv("MAIN_DB_USERNAME"),
    "password": os.getenv("MAIN_DB_PASSWORD"),
    "host": os.getenv("MAIN_DB_HOST", "localhost"),
    "port": os.getenv("MAIN_DB_PORT", "5432"),
}

DB_CONFIG_VECTOR = {
    "dbname": os.getenv("VECTOR_DB_DATABASE"),
    "user": os.getenv("VECTOR_DB_USERNAME"),
    "password": os.getenv("VECTOR_DB_PASSWORD"),
    "host": os.getenv("VECTOR_DB_HOST", "localhost"),
    "port": os.getenv("VECTOR_DB_PORT", "5432"),
}

# File paths for CSV outputs
DATASET_PATH = os.getenv("DATASET_PATH", str(BASE_DIR / "data" / "jobs.csv"))
OUTPUT_PATH = os.getenv("OUTPUT_PATH", str(BASE_DIR / "data" / "related_jobs.csv"))
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models"))

# API and frontend configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DETAILS_SINGLE_JOB_FRONTEND_LINK = os.getenv("DETAILS_SINGLE_JOB_FRONTEND_LINK", "https://job-compass.bunkid.online/single-job")
JOB_API_URL = os.getenv("JOB_API_URL", "https://job-compass.me/api/v1")

# Database connection settings
MAIN_DB_CONNECTION = os.getenv("MAIN_DB_CONNECTION", "postgres")
MAIN_DB_AUTOLOAD = os.getenv("MAIN_DB_AUTOLOAD", "true").lower() == "true"
MAIN_DB_SYNCHRONIZE = os.getenv("MAIN_DB_SYNCHRONIZE", "true").lower() == "true"
MAIN_DB_LOGGING = os.getenv("MAIN_DB_LOGGING", "false").lower() == "true"

VECTOR_DB_CONNECTION = os.getenv("VECTOR_DB_CONNECTION", "postgres")
VECTOR_DB_AUTOLOAD = os.getenv("VECTOR_DB_AUTOLOAD", "true").lower() == "true"
VECTOR_DB_SYNCHRONIZE = os.getenv("VECTOR_DB_SYNCHRONIZE", "true").lower() == "true"
VECTOR_DB_LOGGING = os.getenv("VECTOR_DB_LOGGING", "false").lower() == "true"

# Validate required environment variables
REQUIRED_ENV_VARS = [
    "MAIN_DB_DATABASE",
    "MAIN_DB_USERNAME",
    "MAIN_DB_PASSWORD",
    "MAIN_DB_HOST",
    "VECTOR_DB_DATABASE",
    "VECTOR_DB_USERNAME",
    "VECTOR_DB_PASSWORD",
    "VECTOR_DB_HOST",
    "OPENAI_API_KEY",
]
for var in REQUIRED_ENV_VARS:
    if not os.getenv(var):
        raise ValueError(f"Environment variable {var} is not set")