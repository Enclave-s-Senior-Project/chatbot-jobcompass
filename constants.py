import os
import dotenv

dotenv.load_dotenv()


vector_database_url = (
    f'postgresql://{os.getenv("VECTOR_DB_USERNAME")}:{os.getenv("VECTOR_DB_PASSWORD")}'
    f'@{os.getenv("VECTOR_DB_HOST")}:{os.getenv("VECTOR_DB_PORT", "5432")}'
    f'/{os.getenv("VECTOR_DB_DATABASE")}'
)

main_database_url = (
    f'postgresql://{os.getenv("MAIN_DB_USERNAME")}:{os.getenv("MAIN_DB_PASSWORD")}'
    f'@{os.getenv("MAIN_DB_HOST")}:{os.getenv("MAIN_DB_PORT", "5432")}'
    f'/{os.getenv("MAIN_DB_DATABASE")}'
)
