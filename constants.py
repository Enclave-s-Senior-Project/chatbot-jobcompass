import os
import dotenv

dotenv.load_dotenv()


database_url = (
    f'postgresql://{os.getenv("TYPEORM_USERNAME")}:{os.getenv("TYPEORM_PASSWORD")}'
    f'@{os.getenv("TYPEORM_HOST")}:{os.getenv("TYPEORM_PORT", "5432")}'
    f'/{os.getenv("TYPEORM_DATABASE")}'
)
