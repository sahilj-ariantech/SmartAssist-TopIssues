import os
from typing import Dict

DEFAULT_DB_CONFIG = {
    "DB_NAME": "app-api",
    "DB_USER": "postgres",
    "DB_PASSWORD": "SmartAssist@ATS20240708",
    "DB_HOST": "167.71.231.98",
    "DB_PORT": "5432",
}


def get_db_config() -> Dict[str, str]:
    return {key: os.getenv(key, value) for key, value in DEFAULT_DB_CONFIG.items()}
