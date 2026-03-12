import os
from typing import Dict

DEFAULT_DB_CONFIG = {
    "DB_NAME": "smartassist",
    "DB_USER": "sahil_ats",
    "DB_PASSWORD": "sahil@ats",
    "DB_HOST": "3.7.24.77",
    "DB_PORT": "5432",
}


def get_db_config() -> Dict[str, str]:
    return {key: os.getenv(key, value) for key, value in DEFAULT_DB_CONFIG.items()}
