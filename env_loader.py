import os
from pathlib import Path


def load_local_env(filename: str = ".env") -> bool:
    path = Path(filename)
    if not path.exists() or not path.is_file():
        return False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value
    return True
