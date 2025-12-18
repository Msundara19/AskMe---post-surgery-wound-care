from pathlib import Path
from dotenv import load_dotenv

def load_environment():
    # 1. Render secret file
    secret_file = Path("/etc/secrets/imp.env")
    if secret_file.exists():
        load_dotenv(secret_file)
        return

    # 2. Local imp.env in repo root
    local_file = Path("imp.env")
    if local_file.exists():
        load_dotenv(local_file)
        return

    # 3. Fallback to default .env
    load_dotenv()
