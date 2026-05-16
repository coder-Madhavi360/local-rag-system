import hashlib
import re
from pathlib import Path

def hash_text(text):

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

def safe_filename(filename):

    name = Path(filename).name

    return re.sub(
        r"[^A-Za-z0-9._ -]",
        "_",
        name
    )