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
def format_sources(docs):
    """
    Format retrieved document sources cleanly.
    """

    formatted_sources = []

    for doc in docs:
        metadata = doc.metadata

        source = metadata.get("source", "Unknown")
        page = metadata.get("page", None)
        chunk = metadata.get("chunk", None)

        source_name = source.split("/")[-1]

        if page is not None:
            formatted = f"{source_name} — Page {page}"
        elif chunk is not None:
            formatted = f"{source_name} — Chunk {chunk}"
        else:
            formatted = source_name

        if formatted not in formatted_sources:
            formatted_sources.append(formatted)

    return formatted_sources