from pathlib import Path
import json

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# =========================
# Configuration
# =========================

VECTOR_DB_DIR = Path(
    "data/vector_db/chroma"
)

VECTOR_DB_META = Path(
    "data/vector_db/index_meta.json"
)

CHROMA_COLLECTION_PREFIX = (
    "local_rag_documents"
)

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# Embeddings
# =========================

def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

# =========================
# Build Vector Store
# =========================

def build_vector_store(
    chunks,
):

    VECTOR_DB_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    collection_name = (
        f"{CHROMA_COLLECTION_PREFIX}"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=collection_name,
        persist_directory=str(
            VECTOR_DB_DIR
        ),
    )

    VECTOR_DB_META.write_text(
        json.dumps(
            {
                "embedding_model":
                EMBEDDING_MODEL,
            },
            indent=2,
        )
    )

    return vector_store