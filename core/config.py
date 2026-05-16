from pathlib import Path

DOCS_DIR = Path("data/docs")
UPLOADED_DOCS_DIR = Path("data/uploaded_docs")
VECTOR_DB_DIR = Path("data/vector_db/chroma")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
GENERATION_MODEL = "google/flan-t5-small"
RERANKER_MODEL = "BAAI/bge-reranker-base"