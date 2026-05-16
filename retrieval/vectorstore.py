from pathlib import Path
import json
import logging
from typing import Any, Mapping

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from processing.metadata import build_chroma_where_filter

logger = logging.getLogger(__name__)

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


# =========================
# Metadata Filtering
# =========================


def apply_metadata_filter(
    search_kwargs: Mapping[str, Any] | None,
    metadata_filters: Mapping[str, Any] | None,
    documents=None,
) -> dict[str, Any]:
    """Attach a ChromaDB metadata filter to retriever search kwargs.

    Chroma can filter on stored chunk metadata before vector similarity scores
    are returned. This document filtering improves retrieval precision for
    prompts such as "from AI.pdf", "page 4", or "sheet Sales" while preserving
    embeddings, chunking, reranking, confidence scoring, and the context viewer.

    Future extension points:
    - User-selected filters from Streamlit controls.
    - Advanced metadata search across richer fields.
    - Document collections that expand to multiple source filters.
    """

    updated_kwargs = dict(
        search_kwargs or {}
    )

    if not metadata_filters:
        return updated_kwargs

    chroma_filter = build_chroma_where_filter(
        documents=documents,
        filters=metadata_filters,
    )

    if not chroma_filter:
        logger.info(
            "No Chroma metadata filter applied for filters: %s",
            metadata_filters,
        )
        return updated_kwargs

    updated_kwargs["filter"] = chroma_filter

    logger.info(
        "Applied Chroma metadata filter: %s",
        chroma_filter,
    )

    return updated_kwargs
