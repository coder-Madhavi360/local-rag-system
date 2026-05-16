from __future__ import annotations

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import html
from pathlib import Path

import streamlit as st

# =========================
# Modular Imports
# =========================

from ingestion.loader import (
    load_documents_folder,
    save_uploaded_files,
)

from processing.chunking import (
    split_documents,
)

from retrieval.vectorstore import (
    build_vector_store,
)

from retrieval.hybrid_retriever import (
    hybrid_retrieve,
    reformulate_query,
)

from retrieval.reranker import (
    rerank_documents,
)

from retrieval.confidence import (
    confidence_score,
)

from core.llm import (
    generate_answer,
)

# =========================
# Configuration
# =========================

DOCS_DIR = Path("data/docs")

UPLOADED_DOCS_DIR = Path(
    "data/uploaded_docs"
)

# =========================
# Streamlit Page Config
# =========================

st.set_page_config(
    page_title="Local RAG System",
    layout="wide",
)

# =========================
# Custom CSS
# =========================

st.markdown(
    """
    <style>

    .main {
        padding-top: 10px;
    }

    .stButton button {
        width: 100%;
        border-radius: 10px;
    }

    .chunk-box {
        background-color: #111827;
        color:white;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        border: 1px solid #374151;
        line-height:1.6;
    }

    .source-box {
        color: #60a5fa;
        font-size: 14px;
        margin-bottom: 10px;
        font-weight: bold;
    }

    .answer-box {
        background-color: #0f172a;
        color:white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #334155;
        font-size: 16px;
        line-height:1.7;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Title
# =========================

st.title("📚 Fully Local RAG System")

st.markdown(
    """
Upload multiple documents and ask grounded questions from uploaded files.
"""
)

# =========================
# Sidebar
# =========================

st.sidebar.header(
    "⚙️ Settings"
)

chunk_size = st.sidebar.slider(
    "Chunk Size",
    200,
    1200,
    500,
)

chunk_overlap = st.sidebar.slider(
    "Chunk Overlap",
    0,
    300,
    100,
)

top_k = st.sidebar.slider(
    "Retrieved Chunks",
    1,
    10,
    4,
)

uploaded_files = st.sidebar.file_uploader(
    "Upload Documents",
    accept_multiple_files=True,
)

if st.sidebar.button(
    "🧹 Clear Chat"
):
    st.session_state.messages = []

# =========================
# Save Uploaded Files
# =========================

save_uploaded_files(
    uploaded_files
)

# =========================
# Load Documents
# =========================

documents = []

documents.extend(
    load_documents_folder(
        DOCS_DIR,
        "docs",
    )
)

documents.extend(
    load_documents_folder(
        UPLOADED_DOCS_DIR,
        "uploaded",
    )
)

# =========================
# Process Documents
# =========================

if documents:

    chunks = split_documents(
        documents,
        chunk_size,
        chunk_overlap,
    )

    vector_store = build_vector_store(
        chunks
    )

else:

    chunks = []

    vector_store = None

# =========================
# Chat Memory
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

# =========================
# User Input
# =========================

question = st.chat_input(
    "Ask question from uploaded documents..."
)

# =========================
# Question Pipeline
# =========================

if question:

    previous_messages = list(
        st.session_state.messages
    )

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Generating answer..."
        ):

            # =========================
            # Query Reformulation
            # =========================

            retrieval_query = reformulate_query(
                query=question,
                chat_history=previous_messages,
            )

            # =========================
            # Hybrid Retrieval
            # =========================

            retrieved_docs = hybrid_retrieve(
                question=retrieval_query,
                vector_store=vector_store,
                chunks=chunks,
                top_k=top_k,
            )

            if not retrieved_docs:
                answer = (
                    "No relevant information found in uploaded documents."
                )

                st.warning(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

                st.stop()

            # =========================
            # Reranking
            # =========================

            reranked_docs = rerank_documents(
                query=retrieval_query,
                documents=retrieved_docs,
                top_k=top_k,
            )

            final_docs = reranked_docs

            # =========================
            # Confidence Score
            # =========================

            confidence = confidence_score(
                reranked_docs,
                question,
            )

            # =========================
            # Answer Generation
            # =========================

            answer = generate_answer(
                question,
                final_docs,
            )

            # =========================
            # Display Answer
            # =========================

            st.markdown(
                f"""
                <div class="answer-box">
                    {answer}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"### Confidence Score: {confidence}%"
            )

            # =========================
            # Retrieved Chunks
            # =========================

            st.markdown(
                "### Retrieved Context"
            )

            for doc in final_docs:

                source = doc.metadata.get(
                    "source",
                    "unknown",
                )

                page = doc.metadata.get(
                    "page",
                    None,
                )

                sheet = doc.metadata.get(
                    "sheet",
                    None,
                )

                chunk = doc.metadata.get(
                    "chunk",
                    None,
                )

                source_text = source

                if page:
                    source_text += (
                        f" | Page {page}"
                    )

                if sheet:
                    source_text += (
                        f" | Sheet {sheet}"
                    )

                if chunk:
                    source_text += (
                        f" | Chunk {chunk}"
                    )

                st.markdown(
                    f'<div class="chunk-box">'
                    f'<div class="source-box">{html.escape(source_text)}</div>'
                    f'{html.escape(doc.page_content[:1000])}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )
