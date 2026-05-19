from __future__ import annotations

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import html
from pathlib import Path
from pipeline.rag_pipeline import run_rag_pipeline
from pipeline.insert import insert_into_memory, insert_into_vector_store
import streamlit as st
import time
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

from core.memory import (
    clear_memory,
    get_chat_history,
    initialize_memory,
    restore_memory_from_chat_history,
)

from evaluation.ragas_eval import (
    run_ragas_evaluation,
)

from utils.cache import (
    clear_chat_history,
    initialize_chat_history,
    load_chat_history,
    save_chat_message,
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

conversation_memory = initialize_memory(
    st.session_state
)
chat_history = initialize_chat_history(
    st.session_state
)
restore_memory_from_chat_history(
    conversation_memory,
    chat_history,
)

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
    clear_chat_history(
        st.session_state
    )
    clear_memory(
        conversation_memory
    )

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

    vector_store = insert_into_vector_store(
        chunks
    )

else:

    chunks = []

    vector_store = None

# =========================
# Chat Memory
# =========================

# Chat history is stored in Streamlit session state so it survives reruns in
# the current app session. ConversationBufferMemory stays aligned with it for
# follow-up question handling and multi-turn contextual retrieval.
for message in load_chat_history(
    st.session_state
):

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

# =========================
# User Input
# =========================
# =========================
# Voice Input
# =========================


question = st.chat_input(
    "Ask question from uploaded documents..."
)


# =========================
# Question Pipeline
# =========================

if question:

    # Conversation memory feeds prior turns into query reformulation. That lets
    # follow-up questions become standalone retrieval queries while preserving
    # the existing metadata filtering, hybrid retrieval, reranking, confidence,
    # and context-viewer behavior.
    memory_messages = get_chat_history(
        conversation_memory
    )

    save_chat_message(
        st.session_state,
        "user",
        question,
    )

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Generating answer..."
        ):

            pipeline_result = run_rag_pipeline(
                query=question,
                vector_store=vector_store,
                chunks=chunks,
                top_k=top_k,
                chat_history=memory_messages,
            )

            answer = pipeline_result["answer"]
            final_docs = pipeline_result["final_docs"]
            confidence = pipeline_result["confidence"]

            if not final_docs:

                st.warning(answer)

                save_chat_message(
                    st.session_state,
                    "assistant",
                    answer,
                )
                insert_into_memory(
                    conversation_memory,
                    question,
                    answer,
                )

                st.stop()

            # =========================
            # Display Answer
            # =========================
            response_placeholder=st.empty()
            streamed_text = ""

            for word in answer.split():
                streamed_text += word + " "
                response_placeholder.markdown(
                      f"""
                     <div class="answer-box">
                    {streamed_text}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                time.sleep(0.03)

            # =========================
            # RAG Evaluation
            # =========================

            ragas_scores, ragas_error = run_ragas_evaluation(
                question=question,
                answer=answer,
                retrieved_docs=final_docs,
            )

            st.markdown(
                "### RAG Evaluation"
            )

            if ragas_error:
                st.warning(
                    ragas_error
                )
            elif ragas_scores:
                for metric_name, score in ragas_scores.items():
                    st.markdown(
                        f"- {metric_name}: {score}"
                    )
            else:
                st.info(
                    "RAGAS evaluation did not return scores."
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

    save_chat_message(
        st.session_state,
        "assistant",
        answer,
    )
    insert_into_memory(
        conversation_memory,
        question,
        answer,
    )
