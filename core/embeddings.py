import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from core.config import EMBEDDING_MODEL

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )