# Enhanced embedding model with better job-specific capabilities
from langchain_huggingface import HuggingFaceEmbeddings
import torch


embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v2",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True, "precision": "binary"},
)
