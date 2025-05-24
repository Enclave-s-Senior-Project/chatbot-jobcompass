# Enhanced embedding model with better job-specific capabilities
from langchain_huggingface import HuggingFaceEmbeddings
import torch


embeddings_model = HuggingFaceEmbeddings(
    model_name="Snowflake/snowflake-arctic-embed-m",
    model_kwargs={
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "trust_remote_code": True,
    },
    encode_kwargs={
        "normalize_embeddings": True,
        "precision": "binary",
        "batch_size": 32,
    },
)
