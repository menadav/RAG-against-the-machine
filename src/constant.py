"""Configuration constants for the RAG pipeline.

Attributes:
    EXTENSIONS (List[str]): List of allowed file extensions for ingestion.
    PATH_CP (str): Source directory path for the raw repository files.
    CONFIG_PATH (str): File path for saving/loading
    the ingestion configuration JSON.
    DEFAULT_K (int): Default number of results to retrieve during search.
    DEFAULT_CHUNK (int): Default maximum character size for document chunking.
    LLM (str): Identifier of the Large Language Model used
    for response generation.
"""

EXTENSIONS = [
    ".py",
    ".md",
    ""
]
PATH_CP = "data/raw/vllm-0.10.1"
CONFIG_PATH = "data/processed/config.json"
DEFAULT_K = 10
DEFAULT_CHUNK = 2000
LLM = "Qwen/Qwen3-0.6B"
