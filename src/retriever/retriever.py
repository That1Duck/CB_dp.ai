import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(os.getcwd())/'.env')

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_PATHS ={
    "course":"/vectorstore/courses",
    "blog":"/vectorstore/blogs"
}

# Embedding_model
embedding_model = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-small-en",
    encode_kwargs = {"normalize_embeddings": True}
)

def get_relevant_docs(query, content_type = "course", k = 1):
    """
    Searches the desired collection (course or blog).
    Returns the top k relevant documents.
    """
    db_path = CHROMA_PATHS.get(content_type)
    if not db_path or not os.path.exists(db_path):
        raise ValueError(f"Unknown or missing vectorstore path for type: {content_type}")

    vectordb = Chroma(
        persist_directory=db_path,
        embedding_function=embedding_model
    )

    return vectordb.similarity_search(query, k=k)