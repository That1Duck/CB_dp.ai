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

def load_db(content_type):
    db_path = CHROMA_PATHS.get(content_type)
    if not db_path or not os.path.exists(db_path):
        raise ValueError(f"Unknown or missing vectorstore path for type: {content_type}")
    return Chroma(persist_directory=db_path, embedding_function= embedding_model)

def similarity_with_scores(query, content_type, k = 4):
    db = load_db(content_type)
    # (Document, score)
    return db.similarity_search_with_relevance_scores(query, k=k)

def guarded_retriever(query, content_type, k, thresh = 0.38):
    pairs = similarity_with_scores(query, content_type, k)

    scores = [s for _, s in pairs]
    docs = [d for d, _ in pairs]
    avg_score = sum(scores)/len(scores)

    if not (avg_score >= thresh):
        return [], "low_conf", {"avg_score": avg_score}
        
    return docs, "ok", {"avg_score": avg_score}