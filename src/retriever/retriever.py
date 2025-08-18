import os
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(os.getcwd())/'.env')

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_PATHS ={
    "course":"/vectorstore/courses",
    "blog":"/vectorstore/blogs"
}

class RetrieverService:
    def __init__(self, threshold: float = 0.38, k:int = 4):
        self.threshold = threshold
        self.k = k
        self.embedding_model = self._get_embedding_model()
        self.dbs = {
            "course": self._load_db("course"),
            "blog": self._load_db("blog")
        }

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_embedding_model():
        # Embedding_model
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en",
            encode_kwargs={"normalize_embeddings": True}
        )

    @lru_cache(maxsize=None)
    def _load_db(self, content_type: str):
        db_path = CHROMA_PATHS.get(content_type)
        if not db_path or not os.path.exists(db_path):
            raise ValueError(f"Unknown or missing vectorstore path for type: {content_type}")
        return Chroma(persist_directory=db_path, embedding_function=self.embedding_model)

    def similarity_with_scores(self, query, content_type, k ):
        db = self.dbs[content_type]
        return db.similarity_search_with_relevance_scores(query, k=k)

    def guarded_retriever(self,query, content_type, k, thresh):
        pairs = self.similarity_with_scores(query, content_type, k or self.k)

        scores = [s for _, s in pairs]
        docs = [d for d, _ in pairs]
        avg_score = sum(scores) / len(scores)

        if avg_score < (thresh or self.threshold):
            return [], "low_conf", {"avg_score": avg_score}

        return docs, "ok", {"avg_score": avg_score}
