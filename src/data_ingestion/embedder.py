#
import json
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

RAW_DATA_PATHS = {
    "blog": "data/raw/blog.json",
    "course": "data/raw/courses.json"
}

CHROMA_PATHS ={
    "course":"/vectorstore/courses",
    "blog":"/vectorstore/blogs"
}

# Embedding_model
embedding_model = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-small-en",
    encode_kwargs = {"normalize_embeddings": True}
)

def load_data(content_type):
    path = RAW_DATA_PATHS.get(content_type)

    with open(path, 'r', encoding="utf-8") as f:
        return json.load(f)

def build_documents(data):
    docs = []
    for item in data:
        docs.append(Document(
            page_content=item.get("description"),
            metadata = {
                "title": item.get("title"),
                "url": item.get("url"),
                "type": item.get("type"),
                "source": item.get("source")
            }
        ))
    return docs

def emded_and_store(content_type):
    data = load_data(content_type)
    documents = build_documents(data)
    vector_path = CHROMA_PATHS.get(content_type)

    db = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory='.'+vector_path
    )

if __name__ == "__main__":
    print("Start of embedding .....")
    emded_and_store("blog")
    emded_and_store("course")
    print("Done.")