import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(os.getcwd())/'.env')

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from jinja2 import Environment, FileSystemLoader

from src.retriever.retriever_v0_1 import guarded_retriever

llm = ChatOpenAI(temperature=0)

def format_candidates(docs):
    lines = []
    for i, d in enumerate(docs, 1):
        lines.append(f"[{i}] {d.metadata.get('title', '')} — {d.page_content}\nURL: {d.metadata.get('url', '')}")
    return "\n\n".join(lines)

def answer_with_citation(query, intent, k = 4, conf_thresh = 0.38):
    docs, status, meta = guarded_retriever(query, intent, k=k, thresh=conf_thresh)

    if status in ("empty", "low_conf"):
        return {
            "status": "oos",
            "intent": intent,
            "message": ("It appears that the query does not match the materials on deeplearning.ai "
                        "or the relevance is too low. Please clarify the topic "
                        "for example: ‘RAG evaluation’, ‘Python for AI’, ‘LLMOps’"),
            "meta": meta
        }

    # Jinja template
    env = Environment(loader=FileSystemLoader("prompts"), autoescape=False)
    template = env.get_template("recommendation_prompt.jinja")
    rendered_prompt = template.render(query=query, k=k, candidates=format_candidates(docs))
    prompt = ChatPromptTemplate.from_messages([("human", rendered_prompt)])

    pipeline = prompt | llm | StrOutputParser()
    text = pipeline.invoke({
        "query": query,
        "k": k,
        "candidates": format_candidates(docs)
    })

    out = {"status": "ok", "raw": text, "intent": intent, "meta": meta}

    for line in text.splitlines():
        if line.startswith("Title:"): out["title"] = line.split(":", 1)[1].strip()
        if line.startswith("Why:"): out["why"] = line.split(":", 1)[1].strip()
        if line.startswith("URL:"): out["url"] = line.split(":", 1)[1].strip()

    out["candidates"] = [
        {"title": d.metadata.get("title"), "url": d.metadata.get("url"), "snippet": d.page_content}
        for d in docs
    ]


    return out