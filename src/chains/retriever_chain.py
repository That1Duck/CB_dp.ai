from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.retriever.retriever import get_relevant_docs

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(os.getcwd())/'.env')

llm = ChatOpenAI(temperature=0)

PROMPT = ChatPromptTemplate.from_template("""
You are an assistant who recommends material from deeplearning.ai.
You have a user request and up to {k} candidates. 
Choose ONE best candidate, explain briefly why (1–2 sentences), and provide the exact link.
Answer format:
Title: ...
Why: ...
URL: ...

Request: {query}

Candidates:
{candidates}
""")

def format_candidates(docs):
    lines = []
    for i, d in enumerate(docs, 1):
        lines.append(f"[{i}] {d.metadata.get('title', '')} — {d.page_content}\nURL: {d.metadata.get('url', '')}")
    return "\n\n".join(lines)

def answer_with_citation(query, intent, k ):
    docs = get_relevant_docs(query, intent, k)
    if not docs:
        return {"status":"no_results",
                "message":"Not found"}

    text = (PROMPT | llm | StrOutputParser()).invoke({
        "query": query,
        "k": k,
        "candidates": format_candidates(docs)
    })

    out = {"status": "ok", "raw": text, "intent": intent}

    for line in text.splitlines():
        if line.startswith("Title:"): out["title"] = line.split(":", 1)[1].strip()
        if line.startswith("Why:"): out["why"] = line.split(":", 1)[1].strip()
        if line.startswith("URL:"): out["url"] = line.split(":", 1)[1].strip()

    out["candidates"] =[
        {"title":d.metadata.get("title"), "url":d.metadata.get("url"), "snippet":d.page_content}
        for d in docs
    ]

    return out