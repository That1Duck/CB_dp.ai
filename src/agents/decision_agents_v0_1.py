import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(os.getcwd())/'.env')

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import json
from typing import Literal, TypedDict

class Decision(TypedDict):
    domain: Literal["deeplearning_ai", "oos"]
    intent: Literal["course", "blog", "unknown"]

llm = ChatOpenAI(model = "gpt-4o-mini", temperature= 0)

PROMPT = ChatPromptTemplate.from_template("""
You are an intelligent assistant that classifies user intent and domain.
If the user wants to **learn**, to study a **new skill**, it is better to choose “course.”
If they just want to **read** or **get an overview**, choose “blog.”

Your task is to return a STRICT JSON with two keys: "domain" and "intent".
- "intent": Choose “course” if the user wants to learn, gain skills, or understand how to do something. Choose “blog” if the user wants to read, explore ideas, or get an overview. If unclear, return “unknown”.
- "domain": Choose “deeplearning_ai” if the user's request could be satisfied by deeplearning.ai content (courses or blogs), especially in areas like deep learning, LLMs, autonomous agents, LangChain, RAG, machine learning, or AI systems. Otherwise, return “oos”.

Use this key words to help yourself better classify input query:
- "course":  how, tutorial, guide, step-by-step, implement, build, code, train, fine-tune, learn, course, certificate, lesson
- "blog": can, what, why, is, are, pros/cons, examples, use cases, introduction, overview

Use reasoning based on the user's goal, even if they don't explicitly say “learn” or “read”.

Return ONLY JSON. Examples:
{{"domain":"deeplearning_ai","intent":"course"}}
{{"domain":"deeplearning_ai","intent":"blog"}}
{{"domain":"oos","intent":"unknown"}}

User query: {text}
""")

chain = PROMPT | llm | StrOutputParser()

def decide(text):
    res_chain = chain.invoke({"text": text})
    # JSON parsing
    try:
        data = json.loads(res_chain)
        domain = data.get("domain") or "oos"
        intent = data.get("intent") or "unknown"
    except Exception:
        domain, intent = "oos", "unknown"

    valid_domains = {"deeplearning_ai", "oos"}
    valid_intents = {"course", "blog", "unknown"}

    domain = domain if domain in valid_domains else "oos"
    intent = intent if intent in valid_intents else "unknown"

    return {"domain":domain, "intent":intent}

def decide_type(text):
    d = decide(text)
    return d["intent"]


"""

You work as an intelligent assistant. You receive a user request. 
Determine which is best suited: “course” or “blog.” for "intent"

If the user wants to **learn**, to study a **new skill**, it is better to choose “course.”
If they just want to **read** or **get an overview**, choose “blog.”
Return STRICT JSON with keys: "domain" and "intent".
- "domain": "deeplearning_ai" if the user's need can be satisfied by deeplearning.ai courses/blogs; otherwise "oos".
- "intent": "course" if the user wants to learn a skill or take a class; "blog" if they want to read news/insights; "unknown" if unclear.

Return ONLY JSON. Examples:
{{"domain":"deeplearning_ai","intent":"course"}}
{{"domain":"deeplearning_ai","intent":"blog"}}
{{"domain":"oos","intent":"unknown"}}

User query: {text}

"""