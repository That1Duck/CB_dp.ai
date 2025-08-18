import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(os.getcwd())/'.env')

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from jinja2 import Environment, FileSystemLoader
import json
from typing import Literal, TypedDict

class Decision(TypedDict):
    domain: Literal["deeplearning_ai", "oos"]
    intent: Literal["course", "blog", "unknown"]

llm = ChatOpenAI(model = "gpt-4o-mini", temperature= 0)

def decide(text):
    # Jinja
    env = Environment(loader=FileSystemLoader("prompts"), autoescape=False)
    template = env.get_template("decision_prompt.jinja")
    rendered = template.render(text = text)
    prompt = ChatPromptTemplate.from_messages([("human"), rendered])

    chain = prompt | llm | StrOutputParser()
    res_chain = chain.invoke({})
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
