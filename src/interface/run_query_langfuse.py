from distutils.command.clean import clean

from langfuse import observe, get_client
from langfuse.langchain import CallbackHandler

from src.interface.run_query import run_query

def run_query_with_trace(user_input, meta):
    client = get_client()
    root = client.start_span(name="chatbot_query", input={"query": user_input, "meta": meta})

    result = run_query(user_input)

    # decision
    dec = client.start_span(name="decision")
    dec.update(output=result.get("decision"))
    dec.end()

    # retriever
    ret = client.start_span(name="retriever")
    ret.update(output={"candidates": result.get("candidates", [])})
    ret.end()

    # generate
    gen = client.start_span(name="generate")
    dec.update(output={
        "title": result.get("title"),
        "url": result.get("url"),
        "why": result.get("why"),
        "status": result.get("status"),
        "intent": result.get("intent") or (result.get("decision") or {}).get("intent"),
    })
    gen.end()

    root.update(output={
        "status": result.get("status"),
        "intent": (result.get("decision") or {}).get("intent") or result.get("intent"),
        "title": result.get("title"),
        "url": result.get("url"),
    })
    root.end()

    # summary trace
    client.update_current_trace(output={"result_summary":{
        "status": result.get("status"),
        "intent": (result.get("decision") or {}).get("intent") or result.get("intent"),
        "title": result.get("title"),
        "url": result.get("url"),
    }})

    return result