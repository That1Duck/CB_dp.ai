import tempfile
from pathlib import Path

import streamlit as st

from src.interface.run_query import run_query
from src.utils.logging import configure_logging, get_logger

configure_logging("INFO")
logger = get_logger("UI")

st.set_page_config(page_title="deeplearning.ai Assistant", layout="centered")

# Slider
with st.sidebar:
    st.markdown("Settings")
    show_candidates = st.checkbox("Top-k candidates", value=True)
    show_debug = st.checkbox("Debug info", value = True)
    st.markdown("----")

st.title("Deeplearning.ai Assistant")
st.caption("Enter your query text or upload an image. The bot will select a course/blog and provide a link (with a quote)")

# Inputs
text_query: str | None = st.text_input("Text query (optional)", placeholder="e.g. I want to learn Python for AI")

# Action
run_but = st.button("Search for material")

uploaded_file = st.file_uploader("Upload an image (png/jpg/webp) — optional", type=["png", "jpg", "jpeg", "webp"])
file_path: str | None = None
if uploaded_file is not None:
    tmp_dir = Path(tempfile.gettempdir())
    tmp_path = tmp_dir / f"dlai_ui_{uploaded_file.name}"
    tmp_path.write_bytes(uploaded_file.read())
    file_path = str(tmp_path)

st.markdown("")

def display_answer(res:dict):
    status = res.get("status")
    if status == "oos":
        st.warning(res.get("message") or "Request outside the domain deeplearning.ai. Specify the topic.")
        return

    title = res.get("title") or "-"
    why = res.get("why") or "-"
    url = res.get("url")

    st.markdown(f"### {title}")
    st.write(why)
    if url:
        st.markdown(f"URL: {url}")

    if show_candidates:
        cands = res.get("candidates") or []
        with st.expander(f"CAndidates (top-{len(cands)})", expanded= False):
            for id, cand in enumerate(cands, 1):
                cand_t = cand.get("title") or '-'
                cand_u = cand.get("url") or '-'
                st.markdown(f"{id}: {cand_t} \n{cand_u}")

    if show_debug:
        st.subheader("Debug info")
        st.json({
            "decision": res.get("decision"),
            "meta": res.get("meta"),
            "source_type": res.get("source_type"),
            "query_text": res.get("query_text"),
        })

def choose_input(text_query, file_path):
    text_query = (text_query or "").strip()
    if text_query:
        return text_query
    if file_path and Path(file_path).exists():
        return file_path
    return None

# Running
if run_but:
    user_input = choose_input(text_query, file_path)
    if not user_input:
        st.error("Enter your query or path to image")
    else:
        with st.spinner("Searching... "):
            try:
                logger.info("Searching answer")
                res = run_query(user_input)
                logger.info("Answer found")
                display_answer(res)
            except Exception as e:
                logger.exception("UI Error")
                st.error(f"Error {e}")
