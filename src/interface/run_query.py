import os

from src.image_processing.image2text import process_image
from src.agents.decision_agent import decide_type
from src.chains.retriever_chain import answer_with_citation

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")

def run_query(user_input):
    # Input
    # Check if user_input is the path to image
    if os.path.isfile(user_input) and user_input.lower().endwith(IMAGE_EXT):
        # Processing image
        img = process_image(user_input)
        query_text = img["combined_text"].strip() or img["ocr_text"] or img["blip_text"]
        source_type = "image"
    else:
        query_text = user_input.strip()
        source_type = "text"

    if not query_text:
        return {"status":"bad_input", "message":"Error"}

    # Choose intent for retriever
    intent = decide_type(query_text)
    #

    # Result of retriever
    result = answer_with_citation(query = query_text, intent=intent, k = 4)
    result["query_text"] = query_text
    result["source_type"] = source_type
    return result