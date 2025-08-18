import os

from src.image_processing.image2text import process_image
from src.agents.decision_agents import decide
from src.chains.retriever_chain import answer_with_citation
from src.utils.logging import get_logger

logger = get_logger("Pipeline")

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")

def run_query(user_input):
    logger.info("Start processing request")
    # Input
    # Check if user_input is the path to image
    if os.path.isfile(user_input) and user_input.lower().endswith(IMAGE_EXT):
        # Processing image
        img = process_image(user_input)
        query_text = img["combined"].strip() or img["ocr_text"] or img["blip_text"]
        source_type = "image"
    else:
        query_text = user_input.strip()
        source_type = "text"

    if not query_text:
        return {"status": "bad_input", "message": "Error", "source_type": source_type}

    logger.info("Identification of the source completed")
    # Choose intent for retriever
    decision = decide(query_text)
    logger.info("Routing completed")
    #
    if decision["domain"] == "oos":
        return{
            "status": "oos",
            "message": ("The request does not appear to be relevant to deeplearning.ai materials. Please request a course/post on a specific topic (example: ‘course on RAG’)."),
            "decision": decision,
            "query_text": query_text,
            "source_type": source_type
        }

    intent = decision["intent"] if decision["intent"] in ("course", "blog") else "course"

    # Result of retriever
    result = answer_with_citation(query = query_text, intent=intent, k = 4, conf_thresh=0.38)
    result["query_text"] = query_text
    result["source_type"] = source_type
    result["decision"] = decision
    logger.info("Generation complete")
    return result