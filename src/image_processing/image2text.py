import easyocr
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Initialization OCR
ocr_reader = easyocr.Reader(['en'], gpu = False)

# Initialization Blip
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def process_image(image_path):
    """

    """
    image = Image.open(image_path).convert("RGB")

    # OCR
    ocr_results = ocr_reader.readtext(image_path, detail = 0)
    ocr_text = ' '.join(ocr_results).strip()

    # BLIP
    blip_inputs = blip_processor(image, return_tensors = "pt")
    with torch.no_grad():
        blip_output = blip_model.generate(**blip_inputs)
    blip_text = blip_processor.decode(blip_output[0], skip_special_tokens=True)

    # Comb results
    combined = f"{ocr_text}. {blip_text}".strip()

    return {
        "ocr_text":ocr_text,
        "blip_text":blip_text,
        "combined":combined
    }