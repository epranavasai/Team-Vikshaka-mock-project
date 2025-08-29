from io import BytesIO
import requests
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from ..config import FASHION_MODEL_NAME, DEVICE

async def extract_keywords_from_image(url: str):
    processor = BlipProcessor.from_pretrained(FASHION_MODEL_NAME, use_fast=False)
    model = BlipForConditionalGeneration.from_pretrained(FASHION_MODEL_NAME).to(DEVICE) # type: ignore
    model.eval()

    response = requests.get(url, timeout=10)
    response.raise_for_status()

    img = Image.open(BytesIO(response.content)).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs, max_new_tokens=50, num_beams=5, # type: ignore
            repetition_penalty=1.2, early_stopping=True
        )

    caption = processor.decode(out_ids[0], skip_special_tokens=True)
    return caption.split()
