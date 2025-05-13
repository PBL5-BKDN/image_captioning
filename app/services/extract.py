import io
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, pipeline
import torch

def extract_text_from_image(image_bytes, device="cpu"):
    # Load image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    pipe = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
    caption = pipe(image)
    print(caption)
    return caption[0]["generated_text"]
