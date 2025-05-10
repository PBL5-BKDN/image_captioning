import io

from PIL import Image

from processing.image_processing import process_image
from settings import DEVICE


def extract_text_from_image(image_bytes, model):
    image = Image.open(io.BytesIO(image_bytes))
    image = process_image(image).to(DEVICE)
    text = model.generate_caption(image)
    return text
