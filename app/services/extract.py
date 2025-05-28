import io

from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2TokenizerFast

from settings import DEVICE

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(DEVICE)
tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")


def extract_text_from_image(image_bytes):
    # Load image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = image_processor(image, return_tensors="pt").to(DEVICE)
    output = model.generate(**img)
    caption = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    print(caption)
    return caption
