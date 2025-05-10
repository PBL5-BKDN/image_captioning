from PIL import Image
import io
import pytesseract
from torchvision import transforms

def extract_text_from_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image  = transforms(image)
    text = pytesseract.image_to_string(image, lang='eng')
    print(text)
    return text
