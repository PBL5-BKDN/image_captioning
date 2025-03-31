import torch

import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import Inception_V3_Weights
import numpy as np
# Load mô hình InceptionV3 đã được pretrain, loại bỏ lớp Fully Connected (fc)


# Chuyển ảnh sang tensor


def extract_features(image_path) -> np.ndarray[2048]:
    """
    Hàm trích xuất đặc trưng từ ảnh
    :param image_path:
    :return: np.array[2048]
    """
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)




    return features.squeeze().numpy()

print(extract_features("/Users/nguyenhuynh/Documents/3.5-year-DUT/pbl5/image_captioning/data/ktvic_dataset/train-images/00000000002.jpg").shape)