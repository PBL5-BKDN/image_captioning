import pickle
import os
import json


root_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
from pyvi import ViTokenizer
def load_captions_and_images(json_path: str, saved_path: str, image_dir: str):
    """
    Load captions và images từ file JSON, sau đó lưu vào file .pkl
    :param json_path: Đường dẫn đến file JSON chứa dữ liệu
    :param saved_path: Đường dẫn lưu file .pkl
    :param image_dir: Thư mục chứa ảnh
    :return: None
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]
    captions_data = []

    for annotation in annotations:
        image_filename = next((img["filename"] for img in images if img["id"] == annotation["image_id"]), None)
        image_path = os.path.join(root_dir, image_dir, image_filename)
        caption = annotation["caption"]
        segment_caption = ViTokenizer.tokenize(caption)
        captions_data.append({
            **annotation,
            "segment_caption": segment_caption,
            "image_path": image_path,
        })

    # Lưu dữ liệu vào file .pkl
    with open(saved_path, "wb") as f:
        pickle.dump(captions_data, f)

    print(f"Saved {len(captions_data)} samples to {saved_path}")

# # Chạy hàm để lưu dữ liệu vào file .pkl
load_captions_and_images(
    "../data/ktvic_dataset/train_data.json",
    "../data/train_data_preprocessed.pkl",
    "data/ktvic_dataset/train-images"
)

load_captions_and_images(
    "../data/ktvic_dataset/test_data.json",
    "../data/test_data_preprocessed.pkl",
    "data/ktvic_dataset/public-test-images"
)

