import os
import pandas as pd
from processing.text_processing import caption_preprocessing
from settings import BASE_DIR

image_dir = os.path.join(BASE_DIR, "dataset/flickr30k/flickr30k-images")

def process_data(df, output_file):
    # Giải nén danh sách caption trong cột 'raw' thành nhiều dòng
    df = df[["filename", "raw"]].copy()
    df = df.explode("raw", ignore_index=True)
    df["caption"] = df["raw"].apply(caption_preprocessing)
    df["image_path"] = df["filename"].apply(lambda x: os.path.join(image_dir, x))
    df[["image_path", "caption"]].to_csv(output_file, index=False)

# Đọc toàn bộ file gốc
df = pd.read_csv("flickr_annotations_30k.csv", converters={"raw": eval})  # eval để chuyển từ str sang list

# Xuất từng tập dữ liệu
splits = {"train": "train.csv", "val": "val.csv", "test": "test.csv"}
for split, filename in splits.items():
    process_data(df[df["split"] == split], filename)