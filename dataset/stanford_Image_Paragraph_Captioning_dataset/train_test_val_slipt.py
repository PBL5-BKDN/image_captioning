import os
import re

import nltk
import pandas as pd
from nltk.tokenize import word_tokenize

from settings import BASE_DIR

nltk.download('punkt', quiet=True)
# Tải punkt_tab nếu chưa có (chỉ cần chạy 1 lần)

df = pd.read_csv("stanford_df_rectified.csv")

image_path = os.path.join(BASE_DIR, 'dataset/stanford_Image_Paragraph_Captioning_dataset/stanford_img/content/stanford_images')


def _caption_preprocessing(text):
    text = text.lower().strip()  # Chuyển thường và loại bỏ khoảng trắng đầu/cuối
    text = re.sub(r"[^a-z0-9.,!?;'\s]", '', text)  # Loại bỏ ký tự không mong muốn
    text = re.sub(r'\s+', ' ', text)  # Loại bỏ khoảng trắng thừa
    text = word_tokenize(text.lower())
    return  ' '.join(text)


def train_test_val_split(df: pd.DataFrame, split_type: str, save_path: str):
    subset = df[df[split_type] == True].copy()

    # Drop 2 cột còn lại không thuộc split hiện tại
    drop_cols = {"train", "val", "test"}
    subset = subset.drop(columns=list(drop_cols))

    # Chuẩn hóa đường dẫn và caption
    subset["Image_name"] = subset["Image_name"].apply(lambda value: os.path.join(image_path, str(value) + ".jpg"))
    subset["Paragraph"] = subset["Paragraph"].apply(_caption_preprocessing)

    subset.to_csv(save_path, index=False)


train_test_val_split(df, "train", "train.csv")
train_test_val_split(df, "val", "val.csv")
train_test_val_split(df, "test", "test.csv")
