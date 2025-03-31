import os
import json
from collections import Counter

from pyvi.ViTokenizer import ViTokenizer

import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset
from PIL import Image

class ImageCaptionDataset(Dataset):
    def __init__(self, json_path, img_dir, tokenizer_name, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer_name

        # Load dữ liệu từ file JSON
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Tạo dictionary để map image_id -> filename
        self.image_id_to_filename = {img["id"]: img["filename"] for img in data["images"]}

        # Tạo danh sách (image_path, caption)
        self.data = [
            (self.image_id_to_filename[ann["image_id"]], ann["caption"])
            for ann in data["annotations"]
            if ann["image_id"] in self.image_id_to_filename  # Chỉ lấy các annotation có ảnh hợp lệ
        ]
        # Xây dựng word2idx từ train_caption
        self.word2idx = self.build_word2idx([caption for _, caption in self.data], 2)

    def build_word2idx(self, train_caption, min_freq):
        """
        Tạo word2idx từ tập train_caption
        :param train_caption: Danh sách các câu chú thích
        :param min_freq: Số lần xuất hiện tối thiểu để một từ được thêm vào từ điển
        :return: word2idx (dict)
        """
        word_counter = Counter()
        for caption in train_caption:
            tokens = ViTokenizer.tokenize(caption).split()  # Tokenize tiếng Việt
            word_counter.update(tokens)

        # Lọc những từ có tần suất ≥ min_freq
        vocab = [word for word, freq in word_counter.items() if freq >= min_freq]

        # Tạo word2idx
        word2idx = {word: idx for idx, word in enumerate(vocab, start=1)}

        # Thêm token đặc biệt
        word2idx["<PAD>"] = 0  # Padding
        word2idx["<UNK>"] = len(word2idx)  # Từ không có trong từ điển

        return word2idx
    def encode_caption(self, caption):
        tokens = ViTokenizer.tokenize(caption).split()
        return [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_filename, caption = self.data[idx]
        full_img_path = os.path.join(self.img_dir, img_filename)

        # Load ảnh
        image = Image.open(full_img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)


        # Mã hóa caption
        caption_ids = self.encode_caption(caption)
        caption_tensor = torch.tensor(caption_ids, dtype=torch.long)


        return image, caption_tensor

