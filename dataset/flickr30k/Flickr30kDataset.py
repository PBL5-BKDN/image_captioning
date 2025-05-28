import os.path

import torch
from PIL import Image
from torch.utils.data import Dataset
import pickle
import pandas as pd

from processing.image_processing import process_image


class Flickr30kDataset(Dataset):
    def __init__(self,path, word2idx, max_length):
        self.data = pd.read_csv(path)
        self.word2idx = word2idx
        self.max_length = max_length
        self.pad_idx = word2idx["<PAD>"]  # token padding


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data.iloc[idx]

        caption = data["caption"]
        tokens = caption.split()[:self.max_length - 2]  # chừa chỗ cho <START> và <END>
        caption_tokens = [self.word2idx["<START>"]] + \
                         [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens] + \
                         [self.word2idx["<END>"]]
        caption_tokens += [self.pad_idx] * (self.max_length - len(caption_tokens))
        input_sequence = torch.tensor(caption_tokens, dtype=torch.long)

        image_path = data["image_path"]
        image = Image.open(image_path).convert("RGB")
        processed_image = process_image(image)

        return image, processed_image, input_sequence




