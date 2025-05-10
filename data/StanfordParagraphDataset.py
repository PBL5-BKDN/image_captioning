import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from torchvision.transforms import transforms
from dataset.stanford_Image_Paragraph_Captioning_dataset.build_vocab import vocab
from processing.image_processing import process_image
from settings import BASE_DIR, BATCH_SIZE
import os

class StandfordParagraphDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        self.image_paths = df["Image_name"]
        self.captions = df["Paragraph"]
        self.vocab = vocab
        self.word2idx = vocab.w2i
        self.max_length = vocab.MAX_LENGTH
        self.pad_idx = vocab.w2i["<PAD>"]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        caption = '<START> ' + self.captions[idx] + ' <END>'


        caption_tokens = [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in caption.split()]
        # Cắt hoặc padding caption để có đúng self.max_length token
        caption_tokens = caption_tokens[:self.max_length]
        caption_tokens += [self.pad_idx] * (self.max_length - len(caption_tokens))

        input_sequence = torch.tensor(caption_tokens, dtype=torch.long)


        image = Image.open(image_path).convert("RGB")
        transformed_image = process_image(image)

        return image, transformed_image, input_sequence




