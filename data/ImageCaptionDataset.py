import torch
from PIL import Image
from torch.utils.data import Dataset
import pickle

class ImageCaptionDataset(Dataset):
    def __init__(self, path, word2idx, max_length, transform=None):
        with open(path, "rb") as f:
            self.data = pickle.load(f)
            self.word2idx = word2idx
            self.max_length = max_length
            self.pad_idx = word2idx["<PAD>"]  # token padding
            self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]


        #Chuyển caption thành index
        caption = data["segment_caption"]
        caption_tokens = [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in caption.split()]
        # Cắt hoặc padding caption để có đúng self.max_length token
        caption_tokens = caption_tokens[:self.max_length]
        caption_tokens += [self.pad_idx] * (self.max_length - len(caption_tokens))

        input_sequence = torch.tensor(caption_tokens, dtype=torch.long)

        image_path = data["image_path"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image, input_sequence



