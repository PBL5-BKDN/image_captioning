import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from torchvision.transforms import transforms
import os
from processing import text_processing




class CustomDataset(Dataset):
    def __init__(self, path, word2idx, max_length):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.word2idx = word2idx
        self.max_length = max_length
        self.pad_idx = word2idx["<PAD>"]  # token padding
        df = pd.read_csv(path, delimiter=',')

        self.data = df['image']
        self.labels = df['caption']
        print(f"Loaded {len(self.data)} images and {len(self.labels)} captions.")
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        images_dir = "C:/Users/NguyenPC/Desktop/python_prj/dataset/Images"

        img_path = os.path.join(images_dir, self.data[index])
        image = Image.open(img_path).convert('RGB')
        image_transformed = self.transform(image)

        label = self.labels[index]

        caption_tokens = [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in label.split()]
        caption_tokens = caption_tokens[:self.max_length]
        caption_tokens += [self.pad_idx] * (self.max_length - len(caption_tokens))
        input_sequence = torch.tensor(caption_tokens, dtype=torch.long)
        return image, image_transformed, input_sequence


