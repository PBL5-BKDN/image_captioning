import torch
from PIL import Image
from torch.utils.data import Dataset
import pickle

from torchvision.transforms import transforms




class ImageCaptionDataset(Dataset):
    def __init__(self, path, word2idx, max_length):
        with open(path, "rb") as f:
            self.data = pickle.load(f)
            self.word2idx = word2idx
            self.max_length = max_length
            self.pad_idx = word2idx["<PAD>"]  # token padding
            self.transform = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

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

        input_sequences = []
        target_sequences = []
        for i in range(1, self.max_length):
            in_seq = caption_tokens[:i]
            out_seq = caption_tokens[i]
            in_seq = in_seq + [self.pad_idx] * (self.max_length - len(in_seq))
            input_sequences.append(in_seq)
            target_sequences.append(out_seq)

        image_path = data["image_path"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image, torch.tensor(input_sequences, dtype=torch.long), torch.tensor(target_sequences, dtype=torch.long)



