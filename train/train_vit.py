import os.path
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset.flickr30k.Flickr30kDataset import Flickr30kDataset
from dataset.stanford_Image_Paragraph_Captioning_dataset.build_vocab import Vocab
from model.VIT import VIT

from settings import BASE_DIR, DEVICE
from train.helper import train, collate_fn
print("Using device:", DEVICE)
WORD_COUNT_THRESHOLD = 5
EMBED_DIM = 200
NUM_HEADS = 4
UNITS = 512
BATCH_SIZE = 512

learning_rate = 0.0001
epochs = 100
patience = 5
min_delta = 0.001
MAX_LEN = 30
vocab_path = os.path.join(BASE_DIR, "dataset/flickr30k/train.csv")

vocab = Vocab(vocab_path, WORD_COUNT_THRESHOLD=WORD_COUNT_THRESHOLD, column="caption")
print("DEVICE: ", DEVICE)

config = {
    "vocab": vocab,
    "embed_dim":EMBED_DIM,
    "num_heads": NUM_HEADS,
    "num_blocks": 3,
    "max_len": vocab.MAX_LENGTH,
    "mlp_hidden_size":UNITS,
}

# Khởi tạo mô hình
model = VIT(
  **config,
).to(DEVICE)


train_path = os.path.join(BASE_DIR, "dataset/flickr30k/train.csv")
val_path = os.path.join(BASE_DIR, "dataset/flickr30k/val.csv")


train_dataset = Flickr30kDataset(train_path, word2idx=vocab.w2i, max_length= MAX_LEN)
val_dataset = Flickr30kDataset(val_path,word2idx=vocab.w2i, max_length = MAX_LEN)

total_params = sum(p.numel() for p in model.parameters())
print(f"Tổng số tham số: {total_params:,}")
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Số tham số có thể huấn luyện: {trainable_params:,}")

val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn)


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

criterion = nn.CrossEntropyLoss(ignore_index=vocab.w2i["<PAD>"], label_smoothing=0.05)

if __name__ == "__main__":
    train(
        train_dataloader,
        val_dataloader,
        model=model,
        epochs=epochs,
        optimizer=optimizer,
        criterion=criterion,
        vocab=vocab,
        config=config,
        model_name="vit")


