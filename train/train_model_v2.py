import os.path
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset.flickr30k.Flickr30kDataset import Flickr30kDataset
from dataset.stanford_Image_Paragraph_Captioning_dataset.build_vocab import  Vocab
from model.ImageCaptioningModelV2 import ImageCaptionModelV2

from settings import BASE_DIR, DEVICE
from train.helper import train, collate_fn, load_checkpoint
print("Using device:", DEVICE)
WORD_COUNT_THRESHOLD = 5
EMBED_DIM = 300
NUM_HEADS = 4
UNITS = 256
BATCH_SIZE = 256

learning_rate = 0.05
epochs = 50
patience = 5
min_delta = 0.001
MAX_LEN = 30
vocab_path = os.path.join(BASE_DIR, "dataset/flickr30k/train.csv")

vocab = Vocab(vocab_path, WORD_COUNT_THRESHOLD=WORD_COUNT_THRESHOLD, column="caption")
print("DEVICE: ", DEVICE)


# Khởi tạo mô hình
model = ImageCaptionModelV2(
    vocab,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    vocab_size=vocab.vocab_size,
    max_len=vocab.MAX_LENGTH
).to(DEVICE)


train_path = os.path.join(BASE_DIR, "dataset/flickr30k/train.csv")
val_path = os.path.join(BASE_DIR, "dataset/flickr30k/val.csv")


train_dataset = Flickr30kDataset(train_path, word2idx=vocab.w2i, max_length= MAX_LEN)
val_dataset = Flickr30kDataset(val_path,word2idx=vocab.w2i, max_length = MAX_LEN)



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

save_path = os.path.join(BASE_DIR, "train", "model", "best_model_transformer_v2.pth" )
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# model, optimizer, start_epoch = load_checkpoint(ImageCaptionModelV2,save_path, learning_rate=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=vocab.w2i["<PAD>"], label_smoothing=0.1)


config = {
    "vocab": vocab,
    "max_len": vocab.MAX_LENGTH,
    "embed_dim": EMBED_DIM,
    "num_heads": NUM_HEADS,
    "vocab_size":vocab.vocab_size,
}


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
        save_path=save_path)
