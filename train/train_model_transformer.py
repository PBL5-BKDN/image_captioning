import os.path
import torch
from torch import nn
from torch.utils.data import DataLoader


from data.StanfordParagraphDataset import StandfordParagraphDataset
from dataset.stanford_Image_Paragraph_Captioning_dataset.build_vocab import vocab
from model.GloveEmbedding import glove_tensor
from model.ImageCaptioningModel_transformer import ImageCaptionModel
from settings import EMBED_DIM, NUM_HEADS, UNITS, DEVICE, learning_rate, epochs, min_delta, patience, \
    BASE_DIR
from train.helper import train, collate_fn

# Khởi tạo mô hình
model = ImageCaptionModel(
    w2i=vocab.w2i,
    glove_tensor=glove_tensor,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    units=UNITS,
    vocab_size=vocab.vocab_size,
    max_len=vocab.MAX_LENGTH
).to(DEVICE)

train_path = os.path.join(BASE_DIR, "dataset/stanford_Image_Paragraph_Captioning_dataset", "train.csv")
val_path = os.path.join(BASE_DIR, "dataset/stanford_Image_Paragraph_Captioning_dataset", "val.csv")

train_dataset = StandfordParagraphDataset(train_path)
val_dataset = StandfordParagraphDataset(val_path)

BATCH_SIZE = 256

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


criterion = nn.CrossEntropyLoss(ignore_index=vocab.w2i["<PAD>"])
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

config = {
    "vocab_size": vocab.vocab_size,
    "max_len": vocab.MAX_LENGTH,
    "embed_dim": EMBED_DIM,
    "num_heads": NUM_HEADS,
    "units": UNITS
}

save_path = os.path.join(BASE_DIR, "best_model", "best_model_transformer_v1.pth" )
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
