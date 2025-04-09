import pandas

from model.GloveEmbedding import load_glove_embeddings

TRAIN_DATA_PATH = "dataset/captions_train.txt"
TEST_DATA_PATH = "dataset/captions_test.txt"

train_df = pandas.read_csv(TRAIN_DATA_PATH)
word_counts = {}  # a dict : { word : number of appearances}
max_length = 0
for text in train_df['caption']:
  words = text.split()
  max_length = len(words) if (max_length < len(words)) else max_length
  for w in words:
    try:
      word_counts[w] +=1
    except:
        word_counts[w] = 1


# Chỉ lấy các từ xuất hiện trên 10 lần
word_count_threshold = 5
vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold] + ["<PAD>", "<UNK>"]


i2w = {}
w2i = {}

id = 0
for w in vocab:
    w2i[w] = id
    i2w[id] = w
    id += 1


def tensor_to_caption(tensor, i2w):
    words = []
    for idx in tensor:
        word = i2w.get(idx.item(), "<UNK>")
        if word == "<END>":
            break
        if word not in ("<PAD>", "<START>"):
            words.append(word)
    return words



import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm
from data.CustomDataset import CustomDataset

from model.ImageCaptioningModel_transformer import ImageCaptionModel
import matplotlib.pyplot as plt

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


EMBED_DIM = 200
NUM_HEADS = 4
UNITS = 256
IMAGE_SIZE = 224

VOCAB_SIZE = len(vocab)
MAX_SEQ_LEN = max_length
BATCH_SIZE = 256


learning_rate = 0.001
epochs = 50
patience = 3
min_delta = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
from pickle import load, dump
glove_tensor = load_glove_embeddings("model/glove.6B/glove.6B.200d.txt", w2i,200)
# Khởi tạo mô hình
model = ImageCaptionModel(
    w2i=w2i,
    glove_tensor=glove_tensor,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    units=UNITS,
    vocab_size=VOCAB_SIZE,
    max_len=MAX_SEQ_LEN
).to(device)


test_dataset = CustomDataset(
    TEST_DATA_PATH,
    w2i,
    max_length)

train_dataset = CustomDataset(
    TRAIN_DATA_PATH,
    w2i,
    max_length)


def collate_fn(batch):
    images, caption_tokens = zip(*batch)
    caption_tokens = torch.stack(caption_tokens, dim=0)  # (B, max_length)

    # Tách input và target
    input_sequences = caption_tokens[:, :-1].clone().detach() # BỎ END
    target_sequences = caption_tokens[:, 1:].clone().detach() # BỎ START
    images = torch.stack(images, dim=0)

    return images, input_sequences, target_sequences

test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn)


criterion = nn.CrossEntropyLoss(ignore_index=w2i["<PAD>"])
optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)



def train():
    best_val_loss = float("inf")
    stopping_counter = 0
    train_losses = []
    val_losses = []
    blue_scores = []
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        train_loader = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs} [Training]")
        for images, inputs, targets in train_loader:

            images, inputs, targets = images.to(device), inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            output = model(images, inputs) #(B, MAX_LEN - 1, VOCAB_SIZE)

            loss = criterion(output.reshape(-1, VOCAB_SIZE), targets.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()



        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        bleu_scores = []
        model.eval()
        total_test_loss = 0

        test_loader = tqdm(test_dataloader, desc=f"Epoch {epoch + 1}/{epochs} [Validation]")
        with torch.no_grad():
            for images, inputs, targets in test_loader:
                images, inputs, targets = images.to(device), inputs.to(device), targets.to(device)
                output = model(images, inputs)

                loss = criterion(output.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
                total_test_loss += loss.item()

                predictions = output.argmax(dim=-1)  # (B, seq_len)
                for pred, tgt in zip(predictions, targets):
                    pred_caption = tensor_to_caption(pred, i2w)
                    ref_caption = [tensor_to_caption(tgt, i2w)]  # cần bọc trong list cho nltk

                    bleu = sentence_bleu(
                        ref_caption,
                        pred_caption,
                        smoothing_function=SmoothingFunction().method1  # giúp tránh BLEU=0 khi không trùng
                    )
                    bleu_scores.append(bleu)

        avg_test_loss = total_test_loss / len(test_dataloader)
        val_losses.append(avg_test_loss)
        avg_bleu = sum(bleu_scores) / len(bleu_scores)
        blue_scores.append(avg_bleu)
        print(f"\nEpoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | BLEU: {avg_bleu:.4f}")

        if avg_test_loss < best_val_loss - min_delta:
            best_val_loss = avg_test_loss
            stopping_counter = 0
            torch.save(model.state_dict(), "best_model_flick_v2.pth")  # Lưu model tốt nhất
        else:
            stopping_counter += 1
            if stopping_counter >= patience:
                print("Early stopping triggered!")
                break




    # Vẽ biểu đồ
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(val_losses, label="Validation Loss", marker="o")
    plt.plot(blue_scores, label="BLEU", marker="o")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation & BLUE Loss")
    plt.grid()
    plt.show()
    plt.savefig("train_loss.png")


if __name__ == "__main__":
    train()