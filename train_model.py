import fasttext
import torch
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader

from data.ImageCaptionDataset import ImageCaptionDataset
from data.vocab import Vocab
from model.ImageCaptioningModel import ImageCaptioningModel
import fasttext.util
import torch.nn as nn
fasttext.util.download_model("vi", if_exists="ignore")

batch_size = 256
epochs = 100
patience = 5
learning_rate = 0.001

min_delta = 0.001  # Mức    cải thiện tối thiểu để tránh dừng quá sớm
stopping_counter = 0

train_data_path = "data/train_data_preprocessed.pkl"
test_data_path = "data/test_data_preprocessed.pkl"

vocab = Vocab("data/train_data_preprocessed.pkl")
print(vocab.max_length_caption)
print(vocab.w2i["<START>"])
print(vocab.w2i["<END>"])
print(vocab.w2i["<PAD>"])
print(vocab.w2i["<UNK>"])
transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
            ])

test_dataset = ImageCaptionDataset(
    test_data_path,
    vocab.w2i,
    vocab.max_length_caption,
    transform= transforms)

train_dataset = ImageCaptionDataset(
    train_data_path,
    vocab.w2i,
    vocab.max_length_caption,
    transform= transforms)



test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True)

image_caption_model = ImageCaptioningModel(vocab_size=vocab.vocab_size, image_feature_size=2048, hidden_size=256, vocab=vocab)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
criterion = nn.CrossEntropyLoss(ignore_index=vocab.w2i["<PAD>"])
optimizer = torch.optim.Adam(image_caption_model.parameters(), lr= learning_rate)
image_caption_model.to(device)

best_val_loss = float("inf")
stopping_counter = 0
train_losses = []
val_losses = []


for epoch in range(epochs):
    image_caption_model.train()
    total_train_loss = 0
    train_loader = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs} [Training]")
    for image, captions in train_loader:
        image, captions = image.to(device), captions.to(device)


        optimizer.zero_grad()
        output = image_caption_model(image, captions)  # (batch_size, max_length, vocab_size)

        loss = criterion(output.reshape(-1, vocab.vocab_size), captions.reshape(-1))

        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)

    image_caption_model.eval()
    total_test_loss = 0
    test_loader = tqdm(test_dataloader, desc=f"Epoch {epoch + 1}/{epochs} [Validation]")
    with torch.no_grad():
        for images, captions in test_loader:

            images, captions= images.to(device), captions.to(device)
            output = image_caption_model(images, captions)



            loss = criterion(output.reshape(-1, vocab.vocab_size), captions.reshape(-1))

            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_dataloader)
    val_losses.append(avg_test_loss)
    print(f"\nEpoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

    if avg_test_loss < best_val_loss - min_delta:
        best_val_loss = avg_test_loss
        stopping_counter = 0
        torch.save(image_caption_model.state_dict(), "best_model.pth")  # Lưu model tốt nhất
    else:
        stopping_counter += 1
        if stopping_counter >= patience:
            print("Early stopping triggered!")
            break

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss", marker="o")
plt.plot(val_losses, label="Validation Loss", marker="o")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training & Validation Loss")
plt.grid()
plt.show()
plt.savefig("train_loss.png")









