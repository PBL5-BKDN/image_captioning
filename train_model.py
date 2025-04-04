import fasttext
import torch

from torch.utils.data import DataLoader

from data.ImageCaptionDataset import ImageCaptionDataset
from data.vocab import Vocab
from model.ImageCaptioningModel import ImageCaptioningModel
import fasttext.util
import torch.nn as nn
fasttext.util.download_model("vi", if_exists="ignore")

batch_size = 2
epochs = 100
patience = 5
learning_rate = 0.001
train_data_path = "data/train_data_preprocessed.pkl"
test_data_path = "data/test_data_preprocessed.pkl"

vocab = Vocab("data/train_data_preprocessed.pkl")
print(vocab.max_length_caption)
print(vocab.w2i["<START>"])
print(vocab.w2i["<END>"])
print(vocab.w2i["<PAD>"])
print(vocab.w2i["<UNK>"])
test_dataset = ImageCaptionDataset(test_data_path, vocab.w2i, vocab.max_length_caption)
train_dataset = ImageCaptionDataset(train_data_path, vocab.w2i, vocab.max_length_caption)



test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

image_caption_model = ImageCaptioningModel(vocab_size=vocab.vocab_size, image_feature_size=2048, hidden_size=256, vocab=vocab)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
criterion = nn.CrossEntropyLoss(ignore_index=vocab.w2i["<PAD>"])
optimizer = torch.optim.Adam(image_caption_model.parameters(), lr= learning_rate)
image_caption_model.to(device)

best_val_loss = float("inf")
stopping_counter = 0

for epoch in range(epochs):
    image_caption_model.train()
    total_train_loss = 0
    # for images, inputs, targets in train_dataloader:
    #     # image (batch_size, 3, 299, 299)
    #     # input (batch_size, max_length - 1, max_length)
    #     # target (batch_size, max_length - 1)
    #     optimizer.zero_grad()
    #     images, inputs, targets = images.to(device), inputs.to(device), targets.to(device)
    #     outputs = image_caption_model(images, inputs)
    #
    #     target = targets.reshape(-1)
    #     output = inputs.reshape(-1, vocab.vocab_size)
    #
    #     loss = criterion(output, target)
    #     loss.backward()
    #     optimizer.step()
    #
    #     total_train_loss += loss.item()
    for image, captions in train_dataloader:
        image, captions = image.to(device), captions.to(device)
        print(image.shape, captions.shape)
        optimizer.zero_grad()
        output = image_caption_model(image, captions)  # (batch_size, max_length-1, vocab_size)
        print(output.shape)
        target = captions[:, 1:]  # Bỏ <START>, dự đoán từ từ thứ 2 đến cuối
        print(target.shape)
        loss = criterion(output.reshape(-1, vocab.vocab_size), target.reshape(-1))
        print(loss.item())
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_dataloader)


    image_caption_model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for images, inputs, targets in test_dataloader:

            images, inputs, targets = images.to(device), inputs.to(device), targets.to(device)
            output = image_caption_model(images, input)
            targets = targets.reshape(-1)
            outputs = outputs.reshape(-1, vocab.vocab_size)
            loss = criterion(output, targets)

            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_dataloader)
    print(f"Epoch {epoch+1}/{epochs} train loss :{avg_train_loss} test loss: {avg_test_loss}")

    if avg_train_loss < best_val_loss:
        best_val_loss = avg_train_loss
        stopping_counter = 0
        torch.save(image_caption_model.state_dict(), "best_model.pth")
    else:
        stopping_counter += 1
        if stopping_counter == patience:
            print("Early stopping")
            break










