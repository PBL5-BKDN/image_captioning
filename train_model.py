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
criterion = nn.CrossEntropyLoss(ignore_index=vocab.w2i["<PAD>"])
optimizer = torch.optim.Adam(image_caption_model.parameters(), lr= learning_rate)
image_caption_model.to(device)

best_val_loss = float("inf")
stopping_counter = 0
print(device)
for epoch in range(epochs):
    image_caption_model.train()
    total_train_loss = 0
    for image, input, target in train_dataloader:
        # image (batch_size, 3, 299, 299)
        # input (batch_size, max_length - 1, max_length)
        # target (batch_size, max_length - 1)

        image, input, target = image.to(device), input.to(device), target.to(device)
        output = image_caption_model(image, input)
        print(output.size())
        target = target.reshape(-1)
        output = output.reshape(-1, vocab.vocab_size)
        print(output.size())
        print(target.size())
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_dataloader)


    image_caption_model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for image, input, target in test_dataloader:
            image, input, target = image.to(device), input.to(device), target.to(device)
            output = image_caption_model(image, input)
            loss = criterion(output, target)
            total_test_loss += loss.item()
    avg_test_loss = total_test_loss / len(test_dataloader)
    print(f"Epoch {epoch} train loss: {avg_train_loss} test loss: {avg_test_loss}")

    if avg_test_loss < best_val_loss:
        best_val_loss = avg_test_loss
        stopping_counter = 0
        torch.save(image_caption_model.state_dict(), "best_model.pth")
    else:
        stopping_counter += 1
        if stopping_counter == patience:
            print("Early stopping")
            break










