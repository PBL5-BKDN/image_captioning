import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import nltk
from nltk.translate.bleu_score import corpus_bleu
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import TrainingArguments

from data.CustomDataset import transform
from data.dataset import ImageCaptionDataset
from model import ImageCaptioningModel
from pyvi.ViTokenizer import tokenize

# Tải NLTK package nếu chưa có
nltk.download('punkt')

# Đường dẫn dữ liệu
train_json_path = "/Users/nguyenhuynh/Documents/3.5-year-DUT/pbl5/image_captioning/data/ktvic_dataset/train_data.json"
img_train_dir = "/Users/nguyenhuynh/Documents/3.5-year-DUT/pbl5/image_captioning/data/ktvic_dataset/train-images"

test_json_path = "/Users/nguyenhuynh/Documents/3.5-year-DUT/pbl5/image_captioning/data/ktvic_dataset/test_data.json"
img_test_dir = "/Users/nguyenhuynh/Documents/3.5-year-DUT/pbl5/image_captioning/data/ktvic_dataset/public-test-images"

# Load dataset
train_dataset = ImageCaptionDataset(train_json_path, img_train_dir, tokenize, transform=transform)
test_dataset = ImageCaptionDataset(test_json_path, img_test_dir, tokenize, transform=transform)

import fasttext
import numpy as np
import torch
import fasttext.util
fasttext.util.download_model('vi', if_exists='ignore')
# Load pre-trained FastText model cho tiếng Việt
ft_model = fasttext.load_model('cc.vi.300.bin')  # File pre-trained FastText tiếng Việt

# Kiểm tra thiết bị
device = "cuda" if torch.cuda.is_available() else "cpu"
# Tạo embedding matrix
vocab_size = len(train_dataset.word2idx)  # Số từ trong từ điển
embed_size = 300  # Kích thước vector của FastText
embedding_matrix = np.zeros((vocab_size, embed_size))

# Điền vector embedding cho từng từ
for word, idx in train_dataset.word2idx.items():
    if word in ft_model:
        embedding_matrix[idx] = ft_model[word]
    else:
        embedding_matrix[idx] = np.random.normal(size=(embed_size,))

embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32, device=device)

def collate_fn(batch):
    images, captions = zip(*batch)

    captions_padded = pad_sequence(captions, batch_first=True, padding_value=0)
    print(f"captions_padded shape: {captions_padded.shape}")

    images = torch.stack(images)
    return images, captions_padded

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)



# Khởi tạo mô hình
model = ImageCaptioningModel(vocab_size=1953, embedding_matrix=embedding_matrix).to(device)

# Cấu hình loss function và optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Số epochs
num_epochs = 10

# Danh sách lưu loss và BLEU score để vẽ biểu đồ
train_losses, test_losses, bleu_scores = [], [], []

# === HUẤN LUYỆN MÔ HÌNH ===
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for images, captions in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):


        images, captions = images.to(device), captions  # Captions là text, không cần to(device)

        optimizer.zero_grad()
        outputs = model(images, captions)


        print(f"outputs shape: {outputs.shape}")  # (batch_size, seq_len, vocab_size)
        print(f"captions shape: {captions.shape}")  # (batch_size, seq_len)

        # Tính loss
        loss = criterion(outputs.reshape(-1, outputs.shape[-1]), captions[:, 1:].reshape(-1))



        # Chuyển captions thành tensor 1D

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_dataloader)
    train_losses.append(train_loss)

    # === ĐÁNH GIÁ MÔ HÌNH ===
    model.eval()
    test_loss = 0
    references, hypotheses = [], []

    with torch.no_grad():
        for images, captions in tqdm(test_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Test]"):
            images, captions = images.to(device), captions

            outputs = model(images, captions)
            captions = captions.to(outputs.device)  # Chuyển về cùng device

            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), captions.reshape(-1))

            test_loss += loss.item()
            # Chuyển outputs thành chuỗi từ
            predicted_ids = outputs.argmax(dim=-1)  # (batch_size, seq_len)
            for ref, pred in zip(captions[:, 1:], predicted_ids):
                ref_tokens = [str(token.item()) for token in ref if token != 0]  # Loại bỏ padding
                pred_tokens = [str(token.item()) for token in pred if token != 0]
                references.append([ref_tokens])
                hypotheses.append(pred_tokens)


    test_loss /= len(test_dataloader)
    test_losses.append(test_loss)

    # Tính BLEU Score
    bleu_score = corpus_bleu(references, hypotheses)
    bleu_scores.append(bleu_score)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, BLEU Score: {bleu_score:.4f}")

# === VẼ BIỂU ĐỒ LOSS VÀ BLEU SCORE ===
plt.figure(figsize=(12, 5))

# Biểu đồ loss
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss", marker="o")
plt.plot(range(1, num_epochs + 1), test_losses, label="Test Loss", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Train vs Test Loss")

# Biểu đồ BLEU Score
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), bleu_scores, label="BLEU Score", marker="o", color="g")
plt.xlabel("Epoch")
plt.ylabel("BLEU Score")
plt.legend()
plt.title("BLEU Score Over Epochs")

plt.savefig("training_results.png")  # Lưu thành file PNG
plt.show()

# === LƯU MÔ HÌNH ===
torch.save(model.state_dict(), "image_captioning_model.pth")
print("Mô hình đã được lưu thành công!")
