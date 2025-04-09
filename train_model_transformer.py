import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from data.ImageCaptionDataset import ImageCaptionDataset
from data.vocab import Vocab
from model.ImageCaptioningModel_transformer import ImageCaptioningModel
import matplotlib.pyplot as plt

TRAIN_DATA_PATH = "data/train_data_preprocessed.pkl"
TEST_DATA_PATH = "data/test_data_preprocessed.pkl"
# Định nghĩa các hằng số

EMBED_DIM = 256   # Kích thước vector nhúng
NUM_HEADS = 4    # Số head trong multi-head attention
NUM_LAYERS = 4   # Số layer transformer
IMAGE_SIZE = 224
vocab = Vocab(TRAIN_DATA_PATH)
VOCAB_SIZE = vocab.vocab_size  # Kích thước từ vựng tiếng Việt
MAX_SEQ_LEN = vocab.max_length_caption
BATCH_SIZE = 256


learning_rate = 0.001
epochs = 50
patience = 5
min_delta = 0.001

# Khởi tạo mô hình
model = ImageCaptioningModel(EMBED_DIM, NUM_HEADS, NUM_LAYERS, vocab_size=vocab.vocab_size)

transforms = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


test_dataset = ImageCaptionDataset(
    TEST_DATA_PATH,
    vocab.w2i,
    vocab.max_length_caption,
    transform= transforms)

train_dataset = ImageCaptionDataset(
    TRAIN_DATA_PATH,
    vocab.w2i,
    vocab.max_length_caption,
    transform= transforms)


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



# Hàm sinh caption
def generate_caption(model, image, vocab, max_len=MAX_SEQ_LEN):
    model.eval()
    with torch.no_grad():
        features = model.backbone(image)
        features = model.conv_proj(features).flatten(2).transpose(1, 2)
        features = model.pos_encoding(features)
        memory = model.transformer_encoder(features)

        caption = torch.tensor([[vocab.w2i['<START>']]]).to(image.device)
        for _ in range(max_len):
            output = model.transformer_decoder(memory, caption)
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            caption = torch.cat([caption, next_token], dim=1)
            if next_token.item() == vocab.w2i['<END>']:
                break
        return [vocab.i2w[idx] for idx in caption[0].tolist()]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
criterion = nn.CrossEntropyLoss(ignore_index=vocab.w2i["<PAD>"])
optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)
model.to(device)



def train():
    best_val_loss = float("inf")
    stopping_counter = 0
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        train_loader = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs} [Training]")
        for images, inputs, targets in train_loader:
            images, inputs, targets = images.to(device), inputs.to(device), targets.to(device)


            optimizer.zero_grad()
            output = model(images, inputs) #(B, MAX_LEN - 1, VOCAB_SIZE)

            loss = criterion(output.reshape(-1, vocab.vocab_size), targets.reshape(-1))

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_test_loss = 0
        test_loader = tqdm(test_dataloader, desc=f"Epoch {epoch + 1}/{epochs} [Validation]")
        with torch.no_grad():
            for images, inputs, targets in test_loader:
                images, inputs, targets = images.to(device), inputs.to(device), targets.to(device)
                output = model(images, inputs)
                loss = criterion(output.reshape(-1, vocab.vocab_size), targets.reshape(-1))
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_dataloader)
        val_losses.append(avg_test_loss)
        print(f"\nEpoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

        from PIL import Image

        if (epoch + 1) % 5 == 0:
            print(f"\n[Caption Generation & Save] Epoch {epoch + 1}")
            model.eval()
            fig, axes = plt.subplots(1, 5, figsize=(20, 5))
            for i in range(0,25,5):
                image_tensor, _ = test_dataset[i]
                image_input = image_tensor.unsqueeze(0).to(device)
                caption = generate_caption(model, image_input, vocab)

                # Chuyển tensor ảnh về PIL Image để hiển thị
                image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
                image_np = image_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # Unnormalize
                image_np = (image_np * 255).clip(0, 255).astype("uint8")

                axes[i].imshow(image_np)
                axes[i].set_title(" ".join(caption), fontsize=10)
                axes[i].axis("off")

            plt.tight_layout()
            os.makedirs("plots/samples", exist_ok=True)
            plt.savefig(f"plots/samples/sample_captions_epoch_{epoch + 1}.png")
            plt.close()

        if avg_test_loss < best_val_loss - min_delta:
            best_val_loss = avg_test_loss
            stopping_counter = 0
            torch.save(model.state_dict(), "best_model_v2.pth")  # Lưu model tốt nhất
        else:
            stopping_counter += 1
            if stopping_counter >= patience:
                print("Early stopping triggered!")
                break




    # Vẽ biểu đồ
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

if __name__ == "__main__":
    train()