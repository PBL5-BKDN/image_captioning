import os

from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from settings import DEVICE, min_delta, patience, BASE_DIR
from torch.cuda.amp import GradScaler, autocast

import torch

def collate_fn(batch):
    images, transformed_image, caption_tokens = zip(*batch)
    caption_tokens = torch.stack(caption_tokens, dim=0)  # (B, max_length)

    # Tách input và target
    input_sequences = caption_tokens[:, :-1].clone().detach()  # BỎ END
    target_sequences = caption_tokens[:, 1:].clone().detach()  # BỎ START
    transformed_image = torch.stack(transformed_image, dim=0)

    return transformed_image, input_sequences, target_sequences

def tensor_to_caption(tensor, i2w):
    words = []
    for idx in tensor:
        word = i2w.get(idx.item(), "<UNK>")
        if word == "<END>":
            break
        if word not in ("<PAD>", "<START>"):
            words.append(word)
    return words

def generate_with_scheduled_sampling(model, images, input_sequences, ss_prob, vocab_size):
    B, T = input_sequences.size()
    device = input_sequences.device
    outputs = []
    inputs = input_sequences[:, 0].unsqueeze(1)  # START token

    for t in range(1, T + 1):
        output = model(images, inputs)  # (B, t, vocab_size)
        next_logits = output[:, -1, :]  # (B, vocab_size)
        outputs.append(next_logits.unsqueeze(1))  # (B, 1, vocab)

        _, predicted = torch.max(next_logits, dim=1)  # (B,)

        if t < T:
            use_model_input = torch.rand(B, device=device) < ss_prob  # (B,)
            next_input = input_sequences[:, t]
            next_input[use_model_input] = predicted[use_model_input]
            inputs = torch.cat([inputs, next_input.unsqueeze(1)], dim=1)

    return torch.cat(outputs, dim=1)  # (B, T, vocab_size)

def train(train_dataloader, val_dataloader, model, epochs, optimizer, criterion, vocab, config, model_name):
    import time
    start_time = time.time()

    save_dir = os.path.join(BASE_DIR, "train", "model", model_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model.pth")

    writer = SummaryWriter(log_dir=save_dir)
    best_val_loss = float("inf")
    stopping_counter = 0


    train_losses = []
    val_losses = []
    bleu_scores = []

    model = model.to(DEVICE)
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ss_start_prob = 0.0
    ss_end_prob = 0.25  # Tăng dần đến 25%

    for epoch in range(epochs):
        ss_prob = ss_start_prob + (ss_end_prob - ss_start_prob) * (epoch / epochs)
        writer.add_scalar("ScheduledSampling/ss_prob", ss_prob, epoch)

        model.train()
        total_train_loss = 0
        train_loader = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs} [Training]")

        for images, inputs, targets in train_loader:
            images, inputs, targets = images.to(DEVICE), inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                output = model(images, inputs)
                loss = criterion(output.reshape(-1, vocab.vocab_size), targets.reshape(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            total_train_loss += loss.item()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_test_loss = 0
        test_loader = tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{epochs} [Validation]")

        with torch.no_grad():
            for images, inputs, targets in test_loader:
                images, inputs, targets = images.to(DEVICE), inputs.to(DEVICE), targets.to(DEVICE)
                output = model(images, inputs)  # Validation vẫn dùng ground truth
                loss = criterion(output.reshape(-1, vocab.vocab_size), targets.reshape(-1))
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(val_dataloader)
        val_losses.append(avg_test_loss)

        print(
            f"\nEpoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_test_loss:.4f}")

        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_test_loss, epoch)

        if avg_test_loss < best_val_loss - min_delta:
            best_val_loss = avg_test_loss
            stopping_counter = 0
            torch.save({
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config
            }, save_path)
        else:
            stopping_counter += 1
            if stopping_counter >= patience:
                print("Early stopping triggered!")
                break

    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

    # Vẽ biểu đồ loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(val_losses, label="Validation Loss", marker="o")
    plt.plot([b * 100 for b in bleu_scores], label="BLEU (%)", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("train_loss.png")
    plt.show()
    plt.close()

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import random
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(model_class, checkpoint_path, learning_rate):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    # Khởi tạo model
    model = model_class(**checkpoint['config'])
    model.to(DEVICE)

    # Tải trọng số
    model.load_state_dict(checkpoint['model_state_dict'])

    # Khởi tạo optimizer với tham số model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Tải trạng thái optimizer nếu có
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = checkpoint.get('epoch', 0) + 1

    return model, optimizer, start_epoch

def draw_loss(file_path):
    import re
    import matplotlib.pyplot as plt

    # Đường dẫn đến file log


    # Danh sách lưu loss theo epoch
    train_losses = []
    val_losses = []
    epochs = []

    # Biểu thức chính quy để tìm loss
    loss_pattern = re.compile(r"Epoch (\d+)/\d+ \| Train Loss: ([\d.]+) \| Val Loss: ([\d.]+)")

    # Đọc và phân tích file log
    with open(file_path, "r") as f:
        for line in f:
            match = loss_pattern.search(line)
            if match:
                epoch = int(match.group(1))
                train_loss = float(match.group(2))
                val_loss = float(match.group(3))
                epochs.append(epoch)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

draw_loss("E:\\python_prj\\train\\model\\vit\\log.txt")
draw_loss("E:\\python_prj\\train\model\cnn_transformer\\log.txt")
set_seed(42)