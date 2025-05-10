from matplotlib import pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
from settings import DEVICE, min_delta, patience

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

def train(train_dataloader, val_dataloader, model, epochs, optimizer,criterion ,vocab, config, save_path):
    best_val_loss = float("inf")
    stopping_counter = 0
    train_losses = []
    val_losses = []
    bleu_scores = []
    model = model.to(DEVICE)
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        train_loader = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs} [Training]")
        for images, inputs, targets in train_loader:
            images, inputs, targets = images.to(DEVICE), inputs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            output = model(images, inputs)  # (B, MAX_LEN - 1, VOCAB_SIZE)

            loss = criterion(output.reshape(-1, vocab.vocab_size), targets.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_test_loss = 0
        total_bleu_score = 0
        num_samples = 0
        test_loader = tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{epochs} [Validation]")
        with torch.no_grad():
            for images, inputs, targets in test_loader:
                images, inputs, targets = images.to(DEVICE), inputs.to(DEVICE), targets.to(DEVICE)
                output = model(images, inputs)

                loss = criterion(output.reshape(-1, vocab.vocab_size), targets.reshape(-1))
                total_test_loss += loss.item()

                predictions = output.argmax(dim=-1)  # (B, seq_len)
                for pred, tgt in zip(predictions, targets):
                    num_samples += 1
                    pred_caption = tensor_to_caption(pred, vocab.i2w)
                    ref_caption = [tensor_to_caption(tgt, vocab.i2w)]  # cần bọc trong list cho nltk

                    bleu = sentence_bleu(
                        ref_caption,
                        pred_caption,
                        smoothing_function=SmoothingFunction().method1  # giúp tránh BLEU=0 khi không trùng
                    )
                    total_bleu_score += bleu

        avg_test_loss = total_test_loss / len(val_dataloader)
        val_losses.append(avg_test_loss)

        avg_bleu = total_bleu_score / num_samples
        bleu_scores.append(avg_bleu)
        print(
            f"\nEpoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | BLEU: {avg_bleu:.4f}")

        with open("training_log.txt", "a") as f:
            f.write(
                f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_test_loss:.4f}, BLEU: {avg_bleu:.4f}\n")

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

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(val_losses, label="Validation Loss", marker="o")
    plt.plot([b * 100 for b in bleu_scores], label="BLEU (%)", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss / BLEU")
    plt.title("Training & Validation Loss + BLEU Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("train_loss.png")  # Save BEFORE show
    plt.show()
    plt.close()

import numpy as np
import random
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_checkpoint(model_class, checkpoint_path, optimizer=None):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model = model_class(**checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer :
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint.get('epoch', 0) + 1
    return model, optimizer, start_epoch

set_seed(42)