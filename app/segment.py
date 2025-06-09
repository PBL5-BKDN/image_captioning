import os.path

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time

from app.model.model import load_model
from settings import BASE_DIR

model_path = os.path.join(BASE_DIR, "app", "model", "deeplabv3plus_best.pth")
model = load_model(model_path, num_classes=5)

COLORS = [    
    (0, 255, 0),       # lớp 0
    (255, 0, 0),       # lớp 1
    (0, 0, 255),       # lớp 2
    (0, 0, 0),         # lớp 3 - nền
]

def decode_segmap(mask, num_classes):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(num_classes):
        color_mask[mask == cls] = COLORS[cls]
    return color_mask

def show_prediction(model, dataset, device, idx=0, num_classes=4):
    model.eval()
    with torch.no_grad():
        image, mask = dataset[idx]  # image: CxHxW, mask: HxW
        input_tensor = image.unsqueeze(0).to(device)
        output = model(input_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        # Unnormalize ảnh
        image_np = image.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * np.array([0.229, 0.224, 0.225]) +
                    np.array([0.485, 0.456, 0.406]))
        image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)

        # Mask màu
        pred_color = decode_segmap(pred_mask, num_classes)
        mask_color = decode_segmap(mask.numpy(), num_classes)

        overlay = cv2.addWeighted(image_np, 0.6, pred_color, 0.4, 0)

        # # Hiển thị
        # plt.figure(figsize=(15, 5))
        # plt.subplot(1, 4, 1)
        # plt.imshow(image_np)
        # plt.title("Ảnh gốc")
        # plt.axis('off')
        #
        # plt.subplot(1, 4, 2)
        # plt.imshow(mask_color)
        # plt.title("Mask Ground Truth")
        # plt.axis('off')
        #
        # plt.subplot(1, 4, 3)
        # plt.imshow(pred_color)
        # plt.title("Mask Dự đoán")
        # plt.axis('off')
        #
        # plt.subplot(1, 4, 4)
        # plt.imshow(overlay)
        # plt.title("Overlay")
        # plt.axis('off')
        #
        # plt.tight_layout()
        # plt.show()

def analyze_position(pred):
    h, w = pred.shape
    bottom = pred[-h // 4:, :]  # 1/4 dưới ảnh

    # Kiểm tra lớp trong vùng đáy
    unique_bottom = np.unique(bottom)
    
    # Nếu đang đứng trên vỉa hè hoặc vạch kẻ đường
    if 2 in unique_bottom:
        guidance = "Bạn đang đứng trên vỉa hè."
        return guidance
    elif 0 in unique_bottom:
        guidance = "Bạn đang đứng trên vạch kẻ đường cho người đi bộ."
        return guidance

    # Nếu không ở vị trí an toàn → tìm hướng di chuyển
    guidance = "Đang ở khu vực không an toàn! "

    # Chia ảnh theo chiều dọc làm 2 phần
    left = pred[:, :w // 2]
    right = pred[:, w // 2:]

    def contains(region, cls):
        return cls in np.unique(region)

    # Ưu tiên hướng phải
    if contains(right, 2):
        guidance += "Hãy đi về bên phải để lên vỉa hè."
    elif contains(right, 0):
        guidance += "Hãy đi về bên phải để tới vạch kẻ đường."
    elif contains(left, 2):
        guidance += "Hãy đi về bên trái để lên vỉa hè."
    elif contains(left, 0):
        guidance += "Hãy đi về bên trái để tới vạch kẻ đường."
    else:
        guidance += "Không tìm thấy vỉa hè hoặc vạch kẻ đường."

    return guidance


def predict_image(img, device='cuda', num_classes=4):

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize về 512x384
    transform = A.Compose([
        A.Resize(384, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    aug = transform(image=img_rgb)
    input_tensor = aug['image'].unsqueeze(0).to(device)

    # Suy luận
    with torch.no_grad():
        start = time.time()
        output = model(input_tensor)  # (1, C, H, W)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        end = time.time()
        print(f"⏱ Thời gian suy luận: {(end - start)*1000:.2f} ms")

    # Phân tích logic đứng
    res = analyze_position(pred)

    # Overlay mask màu lên ảnh
    pred_color = decode_segmap(pred, num_classes)
    pred_color_bgr = cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR)

    img_resized = cv2.resize(img, (512, 384))  # Resize ảnh gốc cho khớp
    overlay = cv2.addWeighted(img_resized, 0.5, pred_color_bgr, 0.5, 0)

    output_path = os.path.join(BASE_DIR, "app", "output")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(os.path.join(output_path, f"overlay{time.time()}.png"), overlay)
    print(f"✅ Overlay đã lưu tại: {output_path}")
    return res

