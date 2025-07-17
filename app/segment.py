import os.path

import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time

from app.model.model import load_model
from app.model.model import infer_and_measure
from settings import BASE_DIR
from PIL import Image, ImageDraw, ImageFont
model_path = os.path.join(BASE_DIR, "app", "model", "deeplabv3plus_best.pth")
model = load_model(model_path, num_classes=4)
# Màu overlay cho từng lớp
COLORS = [
    (0, 255, 0),       # lớp 0
    (255, 0, 0),       # lớp 1
    (0, 0, 255),       # lớp 2
    (0, 0, 0),         # lớp 3 - nền
]

# Hàm decode mask thành màu
def decode_segmap(mask, num_classes):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(num_classes):
        color_mask[mask == cls] = COLORS[cls]
    return color_mask

# Hàm phân tích vị trí
def analyze_position(pred):
    h, w = pred.shape

    # Xác định vùng đáy ở giữa ảnh (1/4 dưới và phần giữa theo chiều ngang)
    vertical_start = h * 3 // 4
    vertical_end = h
    horizontal_start = w // 3
    horizontal_end = w * 2 // 3

    bottom_center = pred[vertical_start:vertical_end, horizontal_start:horizontal_end]
    total_pixels = bottom_center.size

    # Đếm số lượng pixel của từng lớp trong vùng này
    unique, counts = np.unique(bottom_center, return_counts=True)
    class_counts = dict(zip(unique, counts))

    # Ngưỡng 50%
    threshold = 0.5 * total_pixels

    # Kiểm tra nếu đang ở vùng an toàn
    if class_counts.get(2, 0) > threshold:
        return "Bạn đang đứng trên vỉa hè."
        #return ""  
    elif class_counts.get(0, 0) > threshold:
        return "Bạn đang đứng trên vạch kẻ đường cho người đi bộ."
        #return ""
    elif class_counts.get(3, 0) > 0.8 * total_pixels:
        return ""
    # Nếu không an toàn → tìm hướng di chuyển
    guidance = "Đang ở khu vực không an toàn! "

    # Chia ảnh theo chiều ngang: trái / phải
    left = pred[:, :w // 2]
    right = pred[:, w // 2:]

    def contains(region, cls, ratio=0.1):
        region_total = region.size
        region_counts = dict(zip(*np.unique(region, return_counts=True)))
        return region_counts.get(cls, 0) > ratio * region_total

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
    import os
    from PIL import Image, ImageDraw, ImageFont

    # Chuyển ảnh BGR (OpenCV) sang RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Áp dụng transform
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
        output, infer_time_ms = infer_and_measure(model, input_tensor)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        print(f"⏱ Thời gian suy luận: {infer_time_ms:.2f} ms")

    # Phân tích logic đứng
    res = analyze_position(pred)

    # Overlay mask màu lên ảnh
    pred_color = decode_segmap(pred, num_classes)
    pred_color_bgr = cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR)

    img_resized = cv2.resize(img, (512, 384))  # Resize ảnh gốc cho khớp
    overlay = cv2.addWeighted(img_resized, 0.5, pred_color_bgr, 0.5, 0)

    # Vẽ khung vùng phân tích đáy nếu muốn (tùy chọn)
    h, w = pred.shape
    vertical_start = h * 3 // 4
    vertical_end = h
    horizontal_start = w // 3
    horizontal_end = w * 2 // 3
    cv2.rectangle(overlay, (horizontal_start, vertical_start), (horizontal_end, vertical_end), (0, 255, 255), 2)

    # Ghi kết quả tiếng Việt lên ảnh (dùng PIL để hỗ trợ Unicode)
    overlay_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(overlay_pil)
    font_path = os.path.join(BASE_DIR, "static", "Roboto_Condensed-Black.ttf")
    try:
        font = ImageFont.truetype(font_path, 16)
    except:
        font = ImageFont.load_default()

    # Ghi nội dung kết quả ở góc trái trên
    draw.text((10, 10), res, font=font, fill=(255, 255, 255))

    # Chuyển lại sang OpenCV để lưu
    overlay = cv2.cvtColor(np.array(overlay_pil), cv2.COLOR_RGB2BGR)

    # Lưu ảnh
    output_path = os.path.join(BASE_DIR, "app", "output")
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"overlay_{int(time.time())}.png")
    cv2.imwrite(output_file, overlay)

    print(f"Overlay đã lưu tại: {output_file}")
    return res

