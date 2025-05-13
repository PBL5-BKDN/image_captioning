import os.path

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from ultralytics import YOLO
from app.services.pipeline import pipeline
import cv2
from fastapi.responses import JSONResponse
from settings import BASE_DIR
import numpy as np
import math
import base64
app = FastAPI(title="Simple FastAPI Server")

model_path = os.path.join(BASE_DIR, "yolo_model/model.pt" )
model = YOLO(model_path)

@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI Server!"}


# Upload image endpoint
@app.post("/upload-image/")
async def upload_image(
        file: UploadFile = File(...),
        question: str = Form(...),
        longitude: float = Form(...),
        latitude: float = Form(...)
):
    # Check if file is an image
    if not file.content_type.startswith("image/"):
        return {
            "error": "File is not an image."
        }

    # Đọc nội dung bytes từ UploadFile
    image_bytes = await file.read()

    # Gọi hàm xử lý (giả định bạn có pipeline được định nghĩa sẵn)
    output = pipeline(image_bytes, question, longitude, latitude)

    return {"data": output}


# @app.post('/detect')
# def detect_objects():
#     if 'image' not in request.files:
#         return jsonify({"error": "No image file found in the request"}), 400
#
#     file = request.files['image']
#     img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
#
#     # Cập nhật vị trí camera là trung tâm ảnh
#     height, width, _ = img.shape
#     camera_position = (width // 2, height)
#     cv2.circle(img, camera_position, radius=5, color=(0, 0, 255), thickness=-1)
#
#     def get_object_direction(obj_center):
#         """Xác định hướng của vật cản so với camera"""
#         dx = obj_center[0] - camera_position[0]
#         dy = obj_center[1] - camera_position[1]
#         if abs(dx) > abs(dy):
#             return "phía bên phải" if dx > 0 else "phía bên trái"
#         else:
#             return "rất gần" if dy > 0 else "phía trước"
#
#     results = model.predict(source=img, conf=0.25)[0]
#     boxes = results.boxes
#     class_names = model.names
#
#     class_translations = {
#         "Ground Animal": "Động vật trên mặt đất",
#         "Fence": "Hàng rào",
#         "Guard Rail": "Lan can bảo vệ",
#         "Barrier": "Rào chắn",
#         "Person": "Người",
#         "Bicyclist": "Người đi xe đạp",
#         "Motorcyclist": "Người đi xe máy",
#         "Other Rider": "Người điều khiển xe",
#         "Vegetation": "Cây cối",
#         "Bench": "Ghế dài",
#         "Fire Hydrant": "Vòi chữa cháy",
#         "Mailbox": "Hộp thư",
#         "Manhole": "Nắp cống",
#         "Pothole": "Ổ gà",
#         "Street Light": "Đèn đường",
#         "Pole": "Cột",
#         "Utility Pole": "Cột điện",
#         "Trash Can": "Thùng rác",
#         "Bicycle": "Xe đạp",
#         "Bus": "Xe buýt",
#         "Car": "Xe ô tô",
#         "Motorcycle": "Xe máy",
#         "Other Vehicle": "Phương tiện khác",
#         "Truck": "Xe tải"
#     }
#
#     closest_object = None
#     min_distance = float('inf')
#
#     for box in boxes:
#         cls_id = int(box.cls)
#         conf = float(box.conf)
#         label_en = class_names[cls_id]
#         if label_en not in class_translations:
#             continue
#
#         label_vi = class_translations.get(label_en, label_en)
#
#         x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
#         obj_center = ((x1 + x2) // 2, (y1 + y2) // 2)
#
#         # Vẽ bounding box và nhãn
#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(img, f"{label_vi} ({conf:.2f})", (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
#         # Tính khoảng cách Euclid từ camera đến vật thể
#         distance = math.hypot(obj_center[0] - camera_position[0], obj_center[1] - camera_position[1])
#
#         if distance < min_distance:
#             min_distance = distance
#             closest_object = {
#                 "label": label_vi,
#                 "confidence": conf,
#                 "distance": round(distance, 2),
#                 "direction": get_object_direction(obj_center)
#             }
#
#     if closest_object is None:
#         return jsonify({"error": "No object detected"}), 400
#
#     _, img_buffer = cv2.imencode('.jpg', img)
#     img_base64 = base64.b64encode(img_buffer).decode('utf-8')
#
#     return jsonify({
#         "closest_object": closest_object,
#         "image": img_base64
#     })

@app.post("/detect/")
async def detect_objects(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        return JSONResponse(content={"error": "File is not an image."}, status_code=400)

    # Decode image
    img_bytes = await image.read()
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    height, width, _ = img.shape
    camera_position = (width // 2, height)
    cv2.circle(img, camera_position, radius=5, color=(0, 0, 255), thickness=-1)

    def get_object_direction(obj_center):
        dx = obj_center[0] - camera_position[0]
        dy = obj_center[1] - camera_position[1]
        if abs(dx) > abs(dy):
            return "phía bên phải" if dx > 0 else "phía bên trái"
        else:
            return "rất gần" if dy > 0 else "phía trước"

    results = model.predict(source=img, conf=0.25)[0]
    boxes = results.boxes
    class_names = model.names

    class_translations = {
        "Ground Animal": "Động vật trên mặt đất",
        "Fence": "Hàng rào",
        "Guard Rail": "Lan can bảo vệ",
        "Barrier": "Rào chắn",
        "Person": "Người",
        "Bicyclist": "Người đi xe đạp",
        "Motorcyclist": "Người đi xe máy",
        "Other Rider": "Người điều khiển xe",
        "Vegetation": "Cây cối",
        "Bench": "Ghế dài",
        "Fire Hydrant": "Vòi chữa cháy",
        "Mailbox": "Hộp thư",
        "Manhole": "Nắp cống",
        "Pothole": "Ổ gà",
        "Street Light": "Đèn đường",
        "Pole": "Cột",
        "Utility Pole": "Cột điện",
        "Trash Can": "Thùng rác",
        "Bicycle": "Xe đạp",
        "Bus": "Xe buýt",
        "Car": "Xe ô tô",
        "Motorcycle": "Xe máy",
        "Other Vehicle": "Phương tiện khác",
        "Truck": "Xe tải"
    }

    closest_object = None
    min_distance = float('inf')

    for box in boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        label_en = class_names[cls_id]
        if label_en not in class_translations:
            continue

        label_vi = class_translations[label_en]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        obj_center = ((x1 + x2) // 2, (y1 + y2) // 2)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{label_vi} ({conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        distance = math.hypot(obj_center[0] - camera_position[0], obj_center[1] - camera_position[1])

        if distance < min_distance:
            min_distance = distance
            closest_object = {
                "label": label_vi,
                "confidence": round(conf, 2),
                "distance": round(distance, 2),
                "direction": get_object_direction(obj_center)
            }

    if closest_object is None:
        return JSONResponse(content={"error": "No object detected."}, status_code=400)

    _, img_encoded = cv2.imencode('.jpg', img)
    scale = 0.01  # 1 cm/pixel
    meters = closest_object['distance'] * scale
    text = f"Có {closest_object['label']} {closest_object['direction']} cách bạn {meters} mét"
    return {
        "data": text,
    }
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
