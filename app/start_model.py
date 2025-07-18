import math
import os.path
from datetime import datetime
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO

from app.segment import predict_image
from app.services.extract import extract_text_from_image
from settings import BASE_DIR

app = FastAPI(title="Simple FastAPI Server")

model_path = os.path.join(BASE_DIR, "app", "model", "yolo_model.pt")
print(model_path)
model = YOLO(model_path)


@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI Server!"}



@app.post("/upload-image/")
async def upload_image(
        file: UploadFile = File(...),
):
    # Check if file is an image
    if not file.content_type.startswith("image/"):
        return {
            "error": "File is not an image."
        }

    # Đọc nội dung bytes từ UploadFile
    image_bytes = await file.read()

    text = extract_text_from_image(image_bytes)


    return {"data": text}

@app.post("/segment/")
async def segment(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        return JSONResponse(content={"error": "File is not an image."}, status_code=400)
    # Decode image
    img_bytes = await image.read()
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    return {
        "data": predict_image(img)
    }

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
        "Barrier": "Rào chắn",
        "Bicycle": "Xe đạp",
        "Bus": "Xe buýt",
        "Car": "Xe ô tô",
        "Motorcycle": "Xe máy",
        "Person": "Người",
        "Pole": "Cột",
        "Signboard": "Biển hiệu",        
        "Trash Can": "Thùng rác",
        "Truck": "Xe tải",
        "Vegetation": "Cây cối"
    }
    class_translations_en = {k: k for k, v in class_translations.items() if v is not None}

    print(class_translations)

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
        cv2.putText(img, f"{label_en} ({conf:.2f})", (x1, y1 - 10),
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
        # luu anh
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(BASE_DIR, "uploads", f'detect_image_{timestamp}.jpg')
    cv2.imwrite(filename, img)
    _, img_encoded = cv2.imencode('.jpg', img)

    if closest_object is None:
        return JSONResponse(content={"data": "Không nhận diện được vật cản nào"}, status_code=200)

    text = f"Có {closest_object['label']} {closest_object['direction']}"
    return {
        "data": text,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000)
