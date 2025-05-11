from fastapi import FastAPI,File, UploadFile, Request
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os

from app.services.extract import extract_text_from_image
from app.services.llm_translate import polish_and_translate
from app.services.model_loader import get_model

app = FastAPI(title="Simple FastAPI Server")

@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI Server!"}

# Upload image endpoint
@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    # Check if file is an image
    if not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=400,
            content={"message": "File must be an image"}
        )
    # Đọc nội dung bytes từ UploadFile
    image_bytes = await file.read()

    model = get_model()
    englist_text = extract_text_from_image(image_bytes, model)
    vietnamese_text = polish_and_translate(englist_text)
    return {
        "message": "Image processed successfully",
        "original_text": englist_text,
        "translated_text": vietnamese_text
    }

@app.post("/gps")
async def receive_gps(request: Request):
    print("📍 Nhận dữ liệu GPS")
    data = await request.json()
    print(f"📍 Nhận dữQ liệu GPS: {data}")
    return {"message": "Đã nhận tọa độ!"}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)