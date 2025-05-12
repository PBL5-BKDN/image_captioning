from fastapi import FastAPI,File, UploadFile, Request, JSONResponse, Form
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os

from app.services.extract import extract_text_from_image
from app.services.llm_translate import polish_and_translate
from app.services.model_loader import get_model
from app.services.pipeline import pipeline

app = FastAPI(title="Simple FastAPI Server")

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
        return JSONResponse(
            status_code=400,
            content={"message": "File must be an image"}
        )

    # Đọc nội dung bytes từ UploadFile
    image_bytes = await file.read()

    # Gọi hàm xử lý (giả định bạn có pipeline được định nghĩa sẵn)
    output = pipeline(image_bytes, question, longitude, latitude)

    return {"data": output}





if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)