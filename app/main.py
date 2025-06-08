

import cv2

import requests
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


from app.services.pipeline import pipeline
from settings import BASE_DIR

app = FastAPI(title="Simple FastAPI Server")


@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI Server!"}


# Upload image endpoint
@app.post("/upload-image/")
async def upload_image(
        file: UploadFile = File(...),
        question: str = Form(...),
):
    # Check if file is an image
    if not file.content_type.startswith("image/"):
        return {
            "error": "File is not an image."
        }

    # Gọi hàm xử lý (giả định bạn có pipeline được định nghĩa sẵn)
    output = await pipeline(file, question)

    return {"data": output}


@app.post("/detect/")
async def detect_objects(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        return JSONResponse(content={"error": "File is not an image."}, status_code=400)
    # Decode image
    img_bytes = await image.read()

    text = requests.post(
        "http://localhost:4000/detect/",
        files={"image": (image.filename, img_bytes, image.content_type)}
    ).json()

    return {
        "data": text,
    }

@app.post("/segment/")
async def detect_objects(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        return JSONResponse(content={"error": "File is not an image."}, status_code=400)
    # Decode image
    img_bytes = await image.read()

    res = requests.post(
        "http://localhost:4000/segment/",
        files={"image": (image.filename, img_bytes, image.content_type)}
    ).json()

    return res

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
