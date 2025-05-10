from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pineline.extract import extract_text_from_image
from pineline.llm_translate import correct_and_translate
import uvicorn

app = FastAPI()

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()

    # Step 1: Trích xuất text từ ảnh
    extracted_text = extract_text_from_image(contents)
    if not extracted_text.strip():
        return JSONResponse(content={"error": "Không thể trích xuất văn bản từ ảnh."}, status_code=400)

    # Step 2: Gửi sang LLM để sửa và dịch
    vietnamese_text = correct_and_translate(extracted_text)

    return {"original_text": extracted_text, "vietnamese_text": vietnamese_text}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
