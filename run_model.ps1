conda activate image-captioning
uvicorn app.start_model:app --host 0.0.0.0 --port 4000