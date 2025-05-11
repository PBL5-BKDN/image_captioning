import requests

def call_openrouter(prompt, model="meta-llama/llama-3.1-8b-instruct:free"):
    headers = {
        "Authorization": "Bearer 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",
        "HTTP-Referer": "PBL5",  # Thay tên dự án nếu cần
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)

    try:
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print("❌ Error:", e)
        print("Response:", response.text)
        return "Lỗi khi gọi API."

# ===== 1. English sentence => Summarize + Polish => Translate to Vietnamese =====
def polish_and_translate(sentence_en):
    prompt = (
        "You are a helpful assistant that processes English sentences as follows:\n"
        "Step 1: Remove any redundant or repeated content to summarize the sentence.\n"
        "Step 2: Correct grammar, spelling, and improve fluency to make the sentence natural.\n"
        "Step 3: Translate the cleaned and corrected sentence into Vietnamese.\n\n"
        "⚠️ IMPORTANT: Only return the final Vietnamese translation. Do NOT explain anything. Do NOT include English text.\n\n"
        "Example:\n"
        "Input: a man is sitting on a table. he is wearing a black shirt. he is wearing a black shirt and black pants. the man is wearing a black shirt.\n"
        "Output: Một người đàn ông đang ngồi trên bàn, mặc áo sơ mi đen và quần đen.\n\n"
        f"Input:\n{sentence_en}\n"
        f"Output:"
    )
    return call_openrouter(prompt)

# ===== 2. Answer general question in Vietnamese =====
def answer_question(question_vi):
    prompt = (
        f"You are a helpful assistant. Please answer the following question in Vietnamese:\n"
        f"{question_vi}"
    )
    return call_openrouter(prompt)
