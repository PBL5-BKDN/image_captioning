import os

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def correct_and_translate(text):
    prompt = (
        "Tôi sẽ gửi cho bạn một đoạn văn bản tiếng Anh.\n"
        "Bạn hãy sửa ngữ pháp, chính tả và làm cho câu văn trôi chảy hơn.\n"
        "Sau đó, hãy dịch đoạn văn bản đã chỉnh sửa sang tiếng Việt.\n\n"
        f"Văn bản: {text}\n\n"
        "Kết quả (chỉ trả lại bản dịch tiếng Việt):"
    )

    response = client.chat.completions.create(model="gpt-4o",
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=0.7,
    max_tokens=500)

    return response.choices[0].message.content.strip()
