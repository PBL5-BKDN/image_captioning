from enum import Enum

import requests

from settings import WEATHER_API_KEY, LLM_API_KEY
print(WEATHER_API_KEY)

def call_openrouter(prompt, model="qwen/qwen3-0.6b-04-28:free"):
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
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
        "You are a helpful assistant that processes English sentences as follows:"
        "Step 1: Remove any redundant or repeated content to summarize the sentence."
        "Step 2: Correct grammar, spelling, and improve fluency to make the sentence natural."
        "Step 3: Translate the cleaned and corrected sentence into Vietnamese."
        "⚠️ IMPORTANT: Only return the final Vietnamese translation. Do NOT explain anything. Do NOT include English text.\n\n"
        "Example:\n"
        "Input: a man is sitting on a table. he is wearing a black shirt. he is wearing a black shirt and black pants. the man is wearing a black shirt.\n"
        "Output: Một người đàn ông đang ngồi trên bàn, mặc áo sơ mi đen và quần đen."
        f"Input:\n{sentence_en}\n"
        f"Output:"
    )
    return call_openrouter(prompt)


def answer_question_basic(question_vi):
    prompt = (
        f"Bạn là một trợ lý giọng nói dành cho người khiếm thị. Hãy trả lời câu hỏi sau bằng tiếng Việt một cách ngắn gọn, rõ ràng và thân thiện:\n"
        f"{question_vi}"
    )
    return call_openrouter(prompt)


def answer_question_about_time_and_weather(question, longitude, latitude):
    response = requests.get(
        "https://api.weatherapi.com/v1/current.json",
        params={
            "key": WEATHER_API_KEY,
            "q": f"{latitude},{longitude}",
            "lang": "vi"
        }
    )
    data = response.json()
    data = {
        "location":{
            "name": data["location"]["name"],
            "region": data["location"]["region"],
            "country": data["location"]["country"],
            "localtime": data["location"]["localtime"]
        },
        "current": {
            "temp_c": data["current"]["temp_c"],
            "condition": {
                "text": data["current"]["condition"]["text"],
            },
        }
    }
    print(data)

    sys_prompt = f"""
        Bạn là một trợ lý giọng nói dành cho người khiếm thị. Dựa trên dữ liệu thời tiết JSON dưới đây, hãy tạo một câu phản hồi ngắn gọn, dễ hiểu và thân thiện để mô tả thời tiết hiện tại và thời gian cho người dùng. Hạn chế dùng từ chuyên môn, ưu tiên giọng nói tự nhiên.
        Dữ liệu JSON: {str(data)} 
        Cau hỏi/ yêu cầu của người dùng: "{question}"
        Yêu cầu đầu ra:
        - Dựa trên câu hỏi hãy phân biệt người dùng hỏi về thời tiết hay thời gian và trả lời đúng về thời tiết hoặc thời gian.
        - Thân thiện, rõ ràng, dễ hiểu với người khiếm thị.
        - Không nhắc đến đơn vị đo nếu không cần thiết.
        - Nếu có thể, mô tả trạng thái như "trời có mây", "gió nhẹ", "độ ẩm cao", v.v.
        - Không cần nhắc lại dữ liệu JSON.
        - Không giải thích gì thêm.
    """
    response = call_openrouter(sys_prompt)
    return response


class Intent(Enum):
    IMAGE_DESCRIPTION = "Mô tả ảnh"
    TIME_WEATHER = "Hỏi về ngày, giờ hoặc thời tiết"
    INFORMATION_QUESTION = "Các câu hỏi thông tin"
    OTHER_ACTION_REQUEST = "Yêu cầu hành động khác"


def analyze_intent(question: str):
    sys_prompt = f"""
        Bạn là một hệ thống phân loại đầu vào cho trợ lý giọng nói hỗ trợ người khiếm thị. Người dùng có thể nói một câu bất kỳ: đó có thể là yêu cầu hoặc câu hỏi.
        
        Dựa vào câu đầu vào, hãy xác định **ý định chính** của người dùng và phân loại nó vào **một trong 4 nhóm sau**:
        
        - Mô tả ảnh (Người dùng muốn biết nội dung trong ảnh, hình ảnh đang hiển thị, hoặc nhờ mô tả ảnh)
        - Hỏi về ngày, giờ hoặc thời tiết (Người dùng muốn biết giờ, ngày, lịch, hoặc tình hình thời tiết)
        - Các câu hỏi thông tin (Người dùng hỏi một thông tin cụ thể khác (ví dụ: "Tổng thống Mỹ là ai?", "Hà Nội có bao nhiêu quận?", "AI là gì?"...))
        - Yêu cầu hành động khác (Người dùng yêu cầu làm một điều gì đó không nằm trong 3 nhóm trên (ví dụ: "Nhắc tôi uống thuốc", "Mở đèn", "Gửi tin nhắn", "Đọc email"))
        
        **Yêu cầu:**
        - Trả lời bằng một dòng duy nhất 1 thể loại duy nhất: Mô tả ảnh; Hỏi về ngày, giờ hoặc thời tiết; Các câu hỏi thông tin; Yêu cầu hành động khác và không cần giải thích thêm.
        - Không lặp lại nguyên câu người dùng.
        - Nếu không chắc chắn giữa 2 loại, chọn loại gần nhất theo ngữ cảnh.
        
        Ví dụ:
        
        Câu: "Hôm nay trời thế nào?"
        → Hỏi về ngày, giờ hoặc thời tiết 
        
        Câu: "Bạn có thể giúp tôi gửi tin nhắn không?"
        → Yêu cầu hành động khác
        
        Câu: "Trong ảnh này có gì?"
        → Mô tả ảnh 
        
        Câu: "Tổng thống Mỹ là ai?"
        → Các câu hỏi thông tin
        
        Câu hỏi/ yêu cầu của người dùng: "{question}"
    """
    response = call_openrouter(sys_prompt)
    try:
        print(response)
        return Intent(response.strip())
    except ValueError:
        print(f"❌ Error: Không thể phân loại ý định cho câu hỏi '{question}'. Phản hồi từ mô hình: {response}")
        return Intent.OTHER_ACTION_REQUEST


def answer_question_out_of_ability(question: str):
    sys_prompt = (
        f"Bạn là một trợ lý giọng nói dành cho người khiếm thị. Trợ lý chỉ có khả năng trả lời các câu hỏi thông tin, mô tả ảnh, hoặc cung cấp thông tin về thời tiết, ngày, giờ. "
        f"Nếu người dùng yêu cầu thực hiện hành động khác (ví dụ: gửi tin nhắn, bật đèn, nhắc nhở), hãy trả lời một cách lịch sự và thân thiện rằng bạn không thể thực hiện yêu cầu đó.\n\n"
        f"Câu hỏi/ yêu cầu: '{question}'\n\n"
        "Yêu cầu đầu ra:\n"
        "- Một câu trả lời tiếng Việt ngắn gọn, lịch sự, giải thích rằng bạn không thể thực hiện yêu cầu và nói về khả năng của mình.\n"
        "- Ví dụ: 'Xin lỗi, mình chỉ có thể trả lời câu hỏi hoặc mô tả ảnh, không thể gửi tin nhắn được.  Tôi chỉ có thể hỗ trợ bạn về mô tả ảnh và trả lời câu hỏi.'"
    )
    response = call_openrouter(sys_prompt)
    return response
