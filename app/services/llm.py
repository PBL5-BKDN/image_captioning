import random
import re
from enum import Enum

import requests

from app.services.services import get_temperature_and_weather, search_web, get_traffic_data
from settings import WEATHER_API_KEY, LLM_API_KEY, SERP_API_KEY

print(WEATHER_API_KEY)


def call_openrouter(user_prompt, sys_prompt, model="qwen3-1.7b", temperature=0.1, max_tokens=2000):
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    response = requests.post("http://127.0.0.1:1234/v1/chat/completions", headers=headers, json=data)

    try:
        res = response.json()["choices"][0]["message"]["content"]
        print("Response:", res)
        return re.sub(r"<think>.*?</think>", "", res, flags=re.DOTALL).strip().replace("\n", " ").replace("*", " ")
    except Exception as e:
        print("❌ Error:", e)
        print("Response:", response.text)
        return "Lỗi khi gọi API."


# ===== 1. English sentence => Summarize + Polish => Translate to Vietnamese =====
def polish_and_translate(sentence_en):
    user_prompt = f"""Input: {sentence_en} """
    sys_prompt = f"""Bạn là một trợ lý thông minh và thân thiện, có nhiệm vụ xử lý câu tiếng Anh như sau:
    Bước 1: Hiểu rõ nghĩa và ngữ cảnh của câu tiếng Anh.
    Bước 2: Viết lại một câu tiếng Việt tự nhiên, giữ nguyên ý nghĩa ban đầu.
    
    LƯU Ý: Chỉ trả về lời chào, lời chúc, và câu tiếng Việt tương đương và chú ý giữ cách xưng hô cho đúng từ đầu đến cuối. KHÔNG giải thích. KHÔNG bao gồm lại câu tiếng Anh.
    
    Format đầu ra:
    - Trực tiếp trả lời câu tiếng Việt tương đương với câu tiếng Anh đã cho bắt đầu bằng: "Phía trước bạn là ".
    
    Ví dụ:
    Input: A woman is cooking dinner in the kitchen.
    Output:
    Phía trước bạn là một người phụ nữ đang nấu bữa tối trong bếp.
    """

    return call_openrouter(user_prompt, sys_prompt)


def answer_question_basic(question_vi):
    context = search_web(question_vi, top_k=5)
    print("Context found:", context)
    user_prompt = f"Câu hỏi: {question_vi}. Thông tin tìm được: {context}"
    sys_prompt = f"""Bạn là trợ lý giọng nói tiếng Việt thân thiện dành cho người khiếm thị.
    Khi có câu hỏi, hãy làm như sau:

    1. Đọc kỹ câu hỏi và tìm thông tin liên quan trong **5 đoạn thông tin** được cung cấp bên dưới.
    2. Mỗi đoạn có định dạng như sau:
    "(**Số thứ tự**) Tiêu đề thông tin: **tiêu đề**. Nội dung thông tin: **nội dung**. Ngày đăng: **ngày đăng**".
    Dưới đây là 5 đoạn thông tin được tìm thấy từ internet. Một số thông tin có thể không chính xác hoặc không liên quan.

    Hướng dẫn xử lý:
    - Chỉ sử dụng tiếng Việt. Không dùng tiếng Anh, tiếng Trung hoặc từ viết tắt.
     - Hãy **tổng hợp có chọn lọc** các thông tin liên quan để tạo ra câu trả lời ngắn gọn, đúng trọng tâm câu hỏi. Nếu nhiều đoạn cùng liên quan, hãy chọn lọc và tổng hợp chúng một cách mạch lạc, tự nhiên.
    - Chọn lọc thông tin đúng nội dung câu hỏi, dựa theo **tiêu đề** và **nội dung**.
    - Không chỉ chọn một đoạn thông tin duy nhất. Có thể dùng nhiều đoạn để tổng hợp nếu cần. Nếu nhiều đoạn cùng liên quan, hãy chọn lọc và tổng hợp chúng một cách mạch lạc, tự nhiên.
        - Nếu có mâu thuẫn giữa các thông tin, hãy **ưu tiên thông tin có ngày đăng mới nhất**.
        - Chỉ sử dụng thông tin thực sự liên quan đến câu hỏi, dựa trên **tiêu đề** hoặc **nội dung**. Bỏ qua những đoạn thông tin không liên quan đến câu hỏi (dựa vào tiêu đề hoặc nội dung).
    - Nếu có nhiều đoạn liên quan, hãy tổng hợp ngắn gọn và tự nhiên.
    - Nếu thông tin mâu thuẫn, chọn đoạn có **ngày đăng mới hơn**.
    - Bỏ qua thông tin không liên quan đến câu hỏi.
    - Trả lời phải ngắn gọn, rõ ràng, dễ đọc to thành tiếng.
    - Câu văn đơn giản, không dùng từ chuyên môn hoặc cấu trúc phức tạp."""
    return call_openrouter(user_prompt, sys_prompt, model="qwen/qwen3-8b")

def answer_question_about_time_and_weather(question):
    data = get_temperature_and_weather()
    user_prompt = f"""
        Câu hỏi: {question}
        dữ liệu thời tiết JSON: {data}
    """
    sys_prompt = f"""
        Bạn là một trợ lý giọng nói thân thiện, tận tâm hỗ trợ người khiếm thị. Dựa trên dữ liệu thời tiết JSON dưới đây, hãy tạo một câu phản hồi ngắn gọn, dễ hiểu, tự nhiên và gần gũi để mô tả thời tiết hoặc thời gian hiện tại cho người dùng, đồng thời đưa ra một lời khuyên nhẹ nhàng về hoạt động phù hợp với điều kiện hôm nay.
        Hướng dẫn:
        - Xác định xem người dùng đang hỏi về **thời tiết** hay **thời gian**, và trả lời đúng theo yêu cầu.
        - Nếu câu hỏi liên quan đến thời tiết, hãy mô tả trạng thái thời tiết bao gồm nhiệt độ và mô tả như: "trời mát mẻ", "có mây nhẹ", "trời nắng đẹp", "gió nhẹ", v.v. nếu phù hợp. Sau phần mô tả, hãy đưa ra một lời khuyên nhẹ nhàng về hoạt động phù hợp với thời tiết.
        - Nếu câu hỏi liên quan đến thời gian, hãy trả lời bằng giờ địa phương với cách diễn đạt đơn giản, ví dụ: "Bây giờ là 3 giờ chiều".
        - Văn phong **hàm súc, cô đọng**, dễ hình dung bằng lời nói.
        - Không cần lặp lại nội dung JSON.
        - Không giải thích lại yêu cầu. 
        - Hãy trả lời bằng tiếng Việt 
        - Vì kết quả sử dụng để phát ra âm thanh nên hãy loại bỏ những kí tự đặc biệt như: *, #, @, $, %, ^, &, (, ), _, +, [, ], |, \, :, ;, ", ', <, >, ?, /, ~, `, "\n" 
        Phản hồi mẫu cần ngắn gọn và trọng tâm.
        Ví dụ minh họa:
        Câu hỏi: "Hôm nay thời tiết thế nào?"
        Dữ liệu thời tiết JSON: {{ "temperature": 29, "description": "Nắng nhẹ", "wind": "gió nhẹ" }}
    
        Phản hồi mẫu:
        "Hôm nay trời nắng nhẹ, nhiệt độ khoảng 29 độ C, có gió nhẹ nên rất dễ chịu. Thời tiết rất thích hợp để đi dạo hoặc tập thể dục nhẹ ngoài trời."
    
        Câu hỏi: "Mấy giờ rồi?"
        Phản hồi mẫu:
        "hiện tại là 2 giờ 15 phút chiều."
    
        Bây giờ hãy tạo câu trả lời phù hợp dựa trên dữ liệu thời tiết và câu hỏi người dùng.
        """
    print(sys_prompt)
    response = call_openrouter(user_prompt, sys_prompt, model="qwen3-1.7b")
    return response


class Intent(Enum):
    IMAGE_DESCRIPTION = "Mô tả ảnh"
    TIME_WEATHER = "Hỏi về ngày, giờ hoặc thời tiết"
    INFORMATION_QUESTION = "Câu hỏi thông tin"
    OTHER_ACTION_REQUEST = "Yêu cầu hành động khác"
    STREET_STATUS = "Hỏi về tình trạng đường phố"


def analyze_intent(question: str):
    sys_prompt = f"""
    Bạn là một hệ thống phân loại đầu vào cho trợ lý giọng nói hỗ trợ người khiếm thị. Người dùng có thể nói một câu bất kỳ: đó có thể là yêu cầu hoặc câu hỏi.

    Dựa vào câu đầu vào, hãy xác định **ý định chính** của người dùng và phân loại nó vào **một trong 4 nhóm sau**:

    Mô tả ảnh
       - Khi người dùng muốn biết nội dung trong ảnh, hình ảnh đang hiển thị, mô tả những gì trước mặt, hoặc nói các cụm như: "mô tả phía trước", "đây là gì", "có gì phía trước", "đây có phải là...", "mô tả ảnh", "xung quanh có gì"...

    Hỏi về ngày, giờ hoặc thời tiết
       - Khi người dùng hỏi hoặc yêu cầu thông tin về thời gian (giờ, ngày, lịch), hoặc thời tiết (nhiệt độ, trời mưa không, thời tiết hôm nay thế nào,...)  
       - Bao gồm các câu như: "Bây giờ là mấy giờ?", "Hôm nay là thứ mấy?", "Ngoài trời có mưa không?", "Thời tiết hôm nay thế nào?"

    Hỏi về tình trạng đường phố
        - Khi người dùng hỏi về tình trạng đường phố, tình trạng giao thông, hoặc các sự cố trên đường phố.  
        - Bao gồm các câu như: "Đường phố Hà Nội có đông không?", "Có sự cố gì trên đường Điện Biên Phủ Đà Nẵng không?", "Tình trạng giao thông tại cầu Rồng hiện tại thế nào?", "đường Nguyễn Phúc có kẹt xe không?"
        
    Câu hỏi thông tin
       - Khi người dùng hỏi về kiến thức cụ thể hoặc yêu cầu thông tin về một đối tượng/sự kiện nào đó (không liên quan đến ảnh, thời gian, thời tiết)  
       - Ví dụ: "AI là gì?", "Tổng thống Mỹ là ai?", "Hà Nội có bao nhiêu quận?", "Cho biết thông tin về con voi"
 
    Yêu cầu hành động khác
       - Mọi yêu cầu còn lại không thuộc 3 nhóm trên, như: yêu cầu thực hiện hành động, điều khiển thiết bị, gửi lời nhắc, nhắn tin, mở đèn, báo thức, v.v.  
       - Ví dụ: "Mở đèn", "Nhắc tôi uống thuốc", "Gửi tin nhắn", "Đọc email"
    
    **Chỉ trả về kết quả phân loại. Không trả lời thêm bất kỳ nội dung nào khác. Không lặp lại câu hỏi người dùng. Không giải thích.**
    """

    user_prompt = f"""Câu hỏi/ yêu cầu của người dùng: "{question}" """
    response = call_openrouter(user_prompt, sys_prompt, model="qwen3-1.7b")
    try:
        print("Phân loại ý định:", response)
        return Intent(response.strip())
    except ValueError:
        print(f"❌ Error: Không thể phân loại ý định cho câu hỏi '{question}'. Phản hồi từ mô hình: {response}")
        return Intent.OTHER_ACTION_REQUEST


def answer_question_out_of_ability(question: str):
    user_prompt = f"""Câu hỏi/ yêu cầu của người dùng: "{question}" """
    sys_prompt = (
        """Bạn là một trợ lý giọng nói dành cho người khiếm thị. Trợ lý chỉ có khả năng trả lời các câu hỏi thông tin, mô tả ảnh, hoặc cung cấp thông tin về thời tiết, ngày, giờ.Nếu người dùng yêu cầu thực hiện hành động khác (ví dụ: gửi tin nhắn, bật đèn, nhắc nhở), hãy trả lời một cách lịch sự và thân thiện rằng bạn không thể thực hiện yêu cầu đó.
        Yêu cầu đầu ra:
        - Một câu trả lời tiếng Việt ngắn gọn, lịch sự, giải thích rằng bạn không thể thực hiện yêu cầu và nói về khả năng của mình."
        - Ví dụ: 'Xin lỗi, mình chỉ có thể trả lời câu hỏi hoặc mô tả ảnh, không thể gửi tin nhắn được.  Tôi chỉ có thể hỗ trợ bạn về mô tả ảnh và trả lời câu hỏi.'
    """
    )
    response = call_openrouter(user_prompt, sys_prompt)
    return response

def answer_question_street_status(question: str):
    user_prompt = f"""Câu hỏi/ yêu cầu của người dùng: "{question}" """
    sys_prompt = """
        Bạn là một trợ lý AI chuyên phân tích các câu hỏi liên quan đến tình trạng đường phố và giao thông.
    
        Nhiệm vụ của bạn là:
        1. Kiểm tra xem trong câu hỏi có địa chỉ cụ thể hay không.
        2. Địa chỉ được xem là **đủ thông tin** nếu chứa ít nhất:
        
           - Tên đường hoặc tên địa danh (bắt buộc). Ví dụ: đường Nguyễn Hữu Thọ ( đường ), cầu Rồng ( địa danh ), ngã Ba Huế ( địa danh ), ... 
           - Tên thành phố (bắt buộc, nếu không mặc định là Đà Nẵng)
           - Có thể có hoặc không có số nhà
           - Quốc gia (nếu không mặc định là Việt Nam)
    
        3. Nếu có đủ thông tin thì trả về địa chỉ dưới định dạng chuẩn hóa như:
           "Số + Tên đường, Thành phố, Quốc gia"
    
        4. Nếu không đủ thông tin thì chỉ rõ thiếu phần gì.
        5. Trả về Python dict với các giá trị boolean đúng chuẩn:
            - is_enough phải là `True` nếu địa chỉ đủ thông tin, `False` nếu không đủ (phải viết hoa đầu và không dùng dạng chuỗi hoặc viết thường).
        

        Trả về Python dict:
        {
            "is_enough": Giá trị [True,False] ( Chú ý : Viết hoa đầu tiên ),
            "address": "Địa chỉ chuẩn hóa nếu đủ, hoặc rỗng nếu không đủ",
            "missing_part": "Mô tả phần còn thiếu (nếu có)"
        }
        ** Chú ý:  Bắt buộc trả về đúng định dạng Python dictionary như trên, nếu không thì dẫn đến sai sót trong quá trình xử lý. Hãy xem ví dụ minh họa ở dưới để trả về đúng định dạng **
         Ví dụ định dạng trả về:
        {
            "is_enough": True,
            "address": "123 Nguyễn Trãi, Quận 5, TP.HCM, Việt Nam",
            "missing_part": ""
        }
    
        hoặc
    
        {
            "is_enough": False,
            "address": "",
            "missing_part": "Thiếu tên thành phố"
        }

        
        """

    response = call_openrouter(user_prompt, sys_prompt)
    res = eval(response)
    if not res["is_enough"]:
        return "Địa chỉ không đủ thông tin. Vui lòng cung cấp đủ thông tin địa chỉ bao gồm số (có thể có hoặc không) + tên đường (bắt buộc) + tên thành phố hoặc tên địa điểm (bắt buộc) + tên thành phố (bắt buộc)."

    street_status = get_traffic_data(res["address"])

    user_prompt = f"""thông tin giao thông JSON: "{street_status}" """
    sys_prompt = """
        Bạn là một trợ lý AI chuyên tóm tắt tình trạng giao thông theo cách dễ hiểu cho người dùng, đặc biệt là người khiếm thị.
    
        Dựa trên đầu vào là thông tin giao thông JSON gồm:
        - Tên vị trí
        - Loại đường
        - Tốc độ hiện tại vs tốc độ lý tưởng
        - Mức độ tắc nghẽn
        - Danh sách các sự cố (nếu có
        
        Bạn cần:
        1. Mô tả ngắn gọn tình trạng hiện tại của đường (ví dụ: "đang kẹt xe nhẹ", "giao thông thông thoáng", ...)
        2. Nếu có sự cố (tai nạn, sửa đường, tắc đường...) thì liệt kê chúng.
        3. Đưa ra lời khuyên cho người đi bộ hoặc người khiếm thị (ví dụ: "nên tránh khu vực này", "có thể di chuyển an toàn", ...)
        4. Văn phong lịch sự, dễ hiểu.
        
        Trả về văn bản tiếng Việt, không cần định dạng JSON.
        """
    response = call_openrouter(user_prompt, sys_prompt)
    return response
