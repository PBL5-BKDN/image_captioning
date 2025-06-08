import os.path

import numpy as np
import requests


from app.services.llm import polish_and_translate, analyze_intent, answer_question_about_time_and_weather, \
    answer_question_basic, Intent, answer_question_out_of_ability, answer_question_street_status

import cv2

from settings import BASE_DIR
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import os

def draw_big_text(image_cv2, text, font_size=48, position=(50, 50), text_color=(255, 0, 0)):
    # Convert OpenCV BGR image to PIL RGB
    image_pil = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)

    # Load default font (suitable for English)
    font = ImageFont.truetype("arial.ttf", font_size)  # Đảm bảo máy bạn có arial.ttf

    # Draw text
    draw.text(position, text, font=font, fill=text_color)

    # Convert back to OpenCV BGR
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)


async def pipeline(image, question):
    """
    Pipeline function to process the image and question.
    """

    # Analyze intent of the question
    intent = analyze_intent(question)

    # Generate response based on intent
    match intent:
        case Intent.IMAGE_DESCRIPTION:
            # Extract text from image

            # Đọc nội dung bytes từ UploadFile
            image_bytes = await image.read()
            res = requests.post("http://localhost:4000/upload-image/", files={"file": (image.filename, image_bytes, image.content_type)}).json()
            # Polish and translate the extracted text
            print(res)
            polished_text = polish_and_translate(res["data"])

            import time
            start_time = time.time()
            # Chuyển ảnh từ bytes sang OpenCV
            img_array = np.frombuffer(image_bytes, np.uint8)
            img_cv2 = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # Ghi chữ lên ảnh
            image_with_text = draw_big_text(
                image_cv2=img_cv2,
                text=res["data"],  # đoạn tiếng Anh bạn muốn ghi
                font_size=80,  # thay đổi kích thước ở đây (ví dụ: 60, 80)
                position=(30, 30),  # toạ độ chữ
                text_color=(0, 0, 255)
            )

# Lưu ảnh có chữ
            save_path = os.path.join(BASE_DIR, "app", "image_caption_output", f"{time.time()}.jpg")
            cv2.imwrite(save_path, image_with_text)
            print(f"⏱ Thời gian ghi chữ lên ảnh: {time.time() - start_time:.2f} giây")
            response = polished_text

        case Intent.TIME_WEATHER:
            response = answer_question_about_time_and_weather(question)
        case Intent.INFORMATION_QUESTION:
            response = answer_question_basic(question)
        case Intent.OTHER_ACTION_REQUEST:
            response = answer_question_out_of_ability(question)
        case Intent.STREET_STATUS:
            response = answer_question_street_status(question)
        case _:
            response = answer_question_out_of_ability(question)

    return response
