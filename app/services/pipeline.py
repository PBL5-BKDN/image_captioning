import requests

from app.services.llm import polish_and_translate, analyze_intent, answer_question_about_time_and_weather, \
    answer_question_basic, Intent, answer_question_out_of_ability, answer_question_street_status


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
