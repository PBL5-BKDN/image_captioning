from app.services.extract import extract_text_from_image
from app.services.llm_translate import polish_and_translate, analyze_intent, answer_question_about_time_and_weather, \
    answer_question_basic, Intent, answer_question_out_of_ability
from app.services.model_loader import get_model


def pipeline(image, question, longitude, latitude):
    """
    Pipeline function to process the image and question.
    """

    # Analyze intent of the question
    intent = analyze_intent(question)

    # Generate response based on intent
    match intent:
        case Intent.IMAGE_DESCRIPTION:
            # Extract text from image
            model = get_model()
            extracted_text = extract_text_from_image(image, model)
            # Polish and translate the extracted text
            polished_text = polish_and_translate(extracted_text)
            response = polished_text

        case Intent.TIME_WEATHER:
            response = answer_question_about_time_and_weather(longitude, latitude)
        case Intent.INFORMATION_QUESTION:
            response = answer_question_basic(question)
        case Intent.OTHER_ACTION_REQUEST:
            response = answer_question_out_of_ability(question)
        case _:
            response = answer_question_out_of_ability(question)

    return response