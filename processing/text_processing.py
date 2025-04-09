from nltk.tokenize import word_tokenize
import re
import nltk

# Tải punkt_tab nếu chưa có (chỉ cần chạy 1 lần)
nltk.download('punkt_tab', quiet=True)
def caption_preprocessing(text, remove_digits=True):
    text = re.sub(r'[^\w\s]', '', text)  # Giữ chữ cái và số
    text = word_tokenize(text.lower())
    text = '<START> ' + ' '.join(text) + ' <END>'
    return text
