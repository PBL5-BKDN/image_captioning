import fasttext.util

fasttext.util.download_model("vi", if_exists="ignore")  # Tải mô hình tiếng Việt
ft = fasttext.load_model("../cc.vi.300.bin")  # Load mô hình
def get_word_embedding(word):
    return ft.get_word_vector(word)  # Vector có kích thước 300

print(get_word_embedding("mèo").shape)  # (300,)