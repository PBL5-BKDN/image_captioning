import pickle
from pyvi import ViTokenizer
from torchtext.vocab import build_vocab_from_iterator


def tokenize_vi(text):
    return ViTokenizer.tokenize(text).split()


class Vocab:
    def __init__(self, path):
        print("Đang tải dữ liệu...")
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
        print("Đã tải dữ liệu!")

        print("Đang lấy captions...")
        captions = [item['segment_caption'] for item in self.data]
        print("Đã lấy captions:", captions[:2])  # In thử 2 caption đầu

        print("Đang xây dựng vocab...")
        self.vocab = build_vocab_from_iterator(map(tokenize_vi, captions),
                                               specials=["<UNK>", "<PAD>", "<START>", "<END>"])
        print("Đã xây dựng vocab!")

        self.vocab.set_default_index(self.vocab["<UNK>"])


# Thử chạy
vocab = Vocab("path/to/your/data.pkl")