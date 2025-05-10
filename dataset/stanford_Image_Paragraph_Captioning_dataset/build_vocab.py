import os
import pandas as pd
from settings import BASE_DIR, WORD_COUNT_THRESHOLD
from collections import defaultdict
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "dataset/stanford_Image_Paragraph_Captioning_dataset", "train.csv")

class Vocab:
    def __init__(self, path: str):
        train_df = pd.read_csv(path)

        word_counts = defaultdict(int)  # a dict : { word : number of appearances}
        self.MAX_LENGTH = 0

        for text in train_df['Paragraph']:
            words = text.split()
            self.MAX_LENGTH = len(words) if (self.MAX_LENGTH < len(words)) else self.MAX_LENGTH
            for w in words:
                word_counts[w] += 1

        self.vocab = ["<PAD>", "<UNK>", "<START>", "<END>"] + [w for w, count in word_counts.items() if
                                                          count >= WORD_COUNT_THRESHOLD]
        self.vocab_size = len(self.vocab)
        self.w2i = {}
        self.i2w = {}
        for i in range(len(self.vocab)):
            self.w2i[self.vocab[i]] = i
            self.i2w[i] = self.vocab[i]

vocab = Vocab(TRAIN_DATA_PATH)


