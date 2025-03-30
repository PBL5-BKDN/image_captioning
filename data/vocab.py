import pickle

class Vocab:
    def __init__(self, path, word_count_threshold = 2):
        with open(path, 'rb') as f:
            self.data = pickle.load(f)


        self.word_counts = {}
        self.nsents = 0

        for item in self.data:
            caption = item['segment_caption']
            self.nsents += 1
            for w in caption.split(' '):
                self.word_counts[w] = self.word_counts.get(w, 0) + 1
        self.special_tokens = ["<PAD>", "<UNK>","<START>", "<END>"]
        self.vocab = self.special_tokens + [w for w in self.word_counts if self.word_counts[w] >= word_count_threshold and w != "<START>" and w != "<END>"]
        self.vocab_size = len(self.vocab)


        self.w2i = {w:i for i, w in enumerate(self.vocab)}

        self.i2w = {i: w for i, w in enumerate(self.vocab)}


        self.max_length_caption = max([len(item['segment_caption'].split(' ')) for item in self.data])


