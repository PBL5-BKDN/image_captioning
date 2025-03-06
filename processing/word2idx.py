from collections import Counter

def build_word2idx(train_caption, min_freq=1):
    """
    Tạo word2idx từ tập train_caption
    :param train_caption: Danh sách các câu chú thích
    :param min_freq: Số lần xuất hiện tối thiểu để một từ được thêm vào từ điển
    :return: word2idx (dict)
    """
    # Đếm số lần xuất hiện của từng từ
    word_counter = Counter()
    for caption in train_caption:
        tokens = ViTokenizer.tokenize(caption).split()
        word_counter.update(tokens)

    # Lọc những từ có tần suất ≥ min_freq
    vocab = [word for word, freq in word_counter.items() if freq >= min_freq]

    # Tạo word2idx
    word2idx = {word: idx for idx, word in enumerate(vocab, start=1)}

    # Thêm token đặc biệt
    word2idx["<PAD>"] = 0  # Padding
    word2idx["<UNK>"] = len(word2idx)  # Từ không có trong từ điển

    return word2idx

# Ví dụ tập caption
train_caption = ata

# Xây dựng word2idx
word2idx = build_word2idx(train_caption, min_freq=1)

# Kiểm tra
print("Số lượng từ vựng:", len(word2idx))
print("Từ 'thuyền' có ID:", word2idx.get("thuyền", word2idx["<UNK>"]))
print("Từ không có sẽ nhận ID:", word2idx["<UNK>"])

def encode_caption(caption, word2idx):

    return [word2idx.get(token.lower(), word2idx["<UNK>"]) for token in caption]

caption = "Ba chiếc thuyền đang di chuyển trên sông"
encoded_caption = encode_caption(caption, word2idx)
print("Caption mã hóa:", encoded_caption)