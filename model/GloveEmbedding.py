import numpy as np
import torch


def load_glove_embeddings(glove_path, word2idx, embedding_dim=100):
    # Tạo dictionary chứa GloVe embeddings
    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector

    # Khởi tạo embedding matrix
    vocab_size = len(word2idx)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    # Điền vector GloVe vào matrix
    for word, idx in word2idx.items():
        if word in embeddings_index:
            embedding_matrix[idx] = embeddings_index[word]
        else:
            # Nếu từ không có trong GloVe, giữ ngẫu nhiên hoặc zero
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))

    return torch.tensor(embedding_matrix, dtype=torch.float32)



