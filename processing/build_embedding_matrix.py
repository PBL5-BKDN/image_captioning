import torch


def build_embedding_matrix(word2idx, ft_model, embed_dim=300):
    # Khởi tạo matrix với các vector ngẫu nhiên
    embedding_matrix = torch.randn(len(word2idx), embed_dim) * 0.01

    for word, idx in word2idx.items():
        if word in ft_model.words:
            embedding_matrix[idx] = torch.tensor(ft_model.get_word_vector(word))
        elif word.lower() in ft_model.words: # Thử lowercase
            embedding_matrix[idx] = torch.tensor(ft_model.get_word_vector(word.lower()))

    # Đặc biệt xử lý các special tokens
    embedding_matrix[word2idx["<PAD>"]] = torch.zeros(embed_dim) # Zero vector cho padding
    return embedding_matrix