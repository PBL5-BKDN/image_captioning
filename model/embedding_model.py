import fasttext
import numpy as np
import torch

# Load pre-trained FastText model cho tiếng Việt
ft_model = fasttext.load_model('cc.vi.300.bin')  # File pre-trained FastText tiếng Việt

# Giả sử bạn có từ điển word2idx từ dataset
# Tạo embedding matrix
vocab_size = len(train_dataset.word2idx)  # Số từ trong từ điển
embed_size = 300  # Kích thước vector của FastText
embedding_matrix = np.zeros((vocab_size, embed_size))

# Điền vector embedding cho từng từ
for word, idx in train_dataset.word2idx.items():
    if word in ft_model:
        embedding_matrix[idx] = ft_model[word]
    else:
        embedding_matrix[idx] = np.random.normal(size=(embed_size,))  # Random cho từ không có

# Chuyển thành tensor
embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)