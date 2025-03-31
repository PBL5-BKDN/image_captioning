import torch
import torch.nn as nn
from torchvision import models

# Load InceptionV3
inception = models.inception_v3(weights=True)
inception.fc = torch.nn.Identity()
inception.eval()

import torch.nn as nn

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_size=300, hidden_size=256, embedding_matrix=None):
        super(ImageCaptioningModel, self).__init__()

        # Embedding layer với pre-trained weights
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embedding.weight = nn.Parameter(embedding_matrix)
        self.embedding.weight.requires_grad = False  # Không cần train embedding

        # LSTM để dự đoán chuỗi token
        self.lstm = nn.LSTM(embed_size + 256, hidden_size, batch_first=True)

        # Linear layer để dự đoán từ tiếp theo
        self.linear = nn.Linear(hidden_size, vocab_size)

        # Xử lý ảnh (giả sử dùng Inception V3)
        self.inception = inception  # Đã định nghĩa trước
        self.linear_img = nn.Linear(2048, 256)  # Giảm chiều đặc trưng ảnh

    def forward(self, images, captions):
        # Xử lý ảnh
        with torch.no_grad():
            image_features = self.inception(images)
            image_features = image_features.logits  # Lấy đặc trưng từ Inception
        image_features = self.linear_img(image_features)  # Giảm chiều

        # Lấy embedding của caption (bỏ token cuối vì dự đoán từ tiếp theo)
        embedded_captions = self.embedding(captions[:, :-1])

        # Mở rộng image_features để khớp với chiều của captions
        seq_len = embedded_captions.shape[1]
        image_features = image_features.unsqueeze(1).repeat(1, seq_len, 1)

        # Kết hợp đặc trưng ảnh và embedding của caption
        lstm_input = torch.cat((image_features, embedded_captions), dim=2)

        # Đưa qua LSTM và dự đoán
        lstm_output, _ = self.lstm(lstm_input)
        outputs = self.linear(lstm_output)
        return outputs