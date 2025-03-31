import torch
import torch.nn as nn

from torchvision.models import Inception_V3_Weights
import torchvision.models as models
import fasttext.util




class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size=5000, image_feature_size=2048, hidden_size=256, vocab=None):
        self.vocab = vocab

        super(ImageCaptioningModel, self).__init__()
        model_inception_v3 = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
        model_inception_v3.fc = torch.nn.Identity()  # Giữ lại feature vector trước lớp fully connected
        model_inception_v3.eval()
        self.inception_v3 = model_inception_v3
        self.image_encoder = nn.Sequential(
            nn.Linear(image_feature_size, hidden_size),
            nn.Dropout(0.5),
            nn.ReLU()
        )

        self.special_token_embedding = nn.Embedding(len(vocab.special_tokens), 300)
        self.ft = fasttext.load_model("cc.vi.300.bin")
        self.caption_projection = nn.Linear(300, hidden_size)

        self.caption_lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True)

        self.fc = nn.Linear(hidden_size, vocab_size)

    def embed_caption(self, caption_tokens):
        batch_embeddings = []

        for batch in caption_tokens:
            sentence_embeddings = []
            for input in batch:
                word_embeddings = []
                for token_id in input:
                    token = self.vocab.i2w[token_id.item()]
                    if token in self.vocab.special_tokens:
                        emb = self.special_token_embedding(torch.tensor(
                            [self.vocab.w2i[token]],
                            device=self.special_token_embedding.weight.device)
                        ).squeeze(0)
                    else:
                        emb = torch.Tensor(
                            self.ft.get_word_vector(token)
                        ).to(self.special_token_embedding.weight.device)
                    word_embeddings.append(emb)

                # Convert word_embeddings list thành tensor (seq_len, 300)
                sentence_embeddings.append(torch.stack(word_embeddings, dim=0))

            # Convert sentence_embeddings list thành tensor (batch_size, seq_len, 300)
            batch_embeddings.append(torch.stack(sentence_embeddings, dim=0))

        # Convert batch_embeddings list thành tensor (batch_size, seq_len, 300)
        return torch.stack(batch_embeddings, dim=0)

    def forward(self, image, caption_tokens):
        """
       image: (batch_size, 3, 299, 299)
       caption_tokens: (batch_size,(num_seq) seq_len-1, seq_len)
       """
        # Embedding cho caption
        caption_embedding = self.embed_caption(caption_tokens) #(batch_size, num_sequences, seq_len, 300)
        batch_size, num_sequences, seq_len, token_dim = caption_embedding.shape
        caption_embedding = caption_embedding.view(batch_size * num_sequences, seq_len, token_dim)

        caption_embedding = self.caption_projection(caption_embedding)  # (batch_size * num_sequences, seq_len, hidden_size)

        self.inception_v3.eval()
        with torch.no_grad():
            features = self.inception_v3(image)
            if isinstance(features, tuple):
                features = features[0]  # Chỉ lấy feature chính nếu output là tuple (batch_size, 2048)

        # Encode ảnh
        image_embedding = self.image_encoder(features)  # (batch_size, hidden_size)

        image_embedding = image_embedding.unsqueeze(1)  # (batch_size, 1, hidden_size)

        image_embedding = image_embedding.repeat(1, num_sequences, 1).view(batch_size * num_sequences, 1, -1)  # (batch_size * num_sequences, 1, hidden_size)


        # Kết hợp ảnh và caption
        lstm_input = torch.cat((image_embedding, caption_embedding), dim=1)  # (batch_size, seq_len+1, hidden_size)

        # Đưa vào LSTM
        lstm_out, _ = self.caption_lstm(lstm_input)  # (batch_size, seq_len+1, hidden_size)

        # Dự đoán từ tiếp theo tại mỗi bước
        output = self.fc(lstm_out)  # (batch_size, seq_len+1, vocab_size)

        return output[:, -1:, :]  # Loại bỏ bước đầu tiên (ảnh) để dự đoán từ tiếp theo
