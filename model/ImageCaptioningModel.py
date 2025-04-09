import torch
import torch.nn as nn

from torchvision.models import Inception_V3_Weights
import torchvision.models as models
import fasttext.util


class Attention(nn.Module):
    """Bahdanau Attention Mechanism"""
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        """
        hidden: (batch_size, hidden_size)
        encoder_outputs: (batch_size, seq_len, hidden_size)
        """
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, hidden_size)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # (batch_size, seq_len, hidden_size)
        attention_weights = torch.softmax(self.v(energy), dim=2)  # (batch_size, seq_len, 1)
        context = torch.sum(attention_weights * encoder_outputs, dim=1)  # (batch_size, hidden_size)
        return context, attention_weights

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size=5000, image_feature_size=2048, hidden_size=256,num_layers=2, vocab=None):
        self.vocab = vocab
        self.num_layers = num_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(ImageCaptioningModel, self).__init__()
        model_inception_v3 = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
        model_inception_v3.fc = torch.nn.Identity()  # Giữ lại feature vector trước lớp fully connected
        model_inception_v3.eval()
        self.inception_v3 = model_inception_v3
        self.image_encoder = nn.Sequential(
            nn.Linear(image_feature_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        # Embedding cho special tokens
        self.special_token_embedding = nn.Embedding(len(vocab.special_tokens), 300)
        self.ft = fasttext.load_model("cc.vi.300.bin")
        self.caption_projection = nn.Linear(300, hidden_size)

        self.attention = Attention(hidden_size)

        self.caption_lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=self.num_layers,
                                    batch_first=True, dropout=0.5, bidirectional=False)

        self.layer_norm = nn.LayerNorm(hidden_size)

        self.fc = nn.Linear(hidden_size, vocab_size)

    def get_embedding(self, caption_tokens):
        batch_size, max_length = caption_tokens.shape
        embeddings = torch.zeros(batch_size, max_length, 300, device=self.device)

        for i in range(batch_size):
            for j in range(max_length):
                token_id = caption_tokens[i, j].item()
                word = self.vocab.i2w[token_id]

                if word in self.vocab.special_tokens:
                    token_index = self.vocab.special_tokens.index(word)
                    embed = self.special_token_embedding.weight[token_index]
                else:
                    embed = torch.tensor(self.ft.get_word_vector(word), dtype=torch.float32, device=self.device)

                embeddings[i, j] = embed

        return embeddings


    def forward(self, images, captions):
        """
       image: (batch_size, 3, 299, 299)
       caption_tokens: (batch_size, seq_len)
       """

        with torch.no_grad():
            features = self.inception_v3(images)  # (batch_size, 2048)
            if isinstance(features, tuple):
                 features = features[0]
        image_embedding = self.image_encoder(features)  # (batch_size, hidden_size)

        num_directions = 2 if self.caption_lstm.bidirectional else 1
        h0 = image_embedding.unsqueeze(0).repeat(self.num_layers * num_directions, 1, 1)
        c0 = torch.zeros_like(h0)

        caption_embedding = self.get_embedding(captions)

        caption_embedding = self.caption_projection(caption_embedding)  # (batch_size, max_length-1, hidden_size)


        lstm_out, _ = self.caption_lstm(caption_embedding, (h0, c0))  # (batch_size, max_length-1, hidden_size)

        context, _ = self.attention(image_embedding, lstm_out)

        lstm_out = self.layer_norm(lstm_out + context.unsqueeze(1))

        output = self.fc(lstm_out)  # (batch_size, max_length-1, vocab_size)

        return output
