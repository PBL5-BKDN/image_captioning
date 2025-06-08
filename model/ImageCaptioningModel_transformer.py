import torch
import torch.nn as nn

from model.CNNEncoder import CNNEncoder
from model.Embedding import Embedding
from model.PositionalEncoding import PositionalEncoding
from model.TransformerDecoder import TransformerDecoder
from model.TransformerEncoder import TransformerEncoder
from settings import DEVICE


class ImageCaptionModel(nn.Module):
    def __init__(self,vocab, embed_dim=300, num_heads = 4, vocab_size = 10000, max_len = 50 ):
        super(ImageCaptionModel, self).__init__()
        self.w2i = vocab.w2i
        self.vocab = vocab

        self.cnn_model = CNNEncoder(embed_dim)
        self.encoder = TransformerEncoder(embed_dim, num_heads)
        self.embedding = Embedding(self.w2i, embed_dim=embed_dim)
        self.position_embedding = PositionalEncoding(embed_dim, max_len=max_len)
        self.decoder = TransformerDecoder( num_heads=num_heads, embed_dim=embed_dim, vocab_size=vocab_size )
    def forward(self, images, inputs):
        """
        :param images: tensor of shape (B, 3, 224, 224)
        :param images:
        :param input:
        :return:
        """
        features = self.cnn_model(images)
        encoded_features = self.encoder(features)
        mask = (inputs != self.w2i["<PAD>"])
        embeddings = self.embedding(inputs)  + self.position_embedding(inputs) # (batch, seq_len, embed_dim)
        outputs = self.decoder(embeddings, encoded_features, decoder_mask=mask)
        return outputs

    def generate_caption(self, image, max_len=30):
        self.eval()
        with torch.no_grad():
            image = image.to(DEVICE).unsqueeze(0)  # (1, 3, 224, 224)
            features = self.cnn_model(image)
            encoder_output = self.encoder(features)

            input_ids = [self.vocab.w2i["<START>"]]

            for _ in range(max_len):
                input_tensor = torch.tensor([input_ids], device=DEVICE)

                mask = (input_tensor != self.vocab.w2i["<PAD>"])


                embeddings = self.embedding(input_tensor) + self.position_embedding(input_tensor)  # (batch, seq_len, embed_dim)
                outputs = self.decoder(embeddings, encoder_output, decoder_mask=mask)

                next_token_logits = outputs[0, -1, :]  # Lấy token cuối cùng
                next_token = torch.argmax(next_token_logits).item()

                if next_token == self.vocab.w2i["<END>"]:
                    break

                input_ids.append(next_token)

            caption = [self.vocab.i2w[idx] for idx in input_ids[1:]]  # Bỏ <START>
            return " ".join(caption)




