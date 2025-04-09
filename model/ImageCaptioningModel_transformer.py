import torch.nn as nn

from model.CNNEncoder import CNNEncoder
from model.Embedding import Embedding
from model.TransformerDecoder import TransformerDecoder
from model.TransformerEncoder import TransformerEncoder


class ImageCaptionModel(nn.Module):
    def __init__(self,w2i, glove_tensor, embed_dim=300, num_heads = 4, units = 512, vocab_size = 10000, max_len = 50 ):
        super(ImageCaptionModel, self).__init__()
        self.w2i = w2i

        self.cnn_model = CNNEncoder(embed_dim)
        self.encoder = TransformerEncoder(embed_dim, num_heads)
        self.embedding = Embedding(w2i,glove_tensor, embed_dim, vocab_size=vocab_size, max_len_seq=max_len)

        self.decoder = TransformerDecoder(w2i,glove_tensor,units, embed_dim, num_heads, vocab_size, max_len)
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
        embeddings = self.embedding(inputs)  # (batch, seq_len, embed_dim)
        outputs = self.decoder(embeddings, encoded_features, mask=mask)
        return outputs






