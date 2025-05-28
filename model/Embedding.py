import os

import torch
import torch.nn as nn

from model.GloveEmbedding import load_glove_embeddings
from model.PositionalEncoding import PositionalEncoding
from settings import BASE_DIR


class Embedding(nn.Module):
    def __init__(self,w2i, embed_dim = 300):
        super(Embedding, self).__init__()
        glove_path = os.path.join(BASE_DIR, "model", "glove.6B", f"glove.6B.{embed_dim}d.txt")
        glove_tensor = load_glove_embeddings(glove_path, w2i, embedding_dim=embed_dim)
        self.token_embedding = nn.Embedding.from_pretrained(glove_tensor, freeze=True,padding_idx=w2i["<PAD>"])
    def forward(self, x):
        """
        :param x: tensor of shape (B, MAX_SEQUENCE_LENGTH)
        :return: tensor of shape (B,MAX_SEQUENCE_LENGTH, EMBED_DIM)
        """
        token_embedding = self.token_embedding(x)  ## (B, S, E)

        return token_embedding


