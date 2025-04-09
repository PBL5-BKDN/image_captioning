import torch
import torch.nn as nn

from model.PositionalEncoding import PositionalEncoding


class Embedding(nn.Module):
    def __init__(self,w2i, glove_tensor, embed_dim = 300, vocab_size = 10000, max_len_seq = 50):
        super(Embedding, self).__init__()

        self.token_embedding = nn.Embedding.from_pretrained(glove_tensor, freeze=True,padding_idx=w2i["<PAD>"])
        # self.pos_embedding = PositionalEncoding(embed_dim, max_len_seq)
        self.pos_embedding = nn.Embedding(max_len_seq, embed_dim)
    def forward(self, x):
        """
        :param x: tensor of shape (B, MAX_SEQUENCE_LENGTH)
        :return: tensor of shape (B,MAX_SEQUENCE_LENGTH, EMBED_DIM)
        """
        token_embedding = self.token_embedding(x)  ## (B, S, E)
        seq_length = x.shape[1]
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0)  # (1, S)
        pos_embed = self.pos_embedding(positions)  # (1, S, E)

        # pos_embed = self.pos_embedding(x)  # (1, S, E)
        return token_embedding + pos_embed


