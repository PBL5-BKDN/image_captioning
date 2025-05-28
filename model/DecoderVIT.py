import torch.nn as nn
import torch

from model.Embedding import Embedding
from model.PositionalEncoding import PositionalEncoding
from settings import DEVICE


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_hidden_size=1024):
        super(DecoderBlock, self).__init__()

        self.mask_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm_1 = nn.LayerNorm(embed_dim)

        self.e_d_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_size, embed_dim),
            nn.Dropout(0.1)
        )
        self.layer_norm_3 = nn.LayerNorm(embed_dim)

    def forward(self, X, encoder_output):
        """
        :param x: (B, Num patchs/num_steps/max_len, embedding dim)
        :return:
            (B, Num patchs, embedding dim)
        """
        num_steps = X.shape[1]
        # Masked attention
        mask = torch.triu(torch.ones(num_steps, num_steps, device=X.device), diagonal=1).bool()
        self_attended = self.mask_attn(X, X, X, attn_mask=mask)
        X = self.layer_norm_1(X + self_attended[0])

        # Encoder-decoder attention
        attended = self.e_d_attn(X, encoder_output, encoder_output)
        X = self.layer_norm_2(X + attended[0])

        # Feed forward network
        X = self.layer_norm_3(X + self.mlp(X))
        return X


class DecoderVIT(nn.Module):
    def __init__(self, w2i, embed_dim, num_heads, vocab_size, num_blocks=2, mlp_hidden_size=1024, max_len=30,):
        super(DecoderVIT, self).__init__()
        self.embedding = Embedding(w2i, embed_dim)
        self.pos_embedding = PositionalEncoding(embed_dim=embed_dim, max_len=max_len)

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, mlp_hidden_size) for _ in range(num_blocks)
        ])

        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, encoder_output):
        """
        :param x: (B, Num patchs, embedding dim)
        :return:
            (B, Num patchs, embedding dim)
        """
        x = self.embedding(x) + self.pos_embedding(x)

        for block in self.decoder_blocks:
            x = block(x, encoder_output)

        x = self.fc(x)
        return x