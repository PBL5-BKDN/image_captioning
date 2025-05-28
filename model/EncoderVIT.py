import torch.nn as nn

from model.PatchEmbedding import PatchEmbedding


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_hidden_size=1024):
        super(EncoderBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.1)
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_size, embed_dim),
            nn.Dropout(0.1)
        )
        self.layer_norm_2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        :param x: (B, Num patchs, embedding dim)
        :return:
            (B, Num patchs, embedding dim)
        """
        x = x + self.attn(x, x, x, need_weights=False)[0]
        x = self.layer_norm_1(x)
        x = self.ffn(x) + x
        x = self.layer_norm_2(x)
        return x


class EncoderVIT(nn.Module):
    def __init__(self, embed_dim, num_heads, num_blocks = 2, mlp_hidden_size=1024):
        super(EncoderVIT, self).__init__()
        self.patch_embedding = PatchEmbedding(img_size=224, patch_size=16, in_channels=3, embed_dim=embed_dim)

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, mlp_hidden_size) for _ in range(num_blocks)
        ])

        self.layer_norm = nn.LayerNorm(embed_dim)

        self.embed_dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        :param x: (B, C, H, W)
        :return:
            (B, Num patchs, embedding dim)
        """
        x = self.patch_embedding(x)
        x = self.embed_dropout(x)
        for block in self.encoder_blocks:
            x = block(x)
        return x

