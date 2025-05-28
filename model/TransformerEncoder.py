import torch.nn as nn
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(TransformerEncoder, self).__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.1)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        :param x: (B, S, E)
        :return:
        """
        x = self.layer_norm_1(x)
        x = self.fc(x)
        attention_output, _ = self.attention(x, x, x)
        output = self.layer_norm_2(x + attention_output)
        return output

