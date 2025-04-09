import torch.nn as nn
import torch

from model.Embedding import Embedding


class TransformerDecoder(nn.Module):
    def __init__(self,w2i, glove_tensor, units, embed_dim = 300, num_heads= 4, vocab_size=10000, max_len = 50):
        super(TransformerDecoder, self).__init__()
        self.vocab_size = vocab_size

        self.attention_1 = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True)
        self.layer_norm_1 = nn.LayerNorm(embed_dim, eps=1e-5)

        self.attention_2 = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1,  batch_first=True)
        self.layer_norm_2 = nn.LayerNorm(embed_dim, eps=1e-5)


        self.ffn_layer_1 = nn.Sequential(
            nn.Linear(embed_dim, units),
            nn.ReLU(),
        )

        self.dropout_1 = nn.Dropout(0.3)

        self.ffn_layer_2 = nn.Sequential(
            nn.Linear(units,embed_dim),
            nn.ReLU(),
        )

        self.layer_norm_3 = nn.LayerNorm(embed_dim, eps=1e-5)

        self.dropout_2 = nn.Dropout(0.5)

        self.out = nn.Sequential(
            nn.Linear(embed_dim, vocab_size),
        )





    def forward(self, embeddings, encoder_output, mask = None):
        """
        :param input_ids: tensor of shape (B, MAX_SEQUENCE_LENGTH)
        :param encoder_output:
        :param mask:
        :return:
        """
        seq_len = embeddings.size(1)

        # Tạo causal mask với kích thước (seq_len, seq_len)
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=embeddings.device))
        causal_mask = causal_mask.masked_fill(causal_mask == 0, float('-inf')).masked_fill(causal_mask == 1, float(0.0))


        key_padding_mask = (~mask).float()

        attn_output_1, attn_weights = self.attention_1(
            embeddings,
            embeddings,
            embeddings,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask
        )

        out_1 = self.layer_norm_1(embeddings + attn_output_1)

        attn_output_2, _ = self.attention_2(
            query=out_1,
            value=encoder_output,
            key=encoder_output,

        )

        out_2 = self.layer_norm_2(out_1 + attn_output_2)

        ffn_output = self.ffn_layer_1(out_2)
        ffn_output = self.dropout_1(ffn_output)
        ffn_output = self.ffn_layer_2(ffn_output)
        ffn_output = self.layer_norm_3(ffn_output + out_2)
        ffn_output = self.dropout_2(ffn_output)
        preds = self.out(ffn_output)

        return preds



