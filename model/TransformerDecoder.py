import torch.nn as nn
import torch




class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim = 300, num_heads= 4, vocab_size=10000):
        super(TransformerDecoder, self).__init__()
        self.vocab_size = vocab_size

        self.attention_1 = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True)
        self.layer_norm_1 = nn.LayerNorm(embed_dim, eps=1e-5)

        self.attention_2 = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1,  batch_first=True)
        self.layer_norm_2 = nn.LayerNorm(embed_dim, eps=1e-5)


        self.ffn_layer_1 = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
        )

        self.dropout_1 = nn.Dropout(0.3)

        self.ffn_layer_2 = nn.Sequential(
            nn.Linear(4 * embed_dim,embed_dim),
            nn.ReLU(),
        )

        self.layer_norm_3 = nn.LayerNorm(embed_dim, eps=1e-5)

        self.dropout_2 = nn.Dropout(0.5)

        self.out = nn.Sequential(
            nn.Linear(embed_dim, vocab_size),
        )





    def forward(self, embeddings, encoder_output, decoder_mask = None, encoder_mask = None):
        """
        :param embeddings: (B, T, E) - decoder input embeddings
        :param encoder_output: (B, S, E) - encoded image patches
        :param decoder_mask: (B, T) - mask cho decoder (1: hợp lệ, 0: pad)
        :param encoder_mask: (B, S) - mask cho encoder (1: hợp lệ, 0: pad)
        :return:
        """
        seq_len = embeddings.size(1)

        # Tạo causal mask với kích thước (seq_len, seq_len)
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=embeddings.device))
        causal_mask = causal_mask.masked_fill(causal_mask == 0, float('-inf')).masked_fill(causal_mask == 1, float(0.0))


        decoder_key_padding_mask = (~decoder_mask).float() if decoder_mask is not None else None



        attn_output_1, attn_weights = self.attention_1(
            embeddings,
            embeddings,
            embeddings,
            attn_mask=causal_mask,
            key_padding_mask=decoder_key_padding_mask
        )

        out_1 = self.layer_norm_1(embeddings + attn_output_1)

        encoder_key_padding_mask = ~encoder_mask if encoder_mask is not None else None
        attn_output_2, _ = self.attention_2(
            query=out_1,
            value=encoder_output,
            key=encoder_output,
            key_padding_mask=encoder_key_padding_mask
        )

        out_2 = self.layer_norm_2(out_1 + attn_output_2)

        ffn_output = self.ffn_layer_1(out_2)
        ffn_output = self.dropout_1(ffn_output)
        ffn_output = self.ffn_layer_2(ffn_output)
        ffn_output = self.layer_norm_3(ffn_output + out_2)
        ffn_output = self.dropout_2(ffn_output)
        preds = self.out(ffn_output)

        return preds



