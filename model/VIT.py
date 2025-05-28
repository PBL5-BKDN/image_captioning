import torch
import torch.nn as nn

from model.DecoderVIT import DecoderVIT
from model.EncoderVIT import EncoderVIT


class VIT(nn.Module):
    def __init__(self,vocab, embed_dim, num_heads,max_len, num_blocks=2, mlp_hidden_size=1024):
        super(VIT, self).__init__()
        self.encoder_vit = EncoderVIT(embed_dim, num_heads, num_blocks, mlp_hidden_size)
        self.decoder_vit = DecoderVIT(vocab.w2i, embed_dim, num_heads, vocab.vocab_size, num_blocks, mlp_hidden_size, max_len=max_len)

    def forward(self, images, captions):
        encoder_output = self.encoder_vit(images)
        decoder_output = self.decoder_vit(captions, encoder_output)
        return decoder_output
