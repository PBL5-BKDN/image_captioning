import torch
import torch.nn as nn

from model.DecoderVIT import DecoderVIT
from model.EncoderVIT import EncoderVIT
from settings import DEVICE


class VIT(nn.Module):
    def __init__(self,vocab, embed_dim, num_heads,max_len, num_blocks=2, mlp_hidden_size=1024):
        super(VIT, self).__init__()
        self.vocab = vocab
        self.encoder_vit = EncoderVIT(embed_dim, num_heads, num_blocks, mlp_hidden_size)
        self.decoder_vit = DecoderVIT(vocab.w2i, embed_dim, num_heads, vocab.vocab_size, num_blocks, mlp_hidden_size, max_len=max_len)

    def forward(self, images, captions):
        encoder_output = self.encoder_vit(images)
        decoder_output = self.decoder_vit(captions, encoder_output)
        return decoder_output

    def generate_caption(self, image):
        self.eval()
        with torch.no_grad():
            image = image.to(DEVICE).unsqueeze(0)  # (1, 3, 224, 224)
            features = self.patch_embedding(image)
            encoder_output = self.encoder(features)

            input_ids = [self.vocab.w2i["<START>"]]

            for _ in range(self.max_len):
                input_tensor = torch.tensor([input_ids], device=DEVICE)

                mask = (input_tensor != self.vocab.w2i["<PAD>"])
                encoder_mask = torch.ones((image.size(0), features.size(1)), dtype=torch.bool, device=image.device)

                embeddings = self.embedding(input_tensor)  # (batch, seq_len, embed_dim)
                output = self.decoder(embeddings, encoder_output, decoder_mask=mask, encoder_mask=encoder_mask)

                next_token_logits = output[0, -1, :]  # Lấy token cuối cùng
                next_token = torch.argmax(next_token_logits).item()

                if next_token == self.vocab.w2i["<END>"]:
                    break

                input_ids.append(next_token)

            caption = [self.vocab.i2w[idx] for idx in input_ids[1:]]  # Bỏ <START>
            return " ".join(caption)
