import torch
import torch.nn as nn


from model.Embedding import Embedding
from model.PatchEmbedding import PatchEmbedding
from model.TransformerDecoder import TransformerDecoder
from model.TransformerEncoder import TransformerEncoder
from settings import DEVICE


class ImageCaptionModelV2(nn.Module):
    def __init__(self, vocab ,img_size=224, patch_size=16, embed_dim=300, num_heads = 4, vocab_size = 10000, max_len = 50 ):
        super(ImageCaptionModelV2, self).__init__()
        self.vocab = vocab
        self.max_len = max_len

        self.patch_embedding = PatchEmbedding(embed_dim=embed_dim, img_size=img_size, patch_size=patch_size)
        self.encoder = nn.Sequential(
            TransformerEncoder(embed_dim, num_heads),
            TransformerEncoder(embed_dim, num_heads),
        )
        self.embedding = Embedding(vocab.w2i, embed_dim)

        self.decoder = TransformerDecoder(embed_dim=embed_dim, num_heads=num_heads, vocab_size=vocab_size)
    def forward(self, images, inputs):
        """
        :param images: tensor of shape (B, 3, 224, 224)
        :param input: tensor of shape (B, seq_len)
        :return:
        """
        features = self.patch_embedding(images) # (batch, num_patches, embed_dim)
        encoded_features = self.encoder(features) # (batch, num_patches, embed_dim)
        mask = (inputs != self.vocab.w2i["<PAD>"])

        embeddings = self.embedding(inputs)  # (batch, seq_len, embed_dim)
        outputs = self.decoder(embeddings, encoded_features, decoder_mask=mask) # (batch, seq_len, vocab_size)
        return outputs

    def generate_caption(self,image):
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





