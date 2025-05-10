import torch
import torch.nn as nn

from dataset.stanford_Image_Paragraph_Captioning_dataset.build_vocab import vocab
from model.Embedding import Embedding
from model.PatchImage import PatchEmbedding
from model.TransformerDecoder import TransformerDecoder
from model.TransformerEncoder import TransformerEncoder
from settings import DEVICE


class ImageCaptionModelV2(nn.Module):
    def __init__(self,w2i,glove_tensor,img_size=224, patch_size=16, embed_dim=300, num_heads = 4, units = 512, vocab_size = 10000, max_len = 50 ):
        super(ImageCaptionModelV2, self).__init__()
        self.w2i = w2i
        self.max_len = max_len

        self.patch_embedding = PatchEmbedding(embed_dim=embed_dim, img_size=img_size, patch_size=patch_size)
        self.encoder = TransformerEncoder(embed_dim, num_heads)
        self.embedding = Embedding(w2i,glove_tensor, embed_dim, vocab_size=vocab_size, max_len_seq=max_len)

        self.decoder = TransformerDecoder(w2i,glove_tensor,units, embed_dim, num_heads, vocab_size, max_len)
    def forward(self, images, inputs):
        """
        :param images: tensor of shape (B, 3, 224, 224)
        :param input: tensor of shape (B, seq_len)
        :return:
        """
        features = self.patch_embedding(images) # (batch, num_patches, embed_dim)
        encoded_features = self.encoder(features) # (batch, num_patches, embed_dim)
        mask = (inputs != self.w2i["<PAD>"])
        embeddings = self.embedding(inputs)  # (batch, seq_len, embed_dim)
        outputs = self.decoder(embeddings, encoded_features, mask=mask) # (batch, seq_len, vocab_size)
        return outputs

    def generate_caption(self,image):
        self.eval()
        with torch.no_grad():
            image = image.to(DEVICE).unsqueeze(0)  # (1, 3, 224, 224)
            features = self.patch_embedding(image)
            encoder_output = self.encoder(features)

            input_ids = [self.w2i["<START>"]]

            for _ in range(self.max_len):
                input_tensor = torch.tensor([input_ids], device=DEVICE)

                mask = (input_tensor != self.w2i["<PAD>"])
                embeddings = self.embedding(input_tensor)  # (batch, seq_len, embed_dim)
                output = self.decoder(embeddings, encoder_output, mask=mask)

                next_token_logits = output[0, -1, :]  # Lấy token cuối cùng
                next_token = torch.argmax(next_token_logits).item()

                if next_token == self.w2i["<END>"]:
                    break

                input_ids.append(next_token)

            caption = [vocab.i2w[idx] for idx in input_ids[1:]]  # Bỏ <START>
            return " ".join(caption)





