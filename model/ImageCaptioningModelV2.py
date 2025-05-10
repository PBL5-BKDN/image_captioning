import torch
import torch.nn as nn

from model.Embedding import Embedding
from model.PatchImage import PatchEmbedding
from model.TransformerDecoder import TransformerDecoder
from model.TransformerEncoder import TransformerEncoder


class ImageCaptionModelV2(nn.Module):
    def __init__(self,w2i,glove_tensor,img_size=224, patch_size=16, embed_dim=300, num_heads = 4, units = 512, vocab_size = 10000, max_len = 50 ):
        super(ImageCaptionModelV2, self).__init__()
        self.w2i = w2i
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
        print("features shape: ", features.shape)
        encoded_features = self.encoder(features) # (batch, num_patches, embed_dim)
        print("encoded_features shape: ", encoded_features.shape)
        mask = (inputs != self.w2i["<PAD>"])
        embeddings = self.embedding(inputs)  # (batch, seq_len, embed_dim)
        print("embeddings shape: ", embeddings.shape)
        outputs = self.decoder(embeddings, encoded_features, mask=mask) # (batch, seq_len, vocab_size)
        return outputs


image_caption_model = ImageCaptionModelV2(w2i={"<PAD>": 0, "<UNK>": 1}, glove_tensor=torch.randn(10000, 300))
fake_images = torch.randn(2, 3, 224, 224)
fake_inputs = torch.randint(0, 10000, (2, 30))
outputs = image_caption_model(fake_images, fake_inputs)
print(outputs.shape)



