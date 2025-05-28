import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)


        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            B: Batch size
            C: Number of channels
            H: Height of the image
            W: Width of the image
        """
        B, C, H, W = x.shape
        P = self.patch_size
        assert H % P == 0 and W % P == 0, "Image size must be divisible by patch size"

        img = self.conv(x) # (B, embed_dim, H/P, W/P)
        img = img.flatten(2) # (B, embed_dim, num_patches)
        img = img.transpose(1, 2) # (B, num_patches, embed_dim)
        img = img + self.pos_embed  # Cộng positional embedding
        return img
