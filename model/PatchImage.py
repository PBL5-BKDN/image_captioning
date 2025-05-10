import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Linear projection from patch to embedding
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)  # Init giống ViT gốc

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)             # [B, embed_dim, H/P, W/P]
        x = x.flatten(2)             # [B, embed_dim, N_patches]
        x = x.transpose(1, 2)        # [B, N_patches, embed_dim]

        # Add positional embedding
        x = x + self.pos_embed       # [B, N_patches, embed_dim]
        return x
