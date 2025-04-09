
from torchvision.models import ResNet50_Weights
from torch import nn
import torch
from torchvision import models
class CNNEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        # Sử dụng ResNet50 pretrained làm backbone
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Loại bỏ avgpool và fc
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.conv_proj = nn.Conv2d(2048, embed_dim, kernel_size=1)  # Dự phóng đặc trưng ResNet sang embed_dim
    def forward(self, image):
        """
        :param image: tensor of shape (B, 3, 224, 224)
        :return: features tensor of shape (B, 49, embed_dim)
        """
        features = self.backbone(image) #(B, 2048, 7, 7)
        features = self.conv_proj(features) #(B, embed_dim , 7, 7)
        features = features.flatten(2).transpose(1, 2)  # (B, 49, embed_dim)
        return features
