import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class EncoderCNN(nn.Module):
    def __init__(self, encoder_dim=256, encoded_image_size=14):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.conv = nn.Conv2d(512, encoder_dim, 1)
        self.bn = nn.BatchNorm2d(encoder_dim)

    def forward(self, images):
        x = self.backbone(images)
        x = self.pool(x)
        x = self.bn(self.conv(x))
        x = x.permute(0, 2, 3, 1)
        return x.view(x.size(0), -1, x.size(-1))
