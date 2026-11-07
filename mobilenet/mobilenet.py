import torch
import torch.nn as nn

from models.blocks import ConvBNReLU, DepthwiseSeparableConv


class MobileNetV1(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(

            # Initial Convolution
            ConvBNReLU(3, 32, stride=2),

            # Depthwise Separable Convolutions
            DepthwiseSeparableConv(32, 64),

            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128),

            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256),

            DepthwiseSeparableConv(256, 512, stride=2),

            DepthwiseSeparableConv(512, 512),
            DepthwiseSeparableConv(512, 512),
            DepthwiseSeparableConv(512, 512),
            DepthwiseSeparableConv(512, 512),
            DepthwiseSeparableConv(512, 512),

            DepthwiseSeparableConv(512, 1024, stride=2),
            DepthwiseSeparableConv(1024, 1024)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
