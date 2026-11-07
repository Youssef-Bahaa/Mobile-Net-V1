import torch
import torch.nn as nn

from models.blocks import ConvBNReLU, DepthwiseSeparableConv


class MobileNetV1(nn.Module):

    def __init__(self, num_classes=1000, alpha=1.0):
        super().__init__()

        self.alpha = alpha

        # Width Multiplier (exact for the paper's alphas: 1.0, 0.75, 0.5, 0.25)
        def c(channels):
            return int(channels * alpha)

        self.features = nn.Sequential(

            # Initial Convolution
            ConvBNReLU(3, c(32), stride=2),

            # Depthwise Separable Convolutions
            DepthwiseSeparableConv(c(32), c(64)),

            DepthwiseSeparableConv(c(64), c(128), stride=2),
            DepthwiseSeparableConv(c(128), c(128)),

            DepthwiseSeparableConv(c(128), c(256), stride=2),
            DepthwiseSeparableConv(c(256), c(256)),

            DepthwiseSeparableConv(c(256), c(512), stride=2),

            DepthwiseSeparableConv(c(512), c(512)),
            DepthwiseSeparableConv(c(512), c(512)),
            DepthwiseSeparableConv(c(512), c(512)),
            DepthwiseSeparableConv(c(512), c(512)),
            DepthwiseSeparableConv(c(512), c(512)),

            DepthwiseSeparableConv(c(512), c(1024), stride=2),
            DepthwiseSeparableConv(c(1024), c(1024))
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(c(1024), num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x