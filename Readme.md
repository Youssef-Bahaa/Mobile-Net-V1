# MobileNetV1 in PyTorch

PyTorch implementation of [*MobileNets: Efficient Convolutional Neural Networks for Mobile
Vision Applications*](https://arxiv.org/abs/1704.04861) (Howard et al., 2017).

> The model, training loop and evaluation script are complete. Training on ImageNet has not
> been run, so this repo reports no accuracy results.

## The paper's approach

**1. Depthwise separable convolutions.** A standard 3x3 convolution filters space and mixes
channels in one step, which is expensive. MobileNet splits it in two:

- a **3x3 depthwise** convolution filters each channel on its own (spatial work only)
- a **1x1 pointwise** convolution mixes the channels (channel work only)

This gives nearly the same result for roughly 8-9x less computation. Each convolution is
followed by BatchNorm and ReLU.

**2. The network.** One standard 3x3 convolution, then 13 depthwise separable blocks, then
global average pooling and a fully connected layer. Stride-2 blocks halve the image size
(224 -> 112 -> 56 -> 28 -> 14 -> 7) while channels grow from 32 to 1024.

**3. Two multipliers to trade accuracy for cost.**

- **Width multiplier (alpha):** scales every layer's channel count, e.g. `alpha=0.5`.
- **Resolution multiplier (rho):** scales the input size (224 / 192 / 160 / 128). It is not a
  model argument; it only changes the input images (`model.rho` in `config.yaml`).

## Code

```
models/blocks.py         ConvBNReLU, DepthwiseSeparableConv
models/mobilenet_v1.py   MobileNetV1(num_classes, alpha)
data/imagenet.py         ImageNet loaders
train.py / evaluate.py   training and evaluation, configured by config.yaml
```

Training follows the paper where it gives details (RMSprop, light augmentation, no label
smoothing, no weight decay on depthwise filters). The paper gives no learning-rate schedule,
so the values in `config.yaml` are the common PyTorch adaptation, not the paper's.

## Usage

```bash
pip install -r requirements.txt
python train.py
python evaluate.py --checkpoint checkpoints/best_model.pth
```

Put ImageNet in `datasets/imagenet/` as `train/<class>/` and `val/<class>/`
