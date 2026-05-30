import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
# ------------------------------------------
# UNet Encoder (Downsampling Path)
# ------------------------------------------
class UNetEncoder(nn.Module):
    def __init__(self, in_channels=24, base_channels=64):
        super().__init__()
        self.enc1 = self.conv_block(in_channels, base_channels)
        self.enc2 = self.conv_block(base_channels, base_channels * 2)
        self.enc3 = self.conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self.conv_block(base_channels * 4, (base_channels * 8)+1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=3)
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        batch, channel, height, width, complex = x.size()
        x = x.permute(1, 2, 3, 0, 4).reshape(channel, height, width, batch * complex)
        x = x.permute(0, 3, 1, 2)
        enc1 = self.enc1(x)  # (batch_size, 64, 384, 144)
        enc2 = self.enc2(self.pool(enc1))  # (batch_size, 128, 192, 72)
        enc3 = self.enc3(self.pool(enc2))  # (batch_size, 256, 96, 36)
        enc4 = self.enc4(self.pool(enc3))  # (batch_size, 512, 48, 18)
        bottleneck = self.pool3(enc4)  # (batch_size, 512, 24, 9)
        # print(f"Encoder Output Shapes:")
        # print(
        #     f"enc1: {enc1.shape}, enc2: {enc2.shape}, enc3: {enc3.shape}, enc4: {enc4.shape}, bottleneck: {bottleneck.shape}")
        return bottleneck