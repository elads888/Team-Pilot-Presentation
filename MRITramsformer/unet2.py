import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim


# ------------------------------------------
# UNet Encoder (Downsampling Path)
# ------------------------------------------
class UNetEncoder(nn.Module):
    def __init__(self, in_channels=2, base_channels=8):
        super().__init__()

        self.enc1 = self.conv_block(in_channels, base_channels)
        self.enc2 = self.conv_block(base_channels, base_channels * 2)
        self.enc3 = self.conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self.conv_block(base_channels * 4, base_channels * 6)
        self.enc5 = self.conv_block(base_channels * 6, base_channels * 8)

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
        x = x.squeeze(1).permute(0,3,1,2)
        #x - (batch_size, 2, 384,144)
        enc1 = self.enc1(x)  # (batch_size, base_channels, 384, 144 )
        enc2 = self.enc2(self.pool(enc1))  # (batch_size, base_channels*2, 192, 72)
        enc3 = self.enc3(self.pool(enc2))  # (batch_size, base_channels*4, 96, 36)
        enc4 = self.enc4(self.pool(enc3))  # (batch_size, base_channels*6, 48, 18)
        enc5 = self.enc5(self.pool(enc4))  # (batch_size, base_channels*8, 24, 9)
        bottleneck = self.pool3(enc5)   # (batch_size, base_channels*8, 8, 3)
        batch_size = bottleneck.size()[0]
        out_channels = bottleneck.size()[1]
        bottleneck = bottleneck.permute(0,3,1,2).reshape(batch_size, out_channels, 8*3)
        # bottleneck - (batch_size, 64, 24)
        # print(f"Encoder Output Shapes:")
        # print(
        #     f"enc1: {enc1.shape}, enc2: {enc2.shape}, enc3: {enc3.shape}, enc4: {enc4.shape}, bottleneck: {bottleneck.shape}")

        return bottleneck