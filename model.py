import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention3D(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super(MultiHeadSelfAttention3D, self).__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.head_dim = in_channels // num_heads
        
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        
        self.qkv = nn.Linear(in_channels, in_channels * 3)
        self.proj = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        B, C, D, H, W = x.shape
        N = D * H * W
        
        x_flat = x.view(B, C, N).permute(0, 2, 1)
        
        qkv = self.qkv(x_flat).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        attn_output = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        output = self.proj(attn_output)
        
        output = output.permute(0, 2, 1).view(B, C, D, H, W)
        return output + x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512 * 7 * 7, 2048)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(2048, 256 * 2 * 2 * 2)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 2, 2, 2)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        return x

class MAR_Refiner(nn.Module):
    def __init__(self):
        super(MAR_Refiner, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(1, 16, 3, padding=1), nn.BatchNorm3d(16), nn.LeakyReLU(0.2))
        self.pool1 = nn.MaxPool3d(2) 
        
        self.conv2 = nn.Sequential(nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32), nn.LeakyReLU(0.2))
        self.pool2 = nn.MaxPool3d(2) 
        
        self.conv3 = nn.Sequential(nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.LeakyReLU(0.2))
        self.pool3 = nn.MaxPool3d(2) 
        
        self.mha = MultiHeadSelfAttention3D(in_channels=64, num_heads=4)
        
        self.up3 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.dconv3 = nn.Sequential(nn.Conv3d(96, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(inplace=True))
        
        self.up2 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.dconv2 = nn.Sequential(nn.Conv3d(48, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(inplace=True))
        
        self.up1 = nn.ConvTranspose3d(16, 8, 2, stride=2)
        self.dconv1 = nn.Sequential(nn.Conv3d(24, 8, 3, padding=1), nn.BatchNorm3d(8), nn.ReLU(inplace=True))
        
        self.final_conv = nn.Sequential(nn.Conv3d(8, 1, 1), nn.Sigmoid())

    def forward(self, x):
        e1 = self.conv1(x)
        e2 = self.conv2(self.pool1(e1))
        e3 = self.conv3(self.pool2(e2))
        bottleneck = self.pool3(e3)
        
        bottleneck = self.mha(bottleneck)
        
        d3 = self.up3(bottleneck)
        d3 = self.dconv3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        d2 = self.dconv2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        d1 = self.dconv1(torch.cat([d1, e1], dim=1))
        
        return self.final_conv(d1)

class Pix2VoxWithMAR(nn.Module):
    def __init__(self):
        super(Pix2VoxWithMAR, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.refiner = MAR_Refiner()

    def forward(self, x):
        B, V, C, H, W = x.size()
        
        x = x.view(B * V, C, H, W)
        features = self.encoder(x)
        coarse_volumes = self.decoder(features)
        
        coarse_volumes = coarse_volumes.view(B, V, 1, 32, 32, 32)
        fused_volume = torch.mean(coarse_volumes, dim=1)
        
        refined_volume = self.refiner(fused_volume)
        return refined_volume