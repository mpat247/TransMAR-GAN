import torch
import torch.nn as nn
import torch.nn.functional as F
import os

import scipy.io as io

# RedCNN model in PyTorch
class RedCNN(nn.Module):
    def __init__(self):
        super(RedCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 96, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(96, 96, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(96, 96, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(96, 96, kernel_size=5, stride=1, padding=2)
        self.conv5 = nn.Conv2d(96, 96, kernel_size=5, stride=1, padding=2)
        
        self.deconv1 = nn.ConvTranspose2d(96, 96, kernel_size=5, stride=1, padding=2)
        self.deconv2 = nn.ConvTranspose2d(96, 96, kernel_size=5, stride=1, padding=2)
        self.deconv3 = nn.ConvTranspose2d(96, 96, kernel_size=5, stride=1, padding=2)
        self.deconv4 = nn.ConvTranspose2d(96, 1, kernel_size=5, stride=1, padding=2)
        
    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x5 = F.relu(self.conv5(x4))
        
        x6 = F.relu(self.deconv1(x5) + x4)
        x7 = F.relu(self.deconv2(x6))
        x8 = F.relu(self.deconv3(x7) + x2)
        x9 = F.relu(self.deconv4(x8))
        
        output = F.relu(x9 + x)
        return output

# Transformer model definition for denoising
class DenoisingTransformer(nn.Module):
    def __init__(self, input_channels=1, num_layers=6, num_heads=8, d_model=64, dim_feedforward=256):
        super(DenoisingTransformer, self).__init__()
        self.input_proj = nn.Conv2d(input_channels, d_model, kernel_size=3, padding=1)
        self.positional_encoding = nn.Parameter(torch.randn(1, d_model, 64, 64))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Conv2d(d_model, input_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Apply input projection
        x = self.input_proj(x)
        b, c, h, w = x.shape

        # Adjust positional encoding size dynamically
        positional_encoding = F.interpolate(self.positional_encoding, size=(h, w), mode='bilinear', align_corners=False)
    
        # Add positional encoding and reshape for transformer
        x = x + positional_encoding[:, :, :h, :w]
        x = x.flatten(2).permute(2, 0, 1)  # (seq_len, batch, d_model)

        # Apply transformer encoder
        x = self.transformer_encoder(x)
        x = x.permute(1, 2, 0).view(b, -1, h, w)  # (batch, d_model, h, w)

        # Apply output projection
        x = self.output_proj(x)
        return x


# Spatial Attention (SA) Module
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=1)
        
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        
        out = conv1 * conv2
        out = conv3 * out
        out = self.conv4(out)
        
        return x + out

# Channel Attention (CA) Module
class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        
    def forward(self, x):
        avg_out = self.avg_pool(x)
        conv1 = F.relu(self.conv1(avg_out))
        conv2 = torch.sigmoid(self.conv2(conv1))
        
        return x * conv2

# BAFB Module
class BAFB(nn.Module):
    def __init__(self, in_channels):
        super(BAFB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.sa = SpatialAttention(64)
        self.ca = ChannelAttention(64)
        self.conv2 = nn.Conv2d(256, 1, kernel_size=1)
        
    def forward(self, x):
        fcr1 = F.relu(self.conv1(x))
        fsa1 = self.sa(fcr1)
        fes_up = fsa1 + fcr1
        fca1 = self.ca(fcr1)
        fes_down = fca1 + fcr1
        fca2 = self.ca(fes_up)
        fsa2 = self.sa(fes_down)
        fcr2 = torch.cat([fca2, fes_up, fes_down, fsa2], dim=1)
        fcr2 = self.conv2(fcr2)
        fc = torch.cat([fcr1, fcr2], dim=1)
        fc = self.conv2(fc)
        return fc

# Boosting Module Groups (BMG)
class BMG(nn.Module):
    def __init__(self, in_channels):
        super(BMG, self).__init__()
        self.bafb1 = BAFB(in_channels)
        self.bafb2 = BAFB(1)
        self.bafb3 = BAFB(1)
        self.bafbn = BAFB(1)
        
    def forward(self, x):
        out = self.bafb1(x)
        out = self.bafb2(out)
        out = self.bafb3(out)
        out = self.bafbn(out)
        return out + x

# BAF ResNet
class BAFResNet(nn.Module):
    def __init__(self):
        super(BAFResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.bmg1 = BMG(64)
        self.bmg2 = BMG(1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv = nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=2, dilation=2)
        
    def forward(self, x):
        f1sf = F.relu(self.bn1(self.conv1(x)))
        f2sf = F.relu(self.bn2(self.conv2(f1sf)))
        f3sf = F.relu(self.bn3(self.conv3(f2sf + x)))
        bmg1_out = self.bmg1(f3sf)
        bmg2_out = self.bmg2(bmg1_out)
        f4sf = F.relu(self.bn3(self.conv3(bmg2_out + f3sf)))
        out = F.relu(self.deconv(f4sf))
        return out

# Instantiate and test models
if __name__ == "__main__":
    # RedCNN Model
    model = RedCNN()
    print(model)
    # Test RedCNN
    input_tensor = torch.rand((1, 1, 256, 256))  # Example input
    output = model(input_tensor)
    print("RedCNN Output Shape:", output.shape)
    
    # BAFResNet Model
    baf_resnet = BAFResNet()
    print(baf_resnet)
    # Test BAFResNet
    output_resnet = baf_resnet(input_tensor)
    print("BAFResNet Output Shape:", output_resnet.shape)

