# %% imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%  Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1)
        self.pool2 = nn.AdaptiveMaxPool2d((2, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        return x
    
# %%  Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.up1 = nn.Upsample(scale_factor=(8, 16), mode='nearest')
        self.convT1 = nn.ConvTranspose2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.convT2 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.up1(x)
        x = F.relu(self.convT1(x))
        x = self.up2(x)
        x = F.relu(self.convT2(x))
        return x
    
# %%  Autoencoder
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, x):
        h = self.encoder(x)
        r = self.decoder(h)
        return r, h
    
