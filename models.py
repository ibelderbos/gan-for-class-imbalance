import torch
from torch import nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.channels = channels
        self.theta = nn.utils.spectral_norm(nn.Conv2d(channels, channels // 8, kernel_size=1, padding=0, bias=False))
        self.phi = nn.utils.spectral_norm(nn.Conv2d(channels, channels // 8, kernel_size=1, padding=0, bias=False))
        self.g = nn.utils.spectral_norm(nn.Conv2d(channels, channels // 2, kernel_size=1, padding=0, bias=False))
        self.o = nn.utils.spectral_norm(nn.Conv2d(channels // 2, channels, kernel_size=1, padding=0, bias=False))
        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x):
        spatial_size = x.shape[2] * x.shape[3]
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), kernel_size=2)
        g = F.max_pool2d(self.g(x), kernel_size=2)
        
        theta = theta.view(-1, self.channels // 8, spatial_size)
        phi = phi.view(-1, self.channels // 8, spatial_size // 4)
        g = g.view(-1, self.channels // 2, spatial_size // 4)
        
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), dim=-1)
        o = self.o(torch.bmm(g, beta.transpose(1, 2)).view(-1, self.channels // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x
    
    
class GenBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        super().__init__()
        
        self.conv = nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
    
class Generator(nn.Module):

    def __init__(self, z_dim, n_classes):
        super(Generator, self).__init__()
        self.input_dim =  z_dim + n_classes

        self.g_blocks = nn.ModuleList([
            
                GenBlock(self.input_dim, 512, kernel_size=4, stride=2),
                # AttentionBlock(512),
                GenBlock(512, 256, kernel_size=4, stride=2),
                # AttentionBlock(256),
                GenBlock(256, 128, kernel_size=4, stride=2),
                GenBlock(128, 64, kernel_size=5, stride=2),
            ])
     
        self.proj_o = nn.Sequential(
                nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2),
                nn.Tanh(),
            )
        
    def forward(self, noise):
        gen_input = noise.view(len(noise), self.input_dim, 1, 1)
             
        h = self.g_blocks[0](gen_input)

        for idx, g_block in enumerate(self.g_blocks):
            if idx != 0:
                h = g_block(h)
        h = self.proj_o(h)
        return h
    
def get_noise(n_samples, z_dim, device='cpu'):

    return torch.randn(n_samples, z_dim, device=device)


class DiscBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=2, stride=2):
        super().__init__()

        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    
class Discriminator(nn.Module):

    def __init__(self, data_shape, n_classes):
        super(Discriminator, self).__init__()
        self.input_dim = data_shape[0] + n_classes 

        self.g_blocks = nn.ModuleList([
                DiscBlock(self.input_dim, 64, kernel_size=2),
                # AttentionBlock(64),
                DiscBlock(64, 128, kernel_size=2),
                # AttentionBlock(128),
                DiscBlock(128, 256, kernel_size=4),
                DiscBlock(256, 512, kernel_size=4),
                
            ])
        
        self.proj_o = nn.Sequential(
                nn.Conv2d(512, 1, kernel_size=4, stride=2),
            )
        
    def forward(self, x):
        
        for idx, g_block in enumerate(self.g_blocks):
            if idx == 0:
                h = g_block(x)
            else:
                h = g_block(h)
        h = self.proj_o(h)
        return h.view(len(h), -1)
