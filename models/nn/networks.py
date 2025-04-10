import torch
import torch.nn as nn
import torch.nn.functional as F


class BigKernel(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, features=7):
        super().__init__()

        #self.encoder1_u = DoubleConv(3, features, kernel_size = 200, padding_same=True)
        self.big_kernel = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size = 40, padding="same"))
        
    
    def forward(self, emitted_light):
            """
            emitted_light: H,W,3
            """
            H, W = int(emitted_light.shape[0]), int(emitted_light.shape[1])
            # Reshape depth to match batch dimension
            emitted_light = emitted_light.permute(2, 0, 1) #(3,H,W)

            # Encoder
            yyy = F.interpolate(emitted_light[None,...], scale_factor=0.2, mode='bilinear', align_corners=True)
            enc1 = self.big_kernel(yyy)
            up_downsampled_img = F.interpolate(enc1, size=(450, 800), mode='bilinear', align_corners=True)

            return up_downsampled_img 
             