import torch
import torch.nn as nn
import torch.nn.functional as F


class BigKernel(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, features=7):
        super().__init__()

        #self.encoder1_u = DoubleConv(3, features, kernel_size = 200, padding_same=True)
        self.big_kernel = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size = 40, padding="same"))
        
    
    def forward(self, emitted_light):
            """
            emitted_light: H,W,3
            """
            H, W = int(emitted_light.shape[0]), int(emitted_light.shape[1])
            # Reshape depth to match batch dimension
            emitted_light = emitted_light.permute(2, 0, 1) #(3,H,W)

            # Encoder
            yyy = F.interpolate(emitted_light[None,...], scale_factor=0.2, mode='bilinear', align_corners=True)
            yyy = yyy.permute(1,0,2,3)
            enc1 = self.big_kernel(yyy)
            enc1 = enc1.permute(1,0,2,3)
            up_downsampled_img = F.interpolate(enc1, size=(300, 533), mode='bilinear', align_corners=True)

            return up_downsampled_img 

class NeuradDecoderFlare(nn.Module):
    def __init__(self, in_channels=9+3+1, out_channels=3, hidden_dim=7, highest_hw = None):
        super().__init__()

        self.rgb_decoder = torch.nn.Sequential(        
                BasicBlock(in_channels, hidden_dim, kernel_size=5, padding=2, use_bn=False),
                BasicBlock(hidden_dim, hidden_dim, kernel_size=5, padding=2, use_bn=False),
            )
        self.rgb_decoder2 = torch.nn.Sequential(
                torch.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                torch.nn.ReLU(inplace=True),   
                torch.nn.Conv2d(hidden_dim, 3, kernel_size=1, padding=0),
                torch.nn.Sigmoid(),   
        )
        self.highest_hw = highest_hw
    
    def forward(self, emitted_light):
            """
            emitted_light: H,W,idk
            """
            H, W = int(emitted_light.shape[0]), int(emitted_light.shape[1])
            H,W = emitted_light.shape[0],emitted_light.shape[1]
            x = torch.linspace(0, 1, W)    # Normalized x coordinates (width)
            y = torch.linspace(0, 1, H)    # Normalized y coordinates (height)
            X, Y = torch.meshgrid(x, y, indexing='xy')  # Create grids
            coordinates = torch.stack([X, Y], dim=-1).to("cuda") # Shape (H, W, 2)

            emitted_light = torch.cat([emitted_light, coordinates], dim=-1)
            # Reshape depth to match batch dimension
            emitted_light = emitted_light.permute(2, 0, 1) #(3,H,W)

            # Encoder
            yyy = F.interpolate(emitted_light[None,...], scale_factor=0.4, mode='bilinear', align_corners=True)
            enc1 = self.rgb_decoder(yyy)
            up_downsampled_img = F.interpolate(enc1, size=(self.highest_hw[0], self.highest_hw[1]), mode='bilinear', align_corners=True)
            up_downsampled_img = self.rgb_decoder2(up_downsampled_img)

            return up_downsampled_img 

class NeuradDecoder_o(nn.Module):
    def __init__(self, in_dim=14, hidden_dim=32, rgb_upsample_factor=1):
        super().__init__()

        self.rgb_decoder = torch.nn.Sequential(
                torch.nn.Conv2d(in_dim, hidden_dim, kernel_size=1, padding=0),
                torch.nn.ReLU(inplace=True),
                BasicBlock(hidden_dim, hidden_dim, kernel_size=3, padding=1, use_bn=True),
                BasicBlock(hidden_dim, hidden_dim, kernel_size=3, padding=1, use_bn=True),
                torch.nn.Conv2d(hidden_dim, 3, kernel_size=1, padding=0),
                torch.nn.Sigmoid(),
            )
    def forward(self, emitted_light):
        H,W = emitted_light.shape[0],emitted_light.shape[1]
        x = torch.linspace(0, 1, W)    # Normalized x coordinates (width)
        y = torch.linspace(0, 1, H)    # Normalized y coordinates (height)
        X, Y = torch.meshgrid(x, y, indexing='xy')  # Create grids
        coordinates = torch.stack([X, Y], dim=-1).to("cuda") # Shape (H, W, 2)
        emitted_light = torch.cat([emitted_light,coordinates], dim=-1)

        emitted_light = emitted_light.permute(2, 0, 1) #(C,H,W)
        emitted_light = emitted_light[None,...] #(1,C,H,W)
        out = self.rgb_decoder(emitted_light)[0] #(3,H,W)
        return out.permute(1,2,0) ##(H,W,3)

class NeuradDecoder(nn.Module):
    def __init__(self, in_dim=11, hidden_dim=32, rgb_upsample_factor=1):
        super().__init__()

        self.rgb_decoder = torch.nn.Sequential(
                torch.nn.Conv2d(in_dim, hidden_dim, kernel_size=1, padding=0),
                torch.nn.ReLU(inplace=True),
                BasicBlock(hidden_dim, hidden_dim, kernel_size=3, padding=1, use_bn=True),
                BasicBlock(hidden_dim, hidden_dim, kernel_size=3, padding=1, use_bn=True),
                torch.nn.Conv2d(hidden_dim, 12+3, kernel_size=1, padding=0),
            )

    def get_ray_directions(self, H, W, intrinsics, extrinsics):
        """
        Get ray directions for each pixel in the image.
        
        Args:
            H (int): Image height
            W (int): Image width
            intrinsics (torch.Tensor): Camera intrinsics matrix (3x3)
            extrinsics (torch.Tensor): Camera extrinsics matrix (4x4), camera_to_world transform
            
        Returns:
            directions (torch.Tensor): Ray directions in world space with shape (H, W, 3)
        """
        # Create pixel coordinates grid
        i, j = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        i, j = i.to("cuda"), j.to("cuda")
        
        # Convert to homogeneous coordinates
        x = (j - intrinsics[0, 2]) / intrinsics[0, 0]
        y = (i - intrinsics[1, 2]) / intrinsics[1, 1]
        z = torch.ones_like(x).to("cuda")
        
        # Stack to create pixel_points of shape (H, W, 3)
        pixel_points = torch.stack([x, y, z], dim=-1)
        
        # Extract rotation matrix from extrinsics (camera_to_world)
        # The rotation is the top-left 3x3 portion of the extrinsics
        rotation = extrinsics[:3, :3]
        
        # Transform the direction vectors to world coordinates
        # For each pixel, apply the rotation matrix
        directions = torch.matmul(pixel_points.reshape(-1, 3), rotation.T)
        directions = directions.reshape(H, W, 3)
        
        # Normalize the directions
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)
        
        return directions
        
    def forward(self, emitted_light, intrinsics, extrinsics):
        H,W = emitted_light.shape[0],emitted_light.shape[1]

        x = torch.linspace(0, 1, W)    # Normalized x coordinates (width)
        y = torch.linspace(0, 1, H)    # Normalized y coordinates (height)
        X, Y = torch.meshgrid(x, y, indexing='xy')  # Create grids
        coordinates = torch.stack([X, Y], dim=-1).to("cuda") # Shape (H, W, 2)
        emitted_light = torch.cat([emitted_light, coordinates], dim=-1)

        # directions = self.get_ray_directions(H, W, intrinsics, extrinsics)
        # emitted_light = torch.cat([emitted_light, directions], dim=-1)

        emitted_light = emitted_light.permute(2, 0, 1) #(C,H,W)
        emitted_light = emitted_light[None,...] #(1,C,H,W)
        output = self.rgb_decoder(emitted_light) #(12,H,W)
        # Split into M and B
        M_flat = output[:, :9, :, :]  # Shape: (batch, 9, H, W)
        B = output[:, 9:12, :, :]       # Shape: (batch, 3, H, W)
        C = output[:, 12:, :, :]       # Shape: (batch, 3, H, W)
        
        # Reshape M into (batch, H, W, 3, 3)
        M = M_flat.permute(0, 2, 3, 1).view(-1, H, W, 3, 3)
        B = B.permute(0, 2, 3, 1)
        C = C.permute(0, 2, 3, 1)[0]

        return M,B,torch.clip(C,0)

class ResidualBlock(nn.Module):
    """Abstract Residual Block class."""

    def __init__(self, in_dim: int, dim: int) -> None:
        super().__init__()
        if in_dim != dim:
            self.res_branch = nn.Conv2d(in_dim, dim, kernel_size=1)
        else:
            self.res_branch = nn.Identity()
        self.main_branch = nn.Identity()
        self.final_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_activation(self.res_branch(x) + self.main_branch(x))


class BasicBlock(ResidualBlock):
    """Basic residual block."""

    def __init__(self, in_dim: int, dim: int, kernel_size: int, padding: int, use_bn: bool = False):
        super().__init__(in_dim, dim)
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_dim, dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(dim) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(dim) if use_bn else nn.Identity(),
        )

