import torch
import torch.nn.functional as F

def get_loss_normal_tensor(tensor) : 
    
    # Define the patch size
    N = torch.randint(low=15,high=50,size=(1,))[0]
    
    # Unfold the first two dimensions (W and H) to extract patches
    patches = tensor.unfold(0, N, N).unfold(1, N, N)  # [W//N, H//N, N, N, 6]

    # Sum over the patches
    patch_mean = patches.mean(dim=(-3, -2))  # Sum over N, N 
    #print("Patch sums shape:", patch_mean.shape)  # Should be [W//N, H//N, 6]

    loss = 10 * torch.sum(torch.square(patch_mean ))

    return loss

def get_loss_mask_flare(image_infos, flare, use_both = False):
    assert ( list(image_infos["flare"].shape) == list(flare.shape) )
    if list(flare.shape) != [450, 800, 3]  :
        return 0.0
     
    is_flare = image_infos["flare"] 

    loss = -1 * torch.sum(is_flare * flare) / (flare.shape[0] * flare.shape[1])
    if use_both :
        loss += torch.mean(torch.square(is_flare - flare))
    else:
        loss *= 1

    return loss    

def depth_to_normals_opengl_mm4(depth):
    """
    Convert a depth map to surface normals in OpenGL's coordinate system.
    (Camera looks along -Z, +Y is up, +X is right).
    
    Args:
        depth (torch.Tensor): Depth map of shape (H, W).
    
    Returns:
        torch.Tensor: Normals tensor of shape (H, W, 3) in OpenGL coordinates.
    """
    H, W = depth.shape
    depth = depth.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)

    # Sobel kernels for gradient computation
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                          dtype=torch.float32, device=depth.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                          dtype=torch.float32, device=depth.device).view(1, 1, 3, 3)

    # Compute gradients with replicate padding
    grad_x = F.conv2d(depth, sobel_x, padding=1) / 8.0
    grad_y = F.conv2d(depth, sobel_y, padding=1) / 8.0

    grad_x, grad_y = grad_x.squeeze(), grad_y.squeeze()  # Remove batch and channel dims

    # Compute normals and adjust for OpenGL conventions:
    normals = torch.stack((-grad_x, -grad_y, torch.ones_like(grad_x)), dim=0)
    normals = F.normalize(normals, p=2, dim=0)  # L2 normalize

    return normals.permute(1, 2, 0)  # Shape: (H, W, 3)

def depth_to_world_normals_opengl_mm4(depth, camera_to_world):
    """
    Convert depth to normals in **world coordinates** (OpenGL convention).
    
    Args:
        depth (torch.Tensor): Depth map of shape (H, W).
        camera_to_world (torch.Tensor): Camera-to-world extrinsic (4x4).
    
    Returns:
        torch.Tensor: Normals in world coordinates, shape (H, W, 3).
    """
    # Get normals in OpenGL camera coordinates
    normals_cam = depth_to_normals_opengl_mm4(depth)  # Shape (H, W, 3)

    # Transform to world coordinates using rotation part of extrinsics
    R = camera_to_world[:3, :3]  # (3, 3) rotation matrix
    H, W = normals_cam.shape[:2]
    normals_world = torch.einsum('ijk,lk->ijl', normals_cam, R)  # Apply rotation

    # Re-normalize (optional, but ensures numerical stability)
    normals_world = F.normalize(normals_world, p=2, dim=-1)
    return normals_world



def quaternion_to_rotation_matrix(q):
    """
    Convert quaternions (w, x, y, z) to rotation matrices.
    
    Args:
        q (torch.Tensor): Quaternions of shape (N, 4) in (w, x, y, z) order.
    
    Returns:
        torch.Tensor: Rotation matrices of shape (N, 3, 3).
    """
    w, x, y, z = q.unbind(dim=-1)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    xx, yy, zz = x * x, y * y, z * z

    # Construct rotation matrix columns
    row0 = torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1)
    row1 = torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1)
    row2 = torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1)
    
    R = torch.stack([row0, row1, row2], dim=-2)
    return R

def get_gaussian_axes(quaternions, scales):
    """
    Compute the two longest axes and the normal (shortest axis) for each 3D Gaussian.
    
    Args:
        quaternions (torch.Tensor): Quaternions of shape (N, 4) in (w, x, y, z) order.
        scales (torch.Tensor): Scale factors of shape (N, 3) for each axis (sx, sy, sz).
    
    Returns:
        tuple: (axis1, axis2, normal) each of shape (N, 3), where axis1 and axis2 are the
               two longest axes sorted by scale (descending), and normal is the shortest axis.
    """
    N = quaternions.size(0)
    device = quaternions.device
    
    # Convert quaternions to rotation matrices (N, 3, 3)
    R = quaternion_to_rotation_matrix(quaternions)
    
    # Find the index of the smallest scale for each Gaussian
    min_scale_idx = torch.argmin(scales, dim=1)  # (N,)
    
    # Precompute the other two indices for each Gaussian
    others = torch.tensor([[1, 2], [0, 2], [0, 1]], device=device)
    other_indices = others[min_scale_idx]  # (N, 2)
    
    # Sort the other indices by their scales in descending order
    other_scales = torch.gather(scales, 1, other_indices)  # (N, 2)
    sorted_order = torch.argsort(other_scales, dim=1, descending=True)  # (N, 2)
    sorted_other_indices = torch.gather(other_indices, 1, sorted_order)  # (N, 2)
    
    # Extract the axes
    axis1 = R[torch.arange(N, device=device), :, sorted_other_indices[:, 0]]
    axis2 = R[torch.arange(N, device=device), :, sorted_other_indices[:, 1]]
    normal = R[torch.arange(N, device=device), :, min_scale_idx]
    
    return axis1, axis2, normal

def flip_opposing_normals(gs_means, gs_normals, camera_to_world):
    """
    Flip normals that point away from the camera.
    
    Args:
        gs_means: Tensor of shape (N, 3) with 3D positions of Gaussians
        gs_normals: Tensor of shape (N, 3) with current normal directions
        camera_to_world: Camera extrinsics matrix (4x4) transforming from camera to world space
    
    Returns:
        Tensor of shape (N, 3) with corrected normals
    """
    # Extract camera position from the extrinsics matrix (last column of inverse)
    # camera_to_world is the extrinsic matrix that transforms from camera to world
    # The camera position in world coordinates is the translation part
    camera_position = camera_to_world[:3, 3]  # Shape: (3,)
    
    # Calculate vectors from Gaussians to camera
    # For each Gaussian position, compute direction vector to camera
    directions_to_camera = camera_position.unsqueeze(0) - gs_means  # Shape: (N, 3)
    
    # Normalize these direction vectors
    directions_to_camera = -1*directions_to_camera / torch.norm(directions_to_camera, dim=1, keepdim=True)
    
    # Compute dot product between normals and directions to camera
    # If dot product is negative, normal points away from camera
    dot_products = torch.sum(gs_normals * directions_to_camera, dim=1)  # Shape: (N,)
    
    # Create a mask for normals that need to be flipped (dot product < 0)
    flip_mask = dot_products < 0  # Shape: (N,)
    
    # Create a copy of the normals tensor
    corrected_normals = gs_normals.clone()
    
    # Flip normals where the mask is True
    corrected_normals[flip_mask] = -corrected_normals[flip_mask]
    
    return corrected_normals