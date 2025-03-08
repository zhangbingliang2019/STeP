import torch
import torch.nn.functional as F
from einops import rearrange
from .base import register_operator, Operator

def uniform_kernel_1d(kernel_size: int, dtype=torch.float32):
    """Generate a 1D uniform blur kernel."""
    if kernel_size <= 0:
        raise ValueError("Kernel size must be positive")
    
    kernel = torch.ones(kernel_size, dtype=dtype)
    kernel = kernel / kernel.sum()
    return kernel

def temporal_blur(video_tensor: torch.Tensor, kernel_size_t: int):
    video_tensor = rearrange(video_tensor, 'b t c h w -> b c t h w').to(torch.float32)
    device = video_tensor.device
    dtype = video_tensor.dtype
    B, C, T, H, W = video_tensor.shape

    # Generate Gaussian kernels for each dimension
    kernel_t = uniform_kernel_1d(kernel_size_t, dtype=dtype).to(device).view(1, 1, kernel_size_t, 1, 1) #* kernel_size_t**2

    padding_t = kernel_size_t // 2

    # Apply temporal blur
    video_tensor = F.pad(video_tensor, (0, 0, 0, 0, padding_t, padding_t), mode='circular')
    video_tensor = F.conv3d(video_tensor.view(B * C, 1, T + 2 * padding_t, H, W), kernel_t, padding=0, groups=1).view(B, C, T, H, W)
    video_tensor = rearrange(video_tensor, 'b c t h w -> b t c h w')
    return video_tensor

def PatchUpsample(x, scale):
    x = F.interpolate(x, scale_factor=scale, mode='nearest')
    return x

def PatchDownsample(x, scale):
    N, B = x.shape[:2]
    x = torch.nn.AdaptiveAvgPool2d((x.shape[-2]//scale, x.shape[-1]//scale))(x.flatten(0, 1))
    x = x.view(N, B, *x.shape[1:])
    return x

def gaussian_kernel_2d(kernel_size: int, sigma: float, dtype=torch.float32):
    """Generate a 2D Gaussian kernel."""
    ax = torch.arange(kernel_size, dtype=dtype) - (kernel_size - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel

def gaussian_blur(tensor: torch.Tensor, kernel_size: int, sigma: float):
    device = tensor.device
    dtype = tensor.dtype
    N, B, C, H, W = tensor.shape
    kernel = gaussian_kernel_2d(kernel_size, sigma).to(device)
    kernel = kernel.view(1, 1, kernel_size, kernel_size).expand(C, 1, kernel_size, kernel_size)
    padding = kernel_size // 2
    # Apply padding
    tensor_padded = F.pad(tensor.flatten(0, 1), (padding, padding, padding, padding), mode='reflect')
    # Apply 2D convolution with groups=C
    blurred_tensor = F.conv2d(tensor_padded, kernel, padding=0, groups=C).view(N, B, C, H, W)
    return blurred_tensor.to(torch.float16)

def generate_random_mask(shape, pixel_ratio):
    """
    Generates a random binary mask with the given pixel ratio.

    Args:
        shape (tuple): Shape of the mask (B, C, H, W).
        pixel_ratio (float): Ratio of pixels to be set to 1.

    Returns:
        torch.Tensor: Random binary mask.
    """
    N, B, C, H, W = shape
    num_pixels = H * W
    num_ones = int(num_pixels * pixel_ratio)
    
    # Generate a flat array with the appropriate ratio of ones and zeros
    flat_mask = torch.zeros(num_pixels, dtype=torch.float32)
    flat_mask[:num_ones] = 1
    
    # Shuffle to randomize the positions of ones and zeros
    flat_mask = flat_mask[torch.randperm(num_pixels)]
    
    # Reshape to the original spatial dimensions and duplicate across channels
    mask = flat_mask.view(1, H, W)
    mask = mask.expand(N, B, C, H, W)
    
    return mask


@register_operator('+deblur')
class GaussianDeblurTemporal(Operator):
    def __init__(self, kernel_size_t=7, gaussian_kernel_size_s=61, intensity_s=3.0,  sigma=0.00):
        super().__init__(sigma)
        self.deg_t = lambda z: temporal_blur(z, kernel_size_t)
        self.deg_s = lambda z: gaussian_blur(z, gaussian_kernel_size_s, intensity_s)
        self.deg_sT = lambda z: z

    def __call__(self, x):
        dtype = x.dtype
        return self.deg_s(self.deg_t(x.float())).to(dtype)


@register_operator('+sr')
class SuperResolutionTemporal(Operator):
    def __init__(self, kernel_size_t=7, scale_factor=4, sigma=0.00):
        super().__init__(sigma)
        self.deg_t = lambda z: temporal_blur(z, kernel_size_t)
        self.deg_s = lambda z: PatchDownsample(z, scale_factor)
        self.deg_sT = lambda z: PatchUpsample(z, scale_factor)

    def __call__(self, x):
        dtype = x.dtype
        return self.deg_s(self.deg_t(x.float())).to(dtype)


@register_operator('+inpainting')
class InpaintingTemporal(Operator):
    def __init__(self, kernel_size_t=7, pixel_ratio=0.5, resolution=256, device='cuda', sigma=0.00):
        super().__init__(sigma)
        self.mask = generate_random_mask((1, 1, 1, resolution, resolution), pixel_ratio).to(device)
        self.deg_t = lambda z: temporal_blur(z, kernel_size_t)
        self.deg_s = lambda z: z*self.mask
        self.deg_sT = lambda z: z*self.mask    
        self.device = device
    
    def __call__(self, x):
        dtype = x.dtype
        return self.deg_s(self.deg_t(x.float())).to(dtype)
    