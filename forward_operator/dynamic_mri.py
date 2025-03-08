from .base import Operator, register_operator
import torch
from torch.fft import fftn, fftshift
import numpy as np


@register_operator('mri')
class DynamicMRI(Operator):
    def __init__(self, acc_factor, mode='same_mask', resolution=192, frames=12, num_acs_lines=12, sigma=0):
        super().__init__(sigma)
        self.acc_factor = acc_factor
        N_FRAMES = frames
        half_resolution = resolution // 2
        if mode == 'same_mask':
            self.mask = torch.zeros(1, 1, 1, resolution, 1)
            self.mask[:, :, :, ::acc_factor, :] = 1
            self.mask[:, :, :, half_resolution-num_acs_lines//2:half_resolution+num_acs_lines//2, :] = 1
        elif mode == 'split_mask':
            self.mask_1d = np.zeros(resolution)
            self.mask_1d[::acc_factor] = 1
            self.mask_1d[half_resolution-num_acs_lines//2:half_resolution+num_acs_lines//2] = 1
            selected_lines = np.where(self.mask_1d)[0]
            selected_lines_split = np.array_split(selected_lines, N_FRAMES)
            self.mask = torch.zeros(1, N_FRAMES, 1, resolution, 1)
            for i in range(N_FRAMES):
                self.mask[:, i, :, selected_lines_split[i], :] = 1
        elif mode == 'random_split_mask':
            np.random.seed(42)
            self.mask_1d = np.zeros(resolution)
            self.mask_1d[::acc_factor] = 1
            self.mask_1d[half_resolution-num_acs_lines//2:half_resolution+num_acs_lines//2] = 1
            selected_lines = np.where(self.mask_1d)[0]
            np.random.shuffle(selected_lines)
            selected_lines_split = np.array_split(selected_lines, N_FRAMES)
            self.mask = torch.zeros(1, N_FRAMES, 1, resolution, 1)
            for i in range(N_FRAMES):
                self.mask[:, i, :, selected_lines_split[i], :] = 1
        else:
            raise ValueError('Unknown mode: {}'.format(mode))
    

    def __call__(self, x):
        # [B, T, C, H, W] -> [B, obs_dim]
        x = x[:, :, 0, ...] + 1j * x[:, :, 1, ...]
        x = fftshift(x, dim=(-2, -1))
        y = fftn(x, dim=(-2, -1))
        y = torch.view_as_real(y).permute(0, 1, 4, 2, 3).contiguous()
        return self.mask.to(x.device) * y