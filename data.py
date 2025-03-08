from abc import ABC, abstractmethod
from PIL import Image
import torchvision.transforms as transforms
import torch
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import functional as TF
import warnings
import os
import glob

__DATASET__ = {}


def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            if __DATASET__[name] != cls:
                warnings.warn(f"Name {name} is already registered!", UserWarning)
        __DATASET__[name] = cls
        cls.name = name
        return cls

    return wrapper


def get_dataset(name: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __DATASET__[name](**kwargs)


class DiffusionData(ABC, Dataset):
    @abstractmethod
    def get_shape(self):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def __len__(self):
        pass

    def get_data(self, size=16, sigma=0):
        data = torch.stack([self.__getitem__(i) for i in range(size)], dim=0)
        return data + torch.randn_like(data) * sigma

    def get_random(self, size=16, sigma=2e-3):
        shape = (size, *self.get_shape())
        return torch.randn(shape) * sigma

    def summary(self):
        print('+------------------------------------+')
        print('Dataset Summary')
        print(f"Dataset : {self.name}")
        print(f"Shape   : {self.get_shape()}")
        print(f"Length  : {len(self)}")
        print('+------------------------------------+')



@register_dataset('mri_image')
class MRIImage(DiffusionData):
    def __init__(self, root='dataset/mri', resolution=192, random_flip=True, clip=1, scaling=1e6, length=50000, image_type='amplitude'):
        self.root = Path(root)
        # find all .npy files in root
        pathlist = list(self.root.rglob('*.npy'))
        self.pathlist = sorted(pathlist)
        self.resolution = resolution
        self.length = min(len(pathlist), length)
        self.random_flip = random_flip
        self.clip = clip
        self.scaling = scaling
        self.image_type = image_type # 'real+imag', 'amplitude+phase', 'amplitude'
       
    def __getitem__(self, idx):
        image = torch.from_numpy(np.load(str(self.pathlist[idx])))
        # image = self.get_zoom_in_out(image)
        # normalize image to around [-1, 1]
        if self.scaling is not None:
            image *= self.scaling
        if self.clip is not None:
            image = image.clip(-self.clip, self.clip)
    
        if self.random_flip and np.random.rand() < 0.5:
            image = torch.flip(image, [2])  # left-right flip
        if self.random_flip and np.random.rand() < 0.5:
            image = torch.flip(image, [1])  # top-down flip
        
        if self.image_type == 'real+imag':
            out = image
        elif self.image_type == 'amplitude+phase':
            real = image[0]
            imag = image[1]
            amplitude = torch.sqrt(real**2 + imag**2) * np.sqrt(2) - 1
            phase = torch.atan2(imag, real) / np.pi
            out = torch.stack([amplitude, phase], dim=0)
        elif self.image_type == 'amplitude':
            real = image[0]
            imag = image[1]
            amplitude = torch.sqrt(real**2 + imag**2) * np.sqrt(2) - 1
            out = amplitude.unsqueeze(0)
        return out.float()

    def get_shape(self):
        return 2, self.resolution, self.resolution

    def __len__(self):
        return self.length
  

@register_dataset('mri_video')
class MRIVideo(DiffusionData):
    def __init__(self, root='dataset/mri', resolution=192, frames=12, random_flip=True, clip=1, scaling=1e6, length=50000, image_type='amplitude'):
        self.root = Path(root)
        pathlist = []
        for folder in self.root.glob('*'):
            pathlist += list(folder.glob('*'))
        self.pathlist = sorted(pathlist)
        self.resolution = resolution
        self.frames = frames
        self.length = min(len(pathlist), length)
        self.random_flip = random_flip
        self.clip = clip
        self.scaling = scaling
        self.image_type = image_type # 'real+imag', 'amplitude+phase', 'amplitude'

    def __getitem__(self, idx):
        npy_list = [self.pathlist[idx]/'timeidx={}.npy'.format(i) for i in range(self.frames)]
        video = torch.stack([torch.from_numpy(np.load(str(path))) for path in npy_list]) # [12, 2, H, W]
        # image = self.get_zoom_in_out(image)
        # normalize image to around [-1, 1]
        if self.scaling is not None:
            video *= self.scaling
        if self.clip is not None:
            video = video.clip(-self.clip, self.clip)
    
        if self.random_flip and np.random.rand() < 0.5:
            video = torch.flip(video, [3])  # left-right flip
        if self.random_flip and np.random.rand() < 0.5:
            video = torch.flip(video, [2])  # top-down flip
        
        if self.image_type == 'real+imag':
            out = video
        elif self.image_type == 'amplitude+phase':
            real = video[0]
            imag = video[1]
            amplitude = torch.sqrt(real**2 + imag**2) * np.sqrt(2) - 1
            phase = torch.atan2(imag, real) / np.pi
            out = torch.stack([amplitude, phase], dim=0)
        elif self.image_type == 'amplitude':
            real = video[0]
            imag = video[1]
            amplitude = torch.sqrt(real**2 + imag**2) * np.sqrt(2) - 1
            out = amplitude.unsqueeze(0)
        return out.float()

    def get_shape(self):
        return 2, self.resolution, self.resolution

    def __len__(self):
        return self.length
  

@register_dataset('blackhole_image')
class BlackHoleImage(DiffusionData):
    def __init__(self, root='data/blackhole_image', resolution=256, original_resolution=400,
                 random_flip=True, zoom_in_out=True, zoom_range=[0.75, 1.25], length=50000):  # [0.833, 1.145]
        super().__init__()
        self.root = root
        pathlist = list(Path(root).glob('*.npy'))
        self.pathlist = sorted(pathlist)
        self.resolution = resolution
        self.original_resolution = original_resolution
        self.length = min(len(pathlist), length)
        self.random_flip = random_flip
        self.zoom_in_out = zoom_in_out
        self.zoom_range = zoom_range

    def __len__(self):
        return self.length

    def get_zoom_in_out(self, img):
        if self.zoom_in_out:
            scale = np.random.uniform(self.zoom_range[0], self.zoom_range[1])
            zoom_shape = [
                int(self.resolution * scale),
                int(self.resolution * scale)
            ]
            img = TF.resize(img, zoom_shape, antialias=True)
            if zoom_shape[0] > self.resolution:
                img = TF.center_crop(img, self.resolution)
            elif zoom_shape[0] < self.resolution:
                diff = self.resolution - zoom_shape[0]
                img = TF.pad(
                    img,
                    (diff // 2 + diff % 2, diff // 2 + diff % 2, diff // 2, diff // 2)
                )
        else:
            img = TF.resize(img, (self.resolution, self.resolution), antialias=True)
        return img

    def __getitem__(self, idx):
        image = torch.from_numpy(np.load(str(self.pathlist[idx])))
        image = self.get_zoom_in_out(image)
        # normalize image
        image /= image.max()
        image = 2 * image - 1
        
        if self.random_flip and np.random.rand() < 0.5:
            image = torch.flip(image, [2])  # left-right flip
        if self.random_flip and np.random.rand() < 0.5:
            image = torch.flip(image, [1])  # top-down flip
        return image.float()

    def get_shape(self):
        return 1, self.resolution, self.resolution


@register_dataset('blackhole_video')
class BlackHoleVideo(DiffusionData):
    def __init__(self, root='data/blackhole_video', resolution=256, original_resolution=400,
                 frames=64, random_flip=True, zoom_in_out=True, zoom_range=[0.75, 1.25], permute=False, total_frames=1000, length=1000):  # [0.833, 1.145]
        super().__init__()
        self.root = root
        pathlist = list(Path(root).glob('*'))
        self.pathlist = sorted(pathlist)
        self.resolution = resolution
        self.original_resolution = original_resolution
        self.length = min(len(pathlist), length)
        self.random_flip = random_flip
        self.zoom_in_out = zoom_in_out
        self.zoom_range = zoom_range
        self.frames = frames
        self.permute = permute
        self.total_frames = total_frames

    def __len__(self):
        return self.length

    def get_zoom_in_out(self, img):
        if self.zoom_in_out:
            scale = np.random.uniform(self.zoom_range[0], self.zoom_range[1])
            zoom_shape = [
                int(self.resolution * scale),
                int(self.resolution * scale)
            ]
            img = TF.resize(img, zoom_shape, antialias=True)
            if zoom_shape[0] > self.resolution:
                img = TF.center_crop(img, self.resolution)
            elif zoom_shape[0] < self.resolution:
                diff = self.resolution - zoom_shape[0]
                img = TF.pad(
                    img,
                    (diff // 2 + diff % 2, diff // 2 + diff % 2, diff // 2, diff // 2)
                )
        else:
            img = TF.resize(img, (self.resolution, self.resolution), antialias=True)
        return img

    @staticmethod
    def get_random_frame_list(frames, total_frames):
        if total_frames == frames:
            start = 0
        else:
            start = np.random.randint(0, total_frames - frames)
        return list(range(start, start + frames))

    def __getitem__(self, idx):
        frame_list = self.get_random_frame_list(self.frames, self.total_frames)
        pathlist = [self.pathlist[idx] / '{:05d}.npy'.format(i) for i in frame_list]
        video = torch.stack([torch.from_numpy(np.load(str(path))) for path in pathlist])

        video = self.get_zoom_in_out(video)
        # normalize image
        video /= video.max()
        video = 2 * video - 1
        
        if self.random_flip and np.random.rand() < 0.5:
            video = torch.flip(video, [3])  # left-right flip
        if self.random_flip and np.random.rand() < 0.5:
            video = torch.flip(video, [2])  # top-down flip
        if self.permute:
            video = video.permute(1, 0, 2, 3) # [C, T, H, W]
        return video.float()

    def get_shape(self):
        return self.frames, 1, self.resolution, self.resolution

        