import torch
import pickle
import torch
import torch.nn as nn
from cores.scheduler import VPScheduler
from omegaconf import OmegaConf
import importlib
from abc import abstractmethod
from diffusers import StableDiffusionPipeline, DiffusionPipeline
import torch.nn.functional as F
import sys
import warnings

__MODEL__ = {}


def register_model(name: str):
    def wrapper(cls):
        if __MODEL__.get(name, None):
            if __MODEL__[name] != cls:
                warnings.warn(f"Name {name} is already registered!", UserWarning)
        __MODEL__[name] = cls
        cls.name = name
        return cls

    return wrapper


def get_model(name: str, **kwargs):
    if __MODEL__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __MODEL__[name](**kwargs)


class LatentDiffusionModel(nn.Module):
    """
    A class representing a latent diffusion model.
    Methods:
        encode(x0): Encodes the input `x0` into latent space.
        decode(z0): Decodes the latent variable `z0` into the output space.
        score(z, sigma): Calculates the score of the latent diffusion model given the latent variable `z` and standard deviation `sigma`.
        tweedie(z, sigma): Calculates the Tweedie distribution given the latent variable `z` and standard deviation `sigma`.
        Must overload either `score` or `tweedie` method.
    """

    def __init__(self):
        super(LatentDiffusionModel, self).__init__()
        # Check if either `score` or `tweedie` is overridden
        if (self.score.__func__ is LatentDiffusionModel.score and
                self.tweedie.__func__ is LatentDiffusionModel.tweedie):
            raise NotImplementedError(
                "Either `score` or `tweedie` method must be implemented."
            )

    @abstractmethod
    def encode(self, x0):
        pass

    @abstractmethod
    def decode(self, z0):
        pass

    def score(self, z, sigma, c=None):
        d = self.tweedie(z, sigma=sigma, c=c)
        return (d - z) / sigma ** 2

    def tweedie(self, z, sigma, c=None):
        # assert exactly one of t and sigma is not None
        return z + self.score(z, sigma=sigma, c=c) * sigma ** 2

    def get_data_shape():
        pass

    def get_latent_shape():
        pass


@register_model(name='st_diffusion')
class SpatialTemporalDiffusionModel(LatentDiffusionModel):
    def __init__(self, vdm_or_path, vae_or_path, scheduler_or_path, video_shape=(64, 1, 256, 256), latent_shape=(64, 1, 32, 32),  device='cuda'):
        super().__init__()
        self.device = device
        if isinstance(vdm_or_path, str):
            self.vdm = torch.load(vdm_or_path).to(device)
            print('load vdm from path:', vdm_or_path)
        else:
            self.vdm = vdm_or_path.to(device)
        if isinstance(vae_or_path, str):
            self.vae = torch.load(vae_or_path).to(device)
            print('load vae from path:', vae_or_path)
        else:
            self.vae = vae_or_path.to(device)
        if isinstance(scheduler_or_path, str):
            scheduler = torch.load(scheduler_or_path)
        else:
            scheduler = scheduler_or_path
    
        self.scheduler = VPScheduler(
            num_steps=scheduler.config.num_train_timesteps,
            beta_max=scheduler.config.beta_end * scheduler.config.num_train_timesteps,
            beta_min=scheduler.config.beta_start * scheduler.config.num_train_timesteps,
            epsilon=0,
            beta_type=scheduler.config.beta_schedule,
        )
        self.video_shape = video_shape
        self.latent_shape = latent_shape
        self.dtype = torch.float32
    
    def get_data_shape(self):
        return self.video_shape
    
    def get_latent_shape(self):
        return self.latent_shape
    
    def get_in_shape(self):
        return self.get_latent_shape()
    
    def encode(self, x0):
        # x0: (batch_size, frame_size, C, H, W)
        source_dtype = x0.dtype
        x0 = x0.to(self.dtype)
        batch_size, frame_size = x0.shape[0], x0.shape[1]
        x0 = x0.view(-1, *x0.shape[2:])
        latents = (self.vae.encode(x0).latent_dist.sample()).to(source_dtype)
        return latents.view(batch_size, frame_size, *latents.shape[1:])

    def decode(self, z0):
        # z0: (batch_size, frame_size, C, H, W)
        source_dtype = z0.dtype
        z0 = z0.to(self.dtype)
        batch_size, frame_size = z0.shape[0], z0.shape[1]
        z0 = z0.view(-1, *z0.shape[2:])
        x0 = self.vae.decode(z0).sample.to(source_dtype)
        return x0.view(batch_size, frame_size, *x0.shape[1:])

    def tweedie(self, z, sigma, c=None):
        # dtype: torch.float32
        source_dtype = z.dtype
        latent = z.to(self.dtype)

        # compute correct sigma
        sigma = sigma.to(self.dtype).view(-1, *([1] * len(latent.shape[1:])))

        # pre conditioning
        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = (self.scheduler.num_steps - 1) * self.scheduler.get_sigma_inv(sigma)

        # get tweedie
        latent_model_input = latent * c_in
        t_input = c_noise.flatten()
        image_only_indicator = torch.zeros(latent_model_input.shape[0:2], dtype=torch.bool, device=latent_model_input.device)
        noise_pred = self.vdm(latent_model_input, t_input, image_only_indicator=image_only_indicator).sample

        denoised = c_skip * z + c_out * noise_pred.to(source_dtype)
        return denoised

