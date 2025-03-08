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


class DiffusionModel(nn.Module):
    """
    A class representing a diffusion model.
    Methods:
        score(x, sigma): Calculates the score of the latent diffusion model given the latent variable `x` and standard deviation `sigma`.
        tweedie(x, sigma): Calculates the Tweedie distribution given the latent variable `x` and standard deviation `sigma`.
        Must overload either `score` or `tweedie` method.
    """

    def __init__(self):
        super(DiffusionModel, self).__init__()
        # Check if either `score` or `tweedie` is overridden
        if (self.score.__func__ is DiffusionModel.score and
                self.tweedie.__func__ is DiffusionModel.tweedie):
            raise NotImplementedError(
                "Either `score` or `tweedie` method must be implemented."
            )

    def score(self, x, sigma, c=None):
        d = self.tweedie(x, sigma=sigma, c=c)
        return (d - x) / sigma ** 2

    def tweedie(self, x, sigma, c=None):
        # assert exactly one of t and sigma is not None
        return x + self.score(x, sigma=sigma, c=c) * sigma ** 2

    def get_data_shape():
        pass



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


@register_model(name='sdm')
class StableDiffusionModel(LatentDiffusionModel):
    def __init__(self, model_id = "stabilityai/stable-diffusion-2-1", inner_resolution=768, target_resolution=256, frames=16, guidance_scale=7.5, prompt='a natural looking human face', device='cuda'):
        super().__init__()
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        self.pipe = pipe.to(device)
        self.vae = self.pipe.vae
        self.guidance_scale = guidance_scale
        self.prompt = prompt
        self.device = device
        self.unet = self.pipe.unet
        self.latent_scale = self.pipe.vae.config.scaling_factor
        self.dtype = torch.float16
        self.resolution = inner_resolution
        self.target_resolution = target_resolution
        self.frames = frames
        # scheduling
        scheduler = pipe.scheduler
        self.scheduler = VPScheduler(
            num_steps=scheduler.config.num_train_timesteps,
            beta_max=scheduler.config.beta_end * scheduler.config.num_train_timesteps,
            beta_min=scheduler.config.beta_start * scheduler.config.num_train_timesteps,
            epsilon=0,
            beta_type=scheduler.config.beta_schedule,
        )
        self.unet.requires_grad_(False)
    
    def get_data_shape(self):
        return (self.frames, 3, self.target_resolution, self.target_resolution)
    
    def get_latent_shape(self):
        num_channels_latents = self.pipe.unet.config.in_channels
        latents = self.pipe.prepare_latents(
            1, num_channels_latents, self.resolution, self.resolution, self.dtype, self.device, None, None
        )
        return self.frames, *latents.shape[1:]
    
    def encode(self, x0):
        source_dtype = x0.dtype
        x0 = x0.to(self.dtype)
        batch_size, frame_size = x0.shape[0], x0.shape[1]
        x0 = x0.view(-1, *x0.shape[2:])
        x0 = F.interpolate(x0, size=self.resolution, mode='bilinear')
        latents = (self.vae.encode(x0).latent_dist.sample()*self.latent_scale).to(source_dtype)
        return latents.view(batch_size, frame_size, *latents.shape[1:])
    
    def decode(self, z0):
        source_dtype = z0.dtype
        z0 = z0.to(self.dtype)
        batch_size, frame_size = z0.shape[0], z0.shape[1]
        z0 = z0.view(-1, *z0.shape[2:])

        x0 = self.vae.decode(z0/self.latent_scale).sample.to(source_dtype)
        x0 = F.interpolate(x0, size=self.target_resolution, mode='bilinear')
        return x0.view(batch_size, frame_size, *x0.shape[1:])
    
    def tweedie(self, z, sigma, c=None):
        if c is None:
            c = self.prompt
        # dtype: torch.float32
        source_dtype = z.dtype
        B, N = z.shape[0], z.shape[1]
        z = z.flatten(0, 1)
        latent = z.to(self.dtype)

        # compute correct sigma
        sigma = sigma.to(self.dtype).view(-1, *([1] * len(latent.shape[1:])))

        # pre conditioning
        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = (self.scheduler.num_steps - 1) * self.scheduler.get_sigma_inv(sigma)

        # get tweedie
        # 1. encode prompt
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt=c, 
            device=z.device, 
            num_images_per_prompt=1, 
            do_classifier_free_guidance=True, 
        )
        prompt_embeds = torch.cat([negative_prompt_embeds] * z.shape[0]+ [prompt_embeds] * z.shape[0], dim=0)

        # 2. get unet output
        latent_model_input = torch.cat([latent] * 2) * c_in
        t_input = c_noise.flatten()
        noise_pred = self.unet(latent_model_input, t_input, encoder_hidden_states=prompt_embeds, return_dict=False)[0]

        # 3. classifier free guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)        
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

        denoised = c_skip * z + c_out * noise_pred.to(source_dtype)
        return denoised.view(B, N, *denoised.shape[1:])


@register_model(name='video_sdm')
class VideoStableDiffusionModel(LatentDiffusionModel):
    def __init__(self, model_id = "ali-vilab/text-to-video-ms-1.7b", inner_resolution=256, target_resolution=256, frames=16, guidance_scale=7.5, prompt='a natural looking human face', image_only=False, device='cuda'):
        super().__init__()
        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
        self.pipe = pipe.to(device)
        self.vae = self.pipe.vae
        self.guidance_scale = guidance_scale
        self.prompt = prompt
        self.device = device
        self.unet = self.pipe.unet
        self.latent_scale = self.pipe.vae.config.scaling_factor
        self.dtype = torch.float16
        self.resolution = inner_resolution
        self.target_resolution = target_resolution
        self.frames = frames
        self.image_only = image_only
        # scheduling
        scheduler = pipe.scheduler
        self.scheduler = VPScheduler(
            num_steps=scheduler.config.num_train_timesteps,
            beta_max=scheduler.config.beta_end * scheduler.config.num_train_timesteps,
            beta_min=scheduler.config.beta_start * scheduler.config.num_train_timesteps,
            epsilon=0,
            beta_type=scheduler.config.beta_schedule,
        )
        self.unet.requires_grad_(False)
    
    def get_data_shape(self):
        return (self.frames, 3, self.target_resolution, self.target_resolution)
    
    def get_latent_shape(self):
        num_channels_latents = self.pipe.unet.config.in_channels
        latents = self.pipe.prepare_latents(
            1, num_channels_latents, self.frames, self.resolution, self.resolution, self.dtype, self.device, None, None
        )
        return self.frames, latents.shape[1], *latents.shape[3:]

    def encode(self, x0):
        source_dtype = x0.dtype
        x0 = x0.to(self.dtype)
        batch_size, frame_size = x0.shape[0], x0.shape[1]
        x0 = x0.view(-1, *x0.shape[2:])
        x0 = F.interpolate(x0, size=self.resolution, mode='bilinear')
        latents = (self.vae.encode(x0).latent_dist.sample()*self.latent_scale).to(source_dtype)
        return latents.view(batch_size, frame_size, *latents.shape[1:])
    
    def decode(self, z0):
        source_dtype = z0.dtype
        z0 = z0.to(self.dtype)
        batch_size, frame_size = z0.shape[0], z0.shape[1]
        z0 = z0.view(-1, *z0.shape[2:])

        x0 = self.vae.decode(z0/self.latent_scale).sample.to(source_dtype)
        x0 = F.interpolate(x0, size=self.target_resolution, mode='bilinear')
        return x0.view(batch_size, frame_size, *x0.shape[1:])

    def tweedie(self, z, sigma, c=None):
        if c is None:
            c = self.prompt
        # dtype: torch.float32
        B, N = z.shape[0], z.shape[1]
        source_dtype = z.dtype # z: [B, T, C, H, W]
        latent = z.to(self.dtype)

        # compute correct sigma
        sigma = sigma.to(self.dtype).view(-1, *([1] * len(latent.shape[1:])))

        # pre conditioning
        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = (self.scheduler.num_steps - 1) * self.scheduler.get_sigma_inv(sigma)

        # get tweedie
        # 1. encode prompt
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt=c, 
            device=z.device, 
            num_images_per_prompt=1, 
            do_classifier_free_guidance=True, 
        )
        prompt_embeds = torch.cat([negative_prompt_embeds] * z.shape[0]+ [prompt_embeds] * z.shape[0], dim=0)

        # 2. get unet output
        latent_model_input = torch.cat([latent] * 2) * c_in
        t_input = c_noise.flatten()

        if self.image_only:
            latent_model_input = latent_model_input.view(B * N * 2, 1, *latent_model_input.shape[2:])
        noise_pred = self.unet(latent_model_input.permute(0, 2, 1, 3, 4), t_input, encoder_hidden_states=prompt_embeds, return_dict=False)[0]
        noise_pred = noise_pred.permute(0, 2, 1, 3, 4)
        if self.image_only:
            noise_pred = noise_pred.view(B * 2, N, *noise_pred.shape[2:])

        # 3. classifier free guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)        
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

        denoised = c_skip * z + c_out * noise_pred.to(source_dtype)
        return denoised


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


@register_model(name='st_diffusion_pixel')
class SpatialTemporalPixelDiffusionModel(DiffusionModel):
    def __init__(self, vdm_or_path, scheduler_or_path, video_shape=(64, 1, 256, 256), device='cuda'):
        super().__init__()
        self.device = device
        if isinstance(vdm_or_path, str):
            self.vdm = torch.load(vdm_or_path).to(device)
        else:
            self.vdm = vdm_or_path.to(device)
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
        self.dtype = torch.float32
    
    def get_data_shape(self):
        return self.video_shape

    def tweedie(self, x, sigma, c=None):
        # dtype: torch.float32
        source_dtype = x.dtype
        latent = x.to(self.dtype)

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

        denoised = c_skip * x + c_out * noise_pred.to(source_dtype)
        return denoised

