import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from accelerate import Accelerator
from tqdm.auto import tqdm
from torch.optim.swa_utils import AveragedModel
from utils.visualize import visualize_grid_bh, save_video_bh, save_video_bh_gif, safe_dir
from utils.helper import print_model_size
from data import get_dataset
from model.st_model import UNetSpatioTemporalDiffusion
from model import SpatialTemporalDiffusionModel
from cores.scheduler import VPScheduler, DiffusionPFODE
import numpy as np
import wandb
import hydra
from omegaconf import DictConfig
import itertools


def set_temporal_trainable(model):
    # Iterate through all named parameters in the model
    for name, param in model.named_parameters():
        # If 'temporal' is in the parameter name, set it to be trainable
        if "temporal" in name:
            param.requires_grad = True
        else:
            # Otherwise, freeze the parameter
            param.requires_grad = False


def get_loss_weight(timesteps, num_train_timesteps, weight_type='cosine'):
    # Normalize timesteps to range [0, 1]
    t_normalized = timesteps / num_train_timesteps
    
    if weight_type == 'cosine':
        # Cosine weighting as suggested by Nichol & Dhariwal (2021)
        weights = torch.cos(t_normalized * torch.pi / 2) ** 2
    elif weight_type == 'sqrt':
        # Square root weighting
        weights = torch.sqrt(1 - t_normalized ** 2)
    else:
        # Default to no weighting
        weights = torch.ones_like(timesteps, dtype=torch.float32, device=timesteps.device)
    
    return weights



def ft_vdm(cfg):
    video_dataset_cfg = cfg.dataset.video
    diffusion_cfg = cfg.diffusion
    wandb_cfg = cfg.wandb
    training_cfg = diffusion_cfg.finetuning

    batch_size = training_cfg.batch_size
    frame_size, image_size = cfg.dataset.video.frames, cfg.dataset.video.resolution
    latent_size = image_size // 2 ** (len(cfg.vae.model.down_block_types) - 1)
    image_channels = cfg.vae.model.out_channels
    latent_channels = cfg.vae.model.latent_channels

    accelerator = Accelerator()
    dtype = torch.float32
    # accelerator = Accelerator(mixed_precision='bf16')
    # dtype = torch.bfloat16
    device = accelerator.device
    if accelerator.is_main_process and wandb_cfg.is_wandb:
        wandb.init(project=wandb_cfg.project, name=wandb_cfg.name)
    

    video_dataset = get_dataset(**video_dataset_cfg)
    video_dataset.summary()
    
    video_dataloader = DataLoader(video_dataset, batch_size=batch_size, shuffle=True, num_workers=training_cfg.num_workers, pin_memory=True, drop_last=True)

    target_dir = safe_dir(cfg.target_dir)
    image_dir = safe_dir(target_dir / 'fvdm_images')
    model_dir = safe_dir(target_dir / 'models')

    # epoch = max([int(file.stem.split('_')[-1]) for file in model_dir.glob("idm_*.pth")])
    epoch = 199
    idm = torch.load(model_dir / f"idm_{epoch:04d}.pth")
    # idm = idm.to(dtype)
    print('Loading IDM model from epoch', epoch)
    # idm = torch.load('/scratch/imaging/projects/bingliang/tsinv/VDMPS/checkpoints/blackhole/idm.pth')
    idm = idm.to(dtype)
    
    # epoch = max([int(file.stem.split('_')[-1]) for file in model_dir.glob("vae_*.pth")])
    epoch = 24
    vae = torch.load(model_dir / f"vae_{epoch:04d}.pth")
    print('Loading VAE model from epoch', epoch)
    # vae = torch.load('/scratch/imaging/projects/bingliang/tsinv/VDMPS/checkpoints/blackhole/vae.pth')
    vae = vae.to(dtype)
    
    # freeze vae parameters
    set_temporal_trainable(idm)
    for param in vae.parameters():
        param.requires_grad = False
    ema_idm = AveragedModel(idm, avg_fn=lambda avg_p, model_p, num_avg: (1 - training_cfg.ema_decay) * model_p + training_cfg.ema_decay * avg_p)
    print_model_size(idm)

    scheduler_cfg = diffusion_cfg.scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=scheduler_cfg.num_train_timesteps, 
        beta_start=scheduler_cfg.beta_start, 
        beta_end=scheduler_cfg.beta_end, 
        beta_schedule=scheduler_cfg.beta_schedule,
        clip_sample=scheduler_cfg.clip_sample,
    )
    trainable_params = [param for param in idm.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=training_cfg.lr)
    
    idm, vae, optimizer, video_dataloader = accelerator.prepare(idm, vae, optimizer, video_dataloader)

    print('+=========================================+')
    print('|  Start training Video Diffusion Model  |')
    print('+=========================================+')
    idm.train()
    for epoch in range(training_cfg.num_epochs):
        progress_bar = tqdm(video_dataloader, desc=f"Epoch {epoch+1}/{training_cfg.num_epochs}")
        for video in progress_bar:
            video = video.to(dtype)
            with torch.no_grad():
                video = video.view(-1, image_channels, image_size, image_size)
                latents = vae.encode(video).latent_dist.sample().detach()
                latents = latents.view(batch_size, frame_size, latent_channels, latent_size, latent_size)
            # batch consistent noise
            noise = torch.randn_like(latents[:, 0:1]).expand_as(latents)
            # normal noise
            # noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device, dtype=torch.int64)
            weights = get_loss_weight(timesteps, noise_scheduler.config.num_train_timesteps, weight_type='cosine')
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            image_only_indicator = torch.zeros(noisy_latents.shape[0:2], dtype=torch.bool, device=noisy_latents.device)

            # print(noisy_latents.shape, timesteps.shape, image_only_indicator.shape)
            noise_pred = idm(noisy_latents, timesteps, image_only_indicator=image_only_indicator).sample
            loss = nn.functional.mse_loss(noise_pred, noise, reduction='none').flatten(1).mean(-1)
            loss = (loss * weights).mean()

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            
            # Update EMA
            ema_idm.update_parameters(idm)

            progress_bar.set_postfix({"Video loss": loss.item()})
            if accelerator.is_main_process and wandb_cfg.is_wandb:
                wandb.log({"Video loss": loss.item()})

        # Visualization and Checkpointing
        if accelerator.is_main_process and (epoch + 1) % training_cfg.log_interval == 0:
            sampling_model = SpatialTemporalDiffusionModel(
                accelerator.unwrap_model(idm), accelerator.unwrap_model(vae), noise_scheduler,
                video_shape=(frame_size, image_channels, image_size, image_size), latent_shape=(frame_size, latent_channels, latent_size, latent_size)
            )
            sampling_scheduler = VPScheduler(100, scheduler_cfg.beta_end * scheduler_cfg.num_train_timesteps, scheduler_cfg.beta_start * scheduler_cfg.num_train_timesteps, 0, scheduler_cfg.beta_schedule)
            sampler = DiffusionPFODE(sampling_model, sampling_scheduler, 'euler')

            with torch.no_grad():
                noise = torch.randn(1, 1, *sampling_model.get_latent_shape()[1:]).cuda().expand_as(noise)
                zT = noise * sampling_scheduler.get_prior_sigma()
                # zT = sampler.get_start(torch.zeros(1, *sampling_model.get_latent_shape()).cuda())
                latents = sampler.sample(zT)
                uncond_samples = sampling_model.decode(latents)[0]

                visualize_grid_bh(uncond_samples, target=str(image_dir / "{:04d}.png".format(epoch)), path=safe_dir(str(image_dir / 'tmp')), nrow=8)
                save_video_bh(uncond_samples, target=str(image_dir / "{:04d}.mp4".format(epoch)), path=safe_dir(str(image_dir / 'tmp')))
                save_video_bh_gif(uncond_samples, target=str(image_dir / "{:04d}.gif".format(epoch)), path=safe_dir(str(image_dir / 'tmp')))
          
   
            if (epoch + 1) % training_cfg.save_interval == 0:
                accelerator.save(accelerator.unwrap_model(idm), model_dir / f"fvdm_{epoch:04d}.pth")
                accelerator.save(ema_idm.module, model_dir / f"ema_fvdm_{epoch:04d}.pth")

if __name__ == "__main__":
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("/scratch/imaging/projects/bingliang/tsinv/VDMPS/configs/blackhole.yaml")
    ft_vdm(cfg)
