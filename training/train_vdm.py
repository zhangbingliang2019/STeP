import torch
from torch import nn
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from accelerate import Accelerator
from tqdm.auto import tqdm
from torch.optim.swa_utils import AveragedModel
from utils.visualize import visualize_grid_bh, save_video_bh, save_video_bh_gif, safe_dir, visualize_grid_mri, save_video_mri
from utils.helper import print_model_size
from data import get_dataset
from model import SpatialTemporalDiffusionModel
from cores.scheduler import VPScheduler, DiffusionPFODE
import numpy as np
import wandb
import hydra
from diffusers.models.attention_processor import AttnProcessor2_0
import itertools


def train_vdm(cfg):
    image_dataset_cfg = cfg.dataset.image
    video_dataset_cfg = cfg.dataset.video
    diffusion_cfg = cfg.diffusion
    wandb_cfg = cfg.wandb
    training_cfg = diffusion_cfg.joint_training

    batch_size = training_cfg.batch_size
    frame_size, image_size = cfg.dataset.video.frames, cfg.dataset.video.resolution
    latent_size = image_size // 2 ** (len(cfg.vae.model.down_block_types) - 1)
    image_channels = cfg.vae.model.out_channels
    latent_channels = cfg.vae.model.latent_channels

    accelerator = Accelerator()
    dtype = torch.float32
    device = accelerator.device
    if accelerator.is_main_process and wandb_cfg.is_wandb:
        wandb.init(project=wandb_cfg.project, name=wandb_cfg.name)
    

    image_dataset = get_dataset(**image_dataset_cfg)
    video_dataset = get_dataset(**video_dataset_cfg)
    image_dataset.summary()
    video_dataset.summary()
    
    image_dataloader = DataLoader(image_dataset, batch_size=batch_size * frame_size, shuffle=True, num_workers=training_cfg.num_workers, pin_memory=True, drop_last=True)
    video_dataloader = DataLoader(video_dataset, batch_size=batch_size, shuffle=True, num_workers=training_cfg.num_workers, pin_memory=True, drop_last=True)

    target_dir = safe_dir(cfg.target_dir)
    image_dir = safe_dir(target_dir / 'vdm_images')
    model_dir = safe_dir(target_dir / 'models')

    epoch = max([int(file.stem.split('_')[-1]) for file in model_dir.glob("ema_idm_*.pth")])
    idm = torch.load(model_dir / f"ema_idm_{epoch:04d}.pth")
    idm.set_attn_processor(AttnProcessor2_0())
    idm = idm.to(dtype)
    print('Loading IDM model from epoch', epoch)
    
    epoch = max([int(file.stem.split('_')[-1]) for file in model_dir.glob("ema_vae_*.pth")])
    vae = torch.load(model_dir / f"ema_vae_{epoch:04d}.pth")
    vae = vae.to(dtype)
    print('Loading VAE model from epoch', epoch)
    # freeze vae parameters
    for param in idm.parameters():
        param.requires_grad = True
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

    optimizer = torch.optim.Adam(idm.parameters(), lr=training_cfg.lr)
    
    idm, vae, optimizer, image_dataloader, video_dataloader = accelerator.prepare(idm, vae, optimizer, image_dataloader, video_dataloader)
    image_dataloader_iter = itertools.cycle(image_dataloader)

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
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device, dtype=torch.int64)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            image_only_indicator = torch.zeros(noisy_latents.shape[0:2], dtype=torch.bool, device=noisy_latents.device)

            # print(noisy_latents.shape, timesteps.shape, image_only_indicator.shape)
            noise_pred = idm(noisy_latents, timesteps, image_only_indicator=image_only_indicator).sample
            loss = nn.functional.mse_loss(noise_pred, noise, reduction='none').flatten(1).mean()

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            
            # Update EMA
            ema_idm.update_parameters(idm)

            progress_bar.set_postfix({"Video loss": loss.item()})
            if accelerator.is_main_process and wandb_cfg.is_wandb:
                wandb.log({"Video loss": loss.item()})

            # image-video joint training
            if np.random.rand() < training_cfg.joint_ratio:
                images = next(image_dataloader_iter).to(dtype)
                with torch.no_grad():
                    latents = vae.encode(images).latent_dist.sample().detach()
                    latents = latents.view(batch_size, frame_size, latent_channels, latent_size, latent_size)
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device, dtype=torch.int64)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                image_only_indicator = torch.ones(noisy_latents.shape[0:2], dtype=torch.bool, device=noisy_latents.device)

                # print(noisy_latents.shape, timesteps.shape, image_only_indicator.shape)
                noise_pred = idm(noisy_latents, timesteps, image_only_indicator=image_only_indicator).sample
                loss = nn.functional.mse_loss(noise_pred, noise, reduction='none').flatten(1).mean()

                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                
                # Update EMA
                ema_idm.update_parameters(idm)

                progress_bar.set_postfix({"Image loss": loss.item()})
                if accelerator.is_main_process and wandb_cfg.is_wandb:
                    wandb.log({"Image loss": loss.item()})

        # Visualization and Checkpointing
        if accelerator.is_main_process and (epoch + 1) % training_cfg.save_interval == 0:
            accelerator.save(accelerator.unwrap_model(idm), model_dir / f"vdm_{epoch:04d}.pth")
            accelerator.save(ema_idm.module, model_dir / f"ema_vdm_{epoch:04d}.pth")
