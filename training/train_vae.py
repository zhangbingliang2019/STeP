import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL
from accelerate import Accelerator
from tqdm.auto import tqdm
from torch.optim.swa_utils import AveragedModel
from utils.visualize import visualize_grid_bh, safe_dir, visualize_grid_mri
from utils.helper import print_model_size
from data import get_dataset
import wandb
import hydra
from omegaconf import DictConfig
from diffusers.models.attention_processor import AttnProcessor2_0

def get_main_module(model):
    if hasattr(model, 'module'):
        return model.module
    return model

def train_vae(cfg):
    dataset_cfg = cfg.dataset.image
    vae_cfg = cfg.vae
    wandb_cfg = cfg.wandb

    accelerator = Accelerator()
    dtype = torch.float32
    device = accelerator.device
    if accelerator.is_main_process and wandb_cfg.is_wandb:
        wandb.init(project=wandb_cfg.project, name=wandb_cfg.name)
    
    dataset = get_dataset(**dataset_cfg)
    dataset.summary()
    training_cfg = vae_cfg.training
    dataloader = DataLoader(dataset, batch_size=training_cfg.batch_size, shuffle=True, num_workers=training_cfg.num_workers, pin_memory=True, persistent_workers=True)

    target_dir = safe_dir(cfg.target_dir)
    image_dir = safe_dir(target_dir / 'vae_images')
    model_dir = safe_dir(target_dir / 'models')

    model_cfg = vae_cfg.model
    vae = AutoencoderKL(
        in_channels=model_cfg.in_channels,
        out_channels=model_cfg.out_channels,
        down_block_types=model_cfg.down_block_types,
        up_block_types=model_cfg.up_block_types,
        block_out_channels=model_cfg.block_out_channels,
        norm_num_groups=model_cfg.norm_num_groups,
        latent_channels=model_cfg.latent_channels,
    )
    vae = vae.to(dtype)
    vae.set_attn_processor(AttnProcessor2_0())
    ema_vae = AveragedModel(vae, avg_fn=lambda avg_p, model_p, num_avg: (1 - training_cfg.ema_decay) * model_p + training_cfg.ema_decay * avg_p)
    print_model_size(vae)

    optimizer = torch.optim.Adam(vae.parameters(), lr=training_cfg.lr)
    criterion = nn.L1Loss()
    vae, optimizer, dataloader = accelerator.prepare(vae, optimizer, dataloader)

    print('+=========================================+')
    print('|  Start training Variational Autoencoder  |')
    print('+=========================================+')
    vae.train()
    for epoch in range(training_cfg.num_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{training_cfg.num_epochs}")
        for images in progress_bar:
            images = images.to(dtype)
            
            latents_dist = get_main_module(vae).encode(images).latent_dist
            latents = latents_dist.sample()
            kl_loss = latents_dist.kl().mean()
            recon_images = get_main_module(vae).decode(latents).sample
            loss = criterion(recon_images, images) + kl_loss * training_cfg.beta

            optimizer.zero_grad()
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(vae.parameters(), 1.0)
            accelerator.backward(loss)
            optimizer.step()

            # Update EMA
            ema_vae.update_parameters(vae)

            progress_bar.set_postfix({"loss": loss.item(), "kl_loss": kl_loss.item()})
            if accelerator.is_main_process and wandb_cfg.is_wandb:
                wandb.log({"loss": loss.item(), "kl_loss": kl_loss.item()})

        # Visualization and Checkpointing
        if accelerator.is_main_process and (epoch + 1) % training_cfg.save_interval == 0:
            accelerator.save(accelerator.unwrap_model(vae), model_dir / f"vae_{epoch:04d}.pth")
            accelerator.save(ema_vae.module, model_dir / f"ema_vae_{epoch:04d}.pth")
