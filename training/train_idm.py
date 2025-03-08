import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from accelerate import Accelerator
from tqdm.auto import tqdm
from torch.optim.swa_utils import AveragedModel
from utils.visualize import visualize_grid_bh, save_video_bh, save_video_bh_gif, safe_dir, save_grid_mri, visualize_grid_mri
from utils.helper import print_model_size
from data import get_dataset
from model.st_model import UNetSpatioTemporalDiffusion
from model import SpatialTemporalDiffusionModel
from cores.scheduler import EDMScheduler, DiffusionPFODE
import wandb
import hydra
from omegaconf import DictConfig
from diffusers.models.attention_processor import AttnProcessor2_0



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



def train_idm(cfg):
    dataset_cfg = cfg.dataset.image
    diffusion_cfg = cfg.diffusion
    wandb_cfg = cfg.wandb
    training_cfg = diffusion_cfg.pre_training

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
    
    dataset = get_dataset(**dataset_cfg)
    dataset.summary()
    
    dataloader = DataLoader(dataset, batch_size=batch_size * frame_size, shuffle=True, num_workers=training_cfg.num_workers, pin_memory=True, drop_last=True)

    target_dir = safe_dir(cfg.target_dir)
    image_dir = safe_dir(target_dir / 'idm_images')
    model_dir = safe_dir(target_dir / 'models')

    model_cfg = diffusion_cfg.model
    idm = UNetSpatioTemporalDiffusion(
        in_channels=model_cfg.in_channels,
        out_channels=model_cfg.out_channels,
        down_block_types=model_cfg.down_block_types,
        up_block_types=model_cfg.up_block_types,
        block_out_channels=model_cfg.block_out_channels,
        num_attention_heads=model_cfg.num_attention_heads,
        layers_per_block=model_cfg.layers_per_block,
        projection_class_embeddings_input_dim=256,
    )
    idm = idm.to(dtype)
    idm.set_attn_processor(AttnProcessor2_0())
    epoch = max([int(file.stem.split('_')[-1]) for file in model_dir.glob("ema_vae_*.pth")])
    vae = torch.load(model_dir / f"ema_vae_{epoch:04d}.pth")
    vae = vae.to(dtype)
    vae.set_attn_processor(AttnProcessor2_0())
    print('Loading VAE model from epoch', epoch)
    # freeze vae parameters
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
    
    idm, optimizer, dataloader = accelerator.prepare(idm, optimizer, dataloader)
    vae = vae.to(device)

    print('+=========================================+')
    print('|  Start training Image Diffusion Model  |')
    print('+=========================================+')
    idm.train()
    for epoch in range(training_cfg.num_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{training_cfg.num_epochs}")
        for images in progress_bar:
            images = images.to(dtype)
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample().detach()
                latents = latents.view(batch_size, frame_size, latent_channels, latent_size, latent_size)
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device, dtype=torch.int64)
            weights = get_loss_weight(timesteps, noise_scheduler.config.num_train_timesteps, weight_type='none')
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            image_only_indicator = torch.ones(noisy_latents.shape[0:2], dtype=torch.bool, device=noisy_latents.device)

            noise_pred = idm(noisy_latents, timesteps, image_only_indicator=image_only_indicator).sample
            loss = nn.functional.mse_loss(noise_pred, noise, reduction='none').flatten(1).mean(-1)
            loss = (loss * weights).mean()

            optimizer.zero_grad()
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(idm.parameters(), 1.0)
            optimizer.step()

            # Update EMA
            ema_idm.update_parameters(idm)

            progress_bar.set_postfix({"loss": loss.item()})
            if accelerator.is_main_process and wandb_cfg.is_wandb:
                wandb.log({"loss": loss.item()})

        # Visualization and Checkpointing
        if accelerator.is_main_process and (epoch + 1) % training_cfg.log_interval == 0:
            sampling_model = SpatialTemporalDiffusionModel(
                accelerator.unwrap_model(idm), vae, noise_scheduler,
                video_shape=(frame_size, image_channels, image_size, image_size), latent_shape=(frame_size, latent_channels, latent_size, latent_size)
            )
            sampling_scheduler = EDMScheduler(100)
            sampler = DiffusionPFODE(sampling_model, sampling_scheduler, 'euler')

            with torch.no_grad():
                zT = sampler.get_start(torch.zeros(1, *sampling_model.get_latent_shape()).cuda())
                latents = sampler.sample(zT)
                uncond_samples = sampling_model.decode(latents)[0]

                # visualize_grid_bh(uncond_samples, target=str(image_dir / "{:04d}.png".format(epoch)), path=safe_dir(str(image_dir / 'tmp')), nrow=8)
                # save_video_bh(uncond_samples, target=str(image_dir / "{:04d}.mp4".format(epoch)), path=safe_dir(str(image_dir / 'tmp')))
                # save_video_bh_gif(uncond_samples, target=str(image_dir / "{:04d}.gif".format(epoch)), path=safe_dir(str(image_dir / 'tmp')))

                visualize_grid_mri(uncond_samples, target=str(image_dir / "{:04d}.png".format(epoch)), nrow=6, image_type=cfg.dataset.image.image_type)
                # save_grid_mri(uncond_samples, target=str(image_dir / "{:04d}_grid.png".format(epoch)), nrow=8)
          
   
            if (epoch + 1) % training_cfg.save_interval == 0:
                accelerator.save(accelerator.unwrap_model(idm), model_dir / f"idm_{epoch:04d}.pth")
                accelerator.save(ema_idm.module, model_dir / f"ema_idm_{epoch:04d}.pth")
                accelerator.save(noise_scheduler, model_dir / f"scheduler.pth")
