from training.train_vae import train_vae
from training.train_idm import train_idm
from training.train_vdm import train_vdm
from training.train_pidm import train_pidm
from training.train_pvdm import train_pvdm
from training.debug import train_vae as train_debug
from pathlib import Path
import hydra
import os


@hydra.main(version_base="1.3", config_path="configs/training")
def main(cfg):
    vae_epoch = cfg.vae.training.num_epochs - 1
    idm_epoch = cfg.diffusion.pre_training.num_epochs - 1 
    vdm_epoch = cfg.diffusion.joint_training.num_epochs - 1 
    # check if the model checkpoint is available
    target_dir = Path(cfg.target_dir)
    model_dir = target_dir / 'models'

    if not os.path.exists(model_dir / f"ema_vae_{vae_epoch:04d}.pth"):
        train_vae(cfg) 
    else:
        print(f"Model checkpoint for VAE at epoch {vae_epoch} already exists")
        print('Skip VAE training ...')
    if not os.path.exists(model_dir / f"ema_idm_{idm_epoch:04d}.pth"):
        train_idm(cfg)
    else:
        print(f"Model checkpoint for IDM at epoch {idm_epoch} already exists")
        print('Skip IDM training ...')
    # if not os.path.exists(model_dir / f"ema_vdm_{vdm_epoch:04d}.pth"):
    #     train_vdm(cfg)
    # else:
    #     print(f"Model checkpoint for VDM at epoch {vdm_epoch} already exists")
    #     print('Skip VDM training ...')
    print('Training completed!')

if __name__ == "__main__":
    # main()
    from omegaconf import OmegaConf
    # cfg = OmegaConf.load('/scratch/imaging/projects/bingliang/tsinv/VDMPS/configs/training/mri.yaml')
    cfg = OmegaConf.load('/scratch/imaging/projects/bingliang/tsinv/VDMPS/configs/training/blackhole.yaml')
    train_vdm(cfg)