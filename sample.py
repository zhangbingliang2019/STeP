import torch
import numpy as np
from data import get_dataset
from forward_operator import get_operator
from model import get_model
from sampler import get_sampler
from evaluate import get_evaluator
from utils.visualize import save_video_bh, safe_dir, visualize_grid_bh
from piq import psnr
import hydra
import wandb
from omegaconf import OmegaConf

def unnormalize(x):
    return (x * 0.5 + 0.5).clip(0, 1)


@hydra.main(version_base="1.3", config_path="configs", config_name="default")
def main(cfg):
    task_cfg = cfg.task
    sampler_cfg = cfg.task.sampler
    dataset_cfg = cfg.task.dataset
    wandb_cfg = cfg.wandb

    # 0. set up device & directory
    gpu = cfg.gpu
    torch.cuda.set_device(gpu)
    torch.random.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    target_dir = safe_dir(task_cfg.target_dir)

    if wandb_cfg.is_wandb:
        wandb.init(
            project=wandb_cfg.project,
            name=wandb_cfg.name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        cfg = OmegaConf.create(dict(wandb.config)) 

    if isinstance(task_cfg.test_idx, int):
        test_idx = [task_cfg.test_idx]
    else:
        test_idx = task_cfg.test_idx
    OmegaConf.save(cfg, str(target_dir / 'cfg.yaml'))
    num_samples = cfg.num_samples

    # 1. load components
    video_dataset = get_dataset(**dataset_cfg)
    video_dataset.summary()
    op = get_operator(**task_cfg.forward_operator)
    model = get_model(**task_cfg.diffusion_wrapper)
    sampler = get_sampler(**sampler_cfg)
    sampler.summary()
    evaluator = get_evaluator(**task_cfg.evaluator, op=op)
    
    # 2. loop for posterior sampling
    wandb_metrics = []
    for idx in test_idx:
        videos = video_dataset[idx].unsqueeze(0).cuda()
        observations = op.measure(videos)
        idx_dir = safe_dir(target_dir / '{:06d}'.format(idx))
        torch.save(observations.to('cpu'), idx_dir / 'observations.pth')
    
        for i in range(num_samples):
            sample = sampler.sample(model, sampler.get_start(torch.randn(1, *model.get_latent_shape()).cuda()), op, observations, evaluator=evaluator, record=cfg.save_trajectory, verbose=True, gt=videos)
            torch.save(videos.to('cpu'), idx_dir / 'videos.pth')
            torch.save(sample.to('cpu'), idx_dir / f'sample_{i}.pth')
            metrics = evaluator(videos, observations, sample)

            if wandb_cfg.is_wandb:
                wandb.log({key: value.item() for key, value in metrics.items()})
                wandb_metrics.append(metrics[evaluator.main_eval_fn_name].item())
            
            if cfg.save_trajectory:
                traj = sampler.trajectory.compile()
                torch.save(traj, idx_dir / f'traj_{i}.pth')
    
    if wandb_cfg.is_wandb:
        wandb.log({evaluator.main_eval_fn_name+'-max': np.max(wandb_metrics)})
        wandb.log({evaluator.main_eval_fn_name+'-min': np.min(wandb_metrics)})
        wandb.log({evaluator.main_eval_fn_name+'-mean': np.mean(wandb_metrics)})
        wandb.log({evaluator.main_eval_fn_name+'-std': np.std(wandb_metrics)})
        wandb.finish()

if __name__ == '__main__':
    main()