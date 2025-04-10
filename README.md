## Get Started

### 1. Prepare the Environment

- python 3.8
- PyTorch 2.3
- CUDA 12.1

```
# in DAPS folder
conda create -n STeP python=3.8
conda activate STeP

pip install -r requirements.txt

# (optional) install PyTorch with proper CUDA
conda install pytorch==2.3.0 torchvision==0.18.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```



#### 1. Training Video Diffusion Model Prior

Training configs are in folder `configs/training`.

* prepare the required image and video dataset. 
* specify the model architecture
* specify the training hyperparameters (e.g. epochs)



**Training Command:**

```
CUDA_VISIBLE_DEVICES=[index] python train.py --config-name [training-config-name]
```

example commands:

```
CUDA_VISIBLE_DEVICES=0 python train.py --config-name blackhole
CUDA_VISIBLE_DEVICES=0 python train.py --config-name mri
```

It will sequentially train:

1. an image VAE model 
2. an image diffusion model
3. a spatiotemporal video diffusion model



#### 2. Solving Inverse Problem with Diffusion Posterior Sampling

Task specific configs are in folder `configs/task` 

* specify the VAE decoder and video diffusion model location
* specify the sampling hyper parameter (e.g. number MCMC steps)



**Sample Command:**

```
python sample.py +task=[task-name] gpu=[gpu] task.target_dir=[save_dir] 
```

example commands:

```
python sample.py +task=blackhole gpu=0 task.target_dir=blackhole_imaging num_samples=10
python sample.py +task=mri gpu=0 task.target_dir=dynamic_mri num_samples=1
```

It will generate `num_samples` in `task.target_dir`.

