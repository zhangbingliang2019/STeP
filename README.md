# STeP: a framework for solving scientific video inverse problems with spatiotemporal diffusion priors

### 📝 [Paper](https://arxiv.org/abs/2504.07549) | 🌐 [Project Page](https://zhangbingliang2019.github.io/STeP/)

![teaser](README.assets/teaser.png)

## 🔥 Introduction

We demonstrate the feasibility of practical and accessible **spatiotemporal diffusion priors** by fine-tuning **latent video diffusion models** from pretrained image diffusion models using limited videos in specific domains. Leveraging this plug-and-play spatiotemporal diffusion prior, we introduce a general and scalable framework for solving video inverse problems. Our framework enables the generation of diverse, high-fidelity video reconstructions that not only fit observations but also recover multi-modal solutions. 



## 🚀 Get Started 

### 1. Prepare the environment

- python 3.8
- PyTorch 2.3
- CUDA 12.1

```
# in STeP folder

conda create -n STeP python=3.8
conda activate STeP

conda install pytorch==2.3.0 torchvision==0.18.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge pynfft

pip install -r requirements.txt
```



### 2. Download the test dataset

Run the following commands to download our test dataset:

```
# in STeP folder

gdown https://drive.google.com/uc?id=1un72wQb24yhv64S6anz7yCf81l51B9Lh -O dataset.zip
unzip dataset.zip
rm dataset.zip
```

**Notice**: the training and testing dataset of GRMHD simulated blackhole datasets are private, thus we provide 20 generated blackhole video samples as testing data.



### 3. Solving Inverse Problem with Diffusion Posterior Sampling

Task specific configs are in folder `configs/task`. 

* specify the VAE decoder and video diffusion model location
* specify the sampling hyper parameter (e.g. number MCMC steps)

By default, we will automatically download and use our pretrained models for [blackhole](https://huggingface.co/bingliangzhang00/STeP-blackhole) and [MRI](https://huggingface.co/bingliangzhang00/STeP-mri) task.



**Sample Command:**

```
python sample.py +task=[task-name] gpu=[gpu] task.target_dir=[save_dir] 
```

example commands:

```
python sample.py +task=blackhole gpu=0 task.target_dir=exps/blackhole_imaging num_samples=10
python sample.py +task=mri gpu=0 task.target_dir=exps/dynamic_mri num_samples=1
```

It will generate `num_samples` in `task.target_dir`.



### 4. Training Video Diffusion Model Prior

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



## Citation

If you find our work interesting, please consider citing

```
@misc{zhang2025stepgeneralscalableframework,
      title={STeP: A General and Scalable Framework for Solving Video Inverse Problems with Spatiotemporal Diffusion Priors}, 
      author={Bingliang Zhang and Zihui Wu and Berthy T. Feng and Yang Song and Yisong Yue and Katherine L. Bouman},
      year={2025},
      eprint={2504.07549},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.07549}, 
}
```

