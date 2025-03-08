import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image
from pathlib import Path
from PIL import Image
import tqdm
import matplotlib.pyplot as plt
import PIL
import imageio
import ehtplot.color
import numpy as np



def safe_dir(dir):
    if not Path(dir).exists():
        Path(dir).mkdir()
    return Path(dir)


def norm_image_01(x):
    return (x * 0.5 + 0.5).clip(0, 1)


def norm(x):
    return (x - x.min()) / (x.max() - x.min())


def visualize_pil(target, figsize=None):
    pil_image = PIL.Image.open(target)
    if figsize is None:
        plt.figure(figsize=(24, 24))
    else:
        plt.figure(figsize=figsize)
    if pil_image.mode == 'L':
        plt.imshow(pil_image, cmap='gray')
    else:
        plt.imshow(pil_image)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()


def visualize_grid(images, target='image.png', nrow=10, figsize=None, normalize=True):
    # Save images.
    save_grid(images, target, nrow=nrow, normalize=normalize)
    visualize_pil(target, figsize=figsize)


def visualize_grid_bh(images, target='image.png', path='black_image', nrow=10, figsize=None, normalize=True):
    save_grid_bh(images, target, path, nrow=nrow, normalize=normalize)
    visualize_pil(target, figsize=figsize)


def visualize_grid_mri(images, target='image.png', nrow=10, figsize=None, normalize=True, image_type='amplitude'):
    save_grid_mri(images, target, nrow=nrow, normalize=normalize, image_type=image_type)
    visualize_pil(target, figsize=figsize)


def save_grid(images, target='image.png', nrow=10, normalize=True):
    # Save images.
    if normalize:
        images = norm_image_01(images)
    save_image(images, target, nrow=nrow)


def save_grid_bh(images, target='image.png', path='black_image', nrow=10, normalize=True):
    if normalize:
        images = norm_image_01(images)
    path = safe_dir(path)
    # save to dir
    for i, image in enumerate(images):
        # blackhole image has only one channel
        np_image = (image[0] * 255).to(torch.uint8).cpu().numpy()
        plt.imsave(str(path / '{:04d}.png'.format(i)), np_image, cmap='afmhot_10us')
    # load from dir
    image_list = []
    trans = transforms.ToTensor()
    for i in range(len(images)):
        pil_image = Image.open(str(path / '{:04d}.png'.format(i)))
        t_image = trans(pil_image)
        image_list.append(t_image)
    images = torch.stack(image_list)
    save_image(images, target, nrow=nrow)


def save_grid_mri(images, target='image.png', nrow=10, normalize=True, image_type='amplitude'):
    if image_type == 'amplitude':
        img = images
    elif image_type == 'amplitude+phase':
        img = images[:, 0:1]
    elif image_type == 'real+imag':
        img = (images[:, 0:1] ** 2 + images[:, 1:2] ** 2).sqrt() * np.sqrt(2) - 1
    if normalize:
        img = norm_image_01(img)
    else:
        img = norm(img)
    # compute amplitude
    save_image(img, target, nrow=nrow)


def save_video(images, target='video.mp4', fps=30, normalize=True):
    if normalize:
        images = norm_image_01(images)
    with imageio.get_writer(target, fps=fps) as writer:
        for frame in images:
            img_frame = (frame * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            writer.append_data(img_frame)
            

def save_video_bh(images, target='video.mp4', path='black_image', fps=30, normalize=True):
    if normalize:
        images = norm_image_01(images)
    path = safe_dir(path)
    # save to dir
    for i, image in enumerate(images):
        # blackhole image has only one channel
        np_image = (image[0] * 255).to(torch.uint8).cpu().numpy()
        plt.imsave(str(path / '{:04d}.png'.format(i)), np_image, cmap='afmhot_10us')
    # load from dir
    image_list = []
    trans = transforms.ToTensor()
    for i in range(len(images)):
        pil_image = Image.open(str(path / '{:04d}.png'.format(i)))
        t_image = trans(pil_image)
        image_list.append(t_image)
    video = torch.stack(image_list)  # [T, C, H, W]

    with imageio.get_writer(target, fps=fps) as writer:
        for frame in video:
            img_frame = (frame * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            writer.append_data(img_frame)
    return video


def save_video_mri(images, target='video.mp4', fps=30, normalize=True, image_type='amplitude'):
    if image_type == 'amplitude':
        img = images
    elif image_type == 'amplitude+phase':
        img = images[:, 0:1]
    elif image_type == 'real+imag':
        img = (images[:, 0:1] ** 2 + images[:, 1:2] ** 2).sqrt() * np.sqrt(2) - 1
    if normalize:
        img = norm_image_01(img)
    else:
        img = norm(img)
    with imageio.get_writer(target, fps=fps) as writer:
        for frame in img:
            img_frame = (frame * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            writer.append_data(img_frame)


def save_video_bh_gif(images, target='video.gif', path='black_image', fps=30, normalize=True):
    if normalize:
        images = norm_image_01(images)
    path = safe_dir(path)
    # save to dir
    for i, image in enumerate(images):
        # blackhole image has only one channel
        np_image = (image[0] * 255).to(torch.uint8).cpu().numpy()
        plt.imsave(str(path / '{:04d}.png'.format(i)), np_image, cmap='afmhot_10us')
    # load from dir
    image_list = []
    trans = transforms.ToTensor()
    for i in range(len(images)):
        pil_image = Image.open(str(path / '{:04d}.png'.format(i)))
        t_image = trans(pil_image)
        image_list.append(t_image)
    video = torch.stack(image_list)  # [T, C, H, W]

    frames = []
    for frame in video:
        # Convert each frame to uint8 format and PIL Image
        img_frame = (frame * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        pil_frame = Image.fromarray(img_frame)
        frames.append(pil_frame)
    
    # Save frames as a GIF
    frames[0].save(target, save_all=True, append_images=frames[1:], duration=1000 // fps, loop=0)
