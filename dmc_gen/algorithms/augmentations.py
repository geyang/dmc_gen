import os

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as TF

# Specify one or more data directories here
DATA_DIRS = '/your/data/path/here/',
PLACES_LOADER = None
PLACES_ITER = None


def _load_places(batch_size=256, image_size=84, num_workers=16):
    global PLACES_LOADER, PLACES_ITER
    print('Loading places365_standard...')
    for data_dir in DATA_DIRS:
        if os.path.exists(data_dir):
            fp = os.path.join(data_dir, 'places365_standard', 'train')
            PLACES_LOADER = torch.utils.data.DataLoader(
                datasets.ImageFolder(fp, TF.Compose([
                    TF.RandomResizedCrop(image_size),
                    TF.RandomHorizontalFlip(),
                    TF.ToTensor()
                ])),
                batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True)
            PLACES_ITER = iter(PLACES_LOADER)
            break
    if PLACES_ITER is None:
        raise FileNotFoundError('failed to find places365 data at any of the specified paths')
    print('Load complete.')


def _get_places_batch(batch_size):
    global PLACES_ITER
    try:
        imgs, _ = next(PLACES_ITER)
        if imgs.size(0) < batch_size:
            PLACES_ITER = iter(PLACES_LOADER)
            imgs, _ = next(PLACES_ITER)
    except StopIteration:
        PLACES_ITER = iter(PLACES_LOADER)
        imgs, _ = next(PLACES_ITER)
    return imgs.cuda()


def random_overlay(x, dataset='places365_standard'):
    """Randomly overlay an image from Places"""
    global PLACES_ITER
    alpha = 0.5

    if dataset == 'places365_standard':
        if PLACES_LOADER is None:
            _load_places(batch_size=x.size(0), image_size=x.size(-1))
        imgs = _get_places_batch(batch_size=x.size(0)).repeat(1, x.size(1) // 3, 1, 1)
    else:
        raise NotImplementedError(f'overlay has not been implemented for dataset "{dataset}"')

    return ((1 - alpha) * (x / 255.) + (alpha) * imgs) * 255.


def batch_from_obs(obs, batch_size=32):
    """Copy a single observation along the batch dimension"""
    if isinstance(obs, torch.Tensor):
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)
        return obs.repeat(batch_size, 1, 1, 1)

    if len(obs.shape) == 3:
        obs = np.expand_dims(obs, axis=0)
    return np.repeat(obs, repeats=batch_size, axis=0)


def prepare_pad_batch(obs, next_obs, action, batch_size=32):
    """Prepare batch for self-supervised policy adaptation at test-time"""
    batch_obs = batch_from_obs(torch.from_numpy(obs).cuda(), batch_size)
    batch_next_obs = batch_from_obs(torch.from_numpy(next_obs).cuda(), batch_size)
    batch_action = torch.from_numpy(action).cuda().unsqueeze(0).repeat(batch_size, 1)

    return random_crop_cuda(batch_obs), random_crop_cuda(batch_next_obs), batch_action


def random_crop_cuda(x, size=84, w1=None, h1=None, return_w1_h1=False):
    """Vectorized CUDA implementation of random crop"""
    assert isinstance(x, torch.Tensor) and x.is_cuda, \
        'input must be CUDA tensor'

    n = x.shape[0]
    img_size = x.shape[-1]
    crop_max = img_size - size

    if crop_max <= 0:
        if return_w1_h1:
            return x, None, None
        return x

    x = x.permute(0, 2, 3, 1)

    if w1 is None:
        w1 = torch.LongTensor(n).random_(0, crop_max)
        h1 = torch.LongTensor(n).random_(0, crop_max)

    windows = view_as_windows_cuda(x, (1, size, size, 1))[..., 0, :, :, 0]
    cropped = windows[torch.arange(n), w1, h1]

    if return_w1_h1:
        return cropped, w1, h1

    return cropped


def view_as_windows_cuda(x, window_shape):
    """PyTorch CUDA-enabled implementation of view_as_windows"""
    assert isinstance(window_shape, tuple) and len(window_shape) == len(x.shape), \
        'window_shape must be a tuple with same number of dimensions as x'

    slices = tuple(slice(None, None, st) for st in torch.ones(4).long())
    win_indices_shape = [
        x.size(0),
        x.size(1) - int(window_shape[1]),
        x.size(2) - int(window_shape[2]),
        x.size(3)
    ]

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(x[slices].stride()) + list(x.stride()))

    return x.as_strided(new_shape, strides)


def random_crop(imgs, size=84, w1=None, h1=None, return_w1_h1=False):
    """Vectorized random crop, imgs: (B,C,H,W), size: output size"""
    assert (w1 is None and h1 is None) or (w1 is not None and h1 is not None), \
        'must either specify both w1 and h1 or neither of them'

    is_tensor = isinstance(imgs, torch.Tensor)
    if is_tensor:
        assert imgs.is_cuda, 'input images are tensors but not cuda!'
        return random_crop_cuda(imgs, size=size, w1=w1, h1=h1, return_w1_h1=return_w1_h1)

    n = imgs.shape[0]
    img_size = imgs.shape[-1]
    crop_max = img_size - size

    if crop_max <= 0:
        if return_w1_h1:
            return imgs, None, None
        return imgs

    imgs = np.transpose(imgs, (0, 2, 3, 1))
    if w1 is None:
        w1 = np.random.randint(0, crop_max, n)
        h1 = np.random.randint(0, crop_max, n)

    windows = view_as_windows(imgs, (1, size, size, 1))[..., 0, :, :, 0]
    cropped = windows[np.arange(n), w1, h1]

    if return_w1_h1:
        return cropped, w1, h1

    return cropped
