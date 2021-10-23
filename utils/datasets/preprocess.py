"""
docstring
"""
import torch
from torchvision import transforms
import numpy as np
from PIL import Image, ImageChops


def get_transform_ops(resize=256, image_mean=None, crop='center',
                      crop_size=224, normalize=False):
    """
    docstring
    """
    ops = []
    if resize:
        ops.append(transforms.Resize(resize, Image.BICUBIC))
    if image_mean is not None:
        ops.append(MeanSubtractNumpy(np.load(image_mean)))
    if crop == 'center':
        crop = CenterCropNumpy(crop_size)
        ops.append(crop)
    elif crop == 'random':
        crop = RandomCropNumpy(crop_size)
        ops.append(crop)
    if normalize:
        ops.append(ToTensorScaled())  # Scale value to [0, 1]
        ops.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]))
    else:
        ops.append(ToTensorUnscaled())
    return transforms.Compose(ops)


class ToTensorScaled():
    """Convert a RGB PIL Image to a CHW ordered Tensor, scale the range to [
    0, 1] """

    def __call__(self, im_val):
        """
        docstring
        """
        im_val = np.array(im_val, dtype=np.float32).transpose((2, 0, 1))
        im_val /= 255.0
        return torch.from_numpy(im_val)

    def __repr__(self):
        """
        docstring
        """
        return 'ToTensorScaled(./255)'


class ToTensorUnscaled():
    """Convert a RGB PIL Image to a CHW ordered Tensor"""

    def __call__(self, im_val):
        """
        docstring
        """
        return torch.from_numpy(
            np.array(im_val, dtype=np.float32).transpose((2, 0, 1)))

    def __repr__(self):
        """
        docstring
        """
        return 'ToTensorUnscaled()'


class MeanSubtractPIL():
    """Mean subtract operates on PIL Images"""

    def __init__(self, im_mean):
        """
        docstring
        """
        self.im_mean = im_mean

    def __call__(self, im_val):
        """
        docstring
        """
        if self.im_mean is None:
            return im_val
        return ImageChops.subtract(im_val, self.im_mean)

    def __repr__(self):
        """
        docstring
        """
        if self.im_mean is None:
            return 'MeanSubtractNumpy(None)'
        return f'MeanSubtractNumpy(im_mean={self.im_mean.filename})'


class MeanSubtractNumpy():
    """Mean subtract operates on numpy ndarrays"""

    def __init__(self, im_mean):
        """
        docstring
        """
        self.im_mean = im_mean

    def __call__(self, im_val):
        """
        docstring
        """
        if self.im_mean is None:
            return im_val
        return np.array(im_val).astype('float') - self.im_mean.astype('float')

    def __repr__(self):
        """
        docstring
        """
        if self.im_mean is None:
            return 'MeanSubtractNumpy(None)'
        return f'MeanSubtractNumpy(im_mean={self.im_mean.shape})'


class CenterCropNumpy():
    """
    docstring
    """
    def __init__(self, size):
        """
        docstring
        """
        self.size = size

    def __call__(self, im_val):
        """
        docstring
        """
        im_val = np.array(im_val)
        size = self.size
        h_val, w_val, _ = im_val.shape
        if w_val == size and h_val == size:
            return im_val
        x_val = int(round((w_val - size) / 2.))
        y_val = int(round((h_val - size) / 2.))
        return im_val[y_val:y_val + size, x_val:x_val + size, :]

    def __repr__(self):
        """
        docstring
        """
        return f'CenterCropNumpy(size={self.size})'


class RandomCropNumpy():
    """
    docstring
    """
    def __init__(self, size):
        """
        docstring
        """
        self.size = size

    def __call__(self, im_val):
        """
        docstring
        """
        im_val = np.array(im_val)
        size = self.size
        h_val, w_val, _ = im_val.shape
        if w_val == size and h_val == size:
            return im_val
        x_val = np.random.randint(0, w_val - size)
        y_val = np.random.randint(0, h_val - size)
        return im_val[y_val:y_val + size, x_val:x_val + size, :]

    def __repr__(self):
        """
        docstring
        """
        return f'RandomCropNumpy(size={self.size})'
