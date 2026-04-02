import torch
import random
import io
from PIL import Image
from torchvision import transforms

class JPEGCompression:
    """A transform simulating JPEG compression artifacts"""
    def __init__(self, quality_min=30, quality_max=70):
        assert 0 < quality_min <= quality_max <= 100
        self.quality_min = quality_min
        self.quality_max = quality_max

    def __call__(self, img):
        quality = random.randint(self.quality_min, self.quality_max)
        buffer = io.BytesIO()
        img.save(buffer, "JPEG", quality=quality)
        buffer.seek(0)
        img_jpeg = Image.open(buffer)
        return img_jpeg

class AddGaussianNoise:
    """A transform adding Gaussian noise"""
    def __init__(self, mean=0., std_min=1, std_max=10):
        self.mean = mean
        self.std_min = std_min
        self.std_max = std_max

    def __call__(self, tensor):
        std = random.uniform(self.std_min, self.std_max) / 255.0
        return tensor + torch.randn(tensor.size()) * std + self.mean

def get_blur_noise_transform():
    return transforms.Compose([
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(1.0, 2.5))], p=1.0),
    ])

def get_low_resolution_transform(quality_min=10, quality_max=40):
    return transforms.Compose([
        JPEGCompression(quality_min=quality_min, quality_max=quality_max),
    ])

def get_dark_transform(brightness=0.2):
    return transforms.ColorJitter(brightness=brightness)

import albumentations as A
def get_rain_transform():
    return A.RandomRain(
        p=1.0,
        slant_range=(-10, 10),
        drop_length=20,
        drop_width=1,
        drop_color=(200, 200, 200),
        blur_value=7,
        brightness_coefficient=0.7,
        rain_type='heavy'
    )

def get_snow_transform():
    """
    Get a snow effect transform.
    We directly use the official A.RandomSnow class from albumentations.
    """
    return A.RandomSnow(
        p=1.0,
        snow_point_range=(0.1, 0.3),
        brightness_coeff=2.5
    )
