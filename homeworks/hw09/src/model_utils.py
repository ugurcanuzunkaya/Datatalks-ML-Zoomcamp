import numpy as np
from PIL import Image
from io import BytesIO
from urllib import request


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size, Image.NEAREST)
    return img


def preprocess_input(img):
    # Convert to numpy
    x = np.array(img, dtype="float32")

    # 1. Rescale to 0-1
    x /= 255.0

    # 2. Normalize (Standard ImageNet stats from previous homework)
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")

    x = (x - mean) / std

    # 3. Transpose to (Channels, Height, Width)
    x = x.transpose((2, 0, 1))

    # 4. Add batch dimension
    x = np.expand_dims(x, axis=0)

    return x
