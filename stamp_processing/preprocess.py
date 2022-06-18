from typing import List, Set, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import torch


def create_batch(
    images: npt.NDArray, shapes: Set[Tuple[int, int]], batch_size: int = 16
) -> Tuple[List[List[npt.NDArray]], List[int]]:
    """
    - Input:
        +) images: List images
        +) shapes: set of all shapes of input images
        +) batch_size: number image in one batch
    - Output:
        +) images_batch: batch of images for inference
        +) indices: order of all input images
    """
    split_batch = []
    images_batch = []
    for shape in shapes:
        mini_batch = []
        images_mini_batch = []  # type: ignore
        for idx, img in enumerate(images):
            if img.shape == shape:
                mini_batch.append(idx)
                img = apply_brightness_contrast(img, 255, 145)
                if len(images_mini_batch) < batch_size:
                    images_mini_batch.append(img)
                else:
                    images_batch.append(images_mini_batch)
                    images_mini_batch = []
                    images_mini_batch.append(img)
        images_batch.append(images_mini_batch)
        split_batch.append(mini_batch)
    del images_mini_batch

    indices = [item for sublist in split_batch for item in sublist]
    return images_batch, indices


def apply_brightness_contrast(input_img, brightness = 255, contrast = 125):
    def map(x, in_min, in_max, out_min, out_max):
        return int((x-in_min) * (out_max-out_min) / (in_max-in_min) + out_min)

    brightness = map(brightness, 0, 510, -255, 255)
    contrast = map(contrast, 0, 254, -127, 127)
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    if contrast != 0:
        f = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    return buf