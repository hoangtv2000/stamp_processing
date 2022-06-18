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
