import os
import cv2
import numpy as np
from typing import List, Union

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from stamp_processing.module.unet_onnx import UnetInferenceONNX
from stamp_processing.preprocess import create_batch
from stamp_processing.utils import REMOVER_WEIGHT_ID, check_image_shape


class StampRemoverONNX:
    def __init__(self, removal_weight: Union[str, None] = None):
        """Create an object to remove stamps from document images"""
        self.remover = UnetInferenceONNX(removal_weight)  # type: ignore
        self.padding = 3


    def __call__(self, image_list: Union[List[npt.NDArray], npt.NDArray], batch_size: int = 16) -> List[npt.NDArray]:
        """Detect and remove stamps from document images
        Args:
            image_list (Union[List[npt.NDArray], npt.NDArray]): list of input images
            batch_size (int, optional): Defaults to 16.
        Returns:
            List[np.ndarray]: Input images with stamps removed
        """
        if not isinstance(image_list, (np.ndarray, list)):
            raise TypeError("Invalid Type: Input must be of type list or np.ndarray")

        if len(image_list) > 0:
            check_image_shape(image_list[0])
        else:
            return []
        return self.__batch_removing(image_list, batch_size)  # type:ignore


    def __batch_removing(self, image_list, batch_size=16):  # type: ignore
        new_pages = []

        shapes = set(list(x.shape for x in image_list))
        images_batch, indices = create_batch(image_list, shapes, batch_size)
        # num_batch = len(image_list) // batch_size
        detection_predictions = []
        for batch in images_batch:
            if len(batch):
                detection_predictions.extend(remove_redstamp_batch(batch))
        z = zip(detection_predictions, indices)
        sorted_result = sorted(z, key=lambda x: x[1])
        detection_predictions, _ = zip(*sorted_result)

        for idx, page_boxes in enumerate(detection_predictions):
            page_img = cv2.cvtColor(image_list[idx], cv2.COLOR_BGR2RGB)
            h, w, c = page_img.shape

            for box in page_boxes:
                x, y, w_, h_ = box[:4]
                x_min, y_min, x_max, y_max = x, y, (x + w_), (y + h_)
                
                stamp_area = page_img[
                    max(y_min - self.padding, 0) : min(y_max + self.padding, h),
                    max(x_min - self.padding, 0) : min(x_max + self.padding, w),
                ]
                stamp_area = self.remover([stamp_area])  # type:ignore

                page_img[
                    max(y_min - self.padding, 0) : min(y_max + self.padding, h),
                    max(x_min - self.padding, 0) : min(x_max + self.padding, w),
                    :,
                ] = stamp_area[0]
            new_pages.append(page_img)

        return new_pages




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



def remove_redstamp(img_path):

    def is_rec_in_rec(recA, recB):
        return recA[0] < recB[0] < recB[2]-recB[0] < recA[2]-recA[0] and recA[1] < recB[1] < recB[3]-recB[1] < recA[3]-recA[1]

    def check_rec_in_rec(this_rect, all_rects):
        return True if True in [is_rec_in_rec(this_rect, that_rect) for that_rect in all_rects] else False

        
    img = cv2.imread(img_path)
    copied_image = img.copy()

    img = apply_brightness_contrast(img, brightness = 255, contrast = 145)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red_mask_1 = np.array([0,25,0], np.uint8)
    upper_red_mask_1 = np.array([10,255,255], np.uint8)
    mask1 = cv2.inRange(img_hsv, lower_red_mask_1, upper_red_mask_1)

    output_mask1 = cv2.bitwise_and(copied_image, copied_image, mask=mask1)

    lower_red_mask_2 = np.array([160,25,0])
    upper_red_mask_2 = np.array([180,255,255])
    mask2 = cv2.inRange(img_hsv, lower_red_mask_2, upper_red_mask_2)

    output_mask2 = cv2.bitwise_and(copied_image, copied_image, mask=mask2)

    output_mask = cv2.add(output_mask1, output_mask2) 
    result_image = copied_image - output_mask

    _, thresh = cv2.threshold(result_image, 5, 255, cv2.THRESH_BINARY_INV)
    opening_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(11, 11)))
    opening_thresh_ = cv2.cvtColor(opening_thresh, cv2.COLOR_BGR2GRAY)

    contours = cv2.findContours(opening_thresh_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    result_bboxes = list()
    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        if (w >= 224 or h >= 224) and not check_rec_in_rec([x, y, w, h], result_bboxes):
            result_bboxes.append([x, y, w, h])

    return result_bboxes
    

def remove_redstamp_batch(batch_image):
    batch_list = list()
    for image in batch_image:
        batch_list.append(remove_redstamp(image))
    return batch_list


def imshow(img, figsize=(25, 25), **kwargs):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.axis('off')
    ax.imshow(img, **kwargs)


