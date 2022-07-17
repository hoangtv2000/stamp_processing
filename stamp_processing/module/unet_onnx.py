from typing import List

import numpy as np
from fastai.vision.all import *

import onnxruntime


class UnetInferenceONNX():

    def __init__(self, model_path):
        self.onnx_session = onnxruntime.InferenceSession(model_path)


    def __call__(self, image_array: str, bs: int = 1) -> List[np.ndarray]:
        """INPUT : List of PIL images
           OUTPUT: List of images  type numpy.ndarray
        """
        if len(image_array) < 1:
            return []

        batches = self.__build_batches(image_array, bs=bs)
        outs = []

        for b in batches:
            ort_inputs = {self.onnx_session.get_inputs()[0].name: b}
            try:
                ort_outs = self.onnx_session.run([self.onnx_session.get_outputs()[0].name], ort_inputs)
                outs.append(ort_outs)
                del b
            except:
                continue

        pil_images = self.__decode_prediction(outs)
        return pil_images


    def __decode_prediction(self, preds):
        out = []
        i2f = IntToFloatTensor()
        for pred in preds:
            img_np = i2f.decodes(pred.squeeze())
            img_np = img_np.transpose(1, 2, 0)
            img_np = img_np.astype(np.uint8)
            out.append(img_np)
            # out.append(Image.fromarray(img_np))
            del img_np
        return out


    def __build_batches(self, image_array: list, bs=1):
        batches = []
        for im in (image_array):
            batches.append(image_transform_onnx(im))
        return batches


def image_transform_onnx(image) -> np.ndarray:
    image = np.array(image)
    image = image.transpose(2,0,1).astype(np.float32)
    image /= 255
    image = image[None,...]
    return image