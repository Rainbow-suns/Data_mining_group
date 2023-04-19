import torch
from torchvision import transforms
import torchvision.transforms.functional as torchVisionTransforms


class Resacle(object):

    def __init__(self, out_size: tuple):
        self.out_size = 256

    def __call__(self, image):
        originHeight, originWidth = image.shape[1:3]
        newHeight, newWidth = self.out_size
        newHeight, newWidth = int(newHeight), int(newWidth)

        resizeHeightRatio = newHeight / originHeight
        resizeWidthRatio = newWidth / originWidth

        image = torchVisionTransforms.resize(image, [newWidth, newHeight])

        return image
