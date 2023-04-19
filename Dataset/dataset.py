import torch
from torch.utils.data import Dataset

from Dataset.transforms import Resacle
from common.util import readImageAsTensor
import os
import cv2


class BinaryDataSet(Dataset):
    image_size = 256  # 640 448
    def __init__(self, pathImage, useSIFT, useSVM):
        imageFileNames = os.listdir(pathImage)
        self.imgFilePaths = [os.path.join(pathImage, img) for img in imageFileNames]
        self.useSIFT = useSIFT
        self.useSVM = useSVM

    def __getitem__(self, index):
        img_path = self.imgFilePaths[index]
        list = img_path.split("\\")
        fileName = list[len(list) - 1][0]

        if self.useSVM is True:
            if fileName == 't':
                label = 1
            else:
                label = -1
        else:
            if fileName == 't':
                label = 1
            else:
                label = 0

        if self.useSIFT is True:
            x = self.generateSift(img_path)
        else:
            x = readImageAsTensor(img_path)
        return x, label

    def __len__(self):
        return len(self.imgFilePaths)

    def generateSift(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.image_size, self.image_size))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        nfeatures = 30
        sift = cv2.SIFT_create(nfeatures, contrastThreshold=0.001)
        keypoints, _ = sift.detectAndCompute(gray, None)
        tensorPoints = []
        for index in range(nfeatures):
            tensorPoints.append(torch.as_tensor(keypoints[index].pt))

        keypointsTensor = torch.stack(tensorPoints, dim=1)
        return keypointsTensor


def createBinaryDataset(negPath, useSIFT=False, useSVM=False):
    return BinaryDataSet(negPath, useSIFT, useSVM)
