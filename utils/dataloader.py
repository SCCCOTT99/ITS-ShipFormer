import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from .utils import cvtColor, preprocess_input
from .utils_aug import CenterCrop, ImageNetPolicy, RandomResizedCrop, Resize


class DataGenerator(data.Dataset):
    def __init__(self, annotation_lines, input_shape, random=True, autoaugment_flag=True):
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.random = random

        self.autoaugment_flag = autoaugment_flag
        if self.autoaugment_flag:
            self.resize_crop = RandomResizedCrop(input_shape)
            self.policy = ImageNetPolicy()

            self.resize = Resize(input_shape[0] if input_shape[0] == input_shape[1] else input_shape)
            self.center_crop = CenterCrop(input_shape)

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):
        annotation_path = self.annotation_lines[index].split(";")[1].split()[0]
        image = Image.open(annotation_path)

        image = cvtColor(image)
        if self.autoaugment_flag:
            image = self.AutoAugment(image, random=self.random)
        else:
            image = self.get_random_data(image, self.input_shape, random=self.random)
        image = np.transpose(preprocess_input(np.array(image).astype(np.float32)), [2, 0, 1])

        y = int(self.annotation_lines[index].split(";")[0])
        return image, y

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, input_shape, jitter=0.3, hue=0.1, sat=1.5, val=1.5, random=True):
        iw, ih = image.size
        h, w = input_shape

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new("RGB", (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            return image_data

        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.75, 1.5)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new("RGB", (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        flip = self.rand() < 0.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        rotate = self.rand() < 0.5
        if rotate:
            angle = np.random.randint(-15, 15)
            a, b = w / 2, h / 2
            M = cv2.getRotationMatrix2D((a, b), angle, 1)
            image = cv2.warpAffine(np.array(image), M, (w, h), borderValue=[128, 128, 128])

        image_data = np.array(image, np.uint8)

        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1

        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        return image_data

    def AutoAugment(self, image, random=True):
        if not random:
            image = self.resize(image)
            image = self.center_crop(image)
            return image

        image = self.resize_crop(image)

        flip = self.rand() < 0.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image = self.policy(image)
        return image


def detection_collate(batch):
    images = []
    targets = []
    for image, y in batch:
        images.append(image)
        targets.append(y)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    targets = torch.from_numpy(np.array(targets)).type(torch.FloatTensor).long()
    return images, targets
