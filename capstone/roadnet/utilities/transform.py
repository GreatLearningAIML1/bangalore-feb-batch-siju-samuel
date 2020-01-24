import random
import numpy as np
import torch
from torchvision.transforms import Compose, CenterCrop, Normalize, Scale, Pad
from torchvision.transforms import ToTensor, ToPILImage

from PIL import Image

class Relabel:

    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        assert (isinstance(tensor, torch.LongTensor) or isinstance(
            tensor, torch.ByteTensor)), 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor


class ToLabel:

    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long().unsqueeze(0)



NUM_CHANNELS = 3
NUM_CLASSES = 10

image_transform = ToPILImage()

'''input_transform = Compose([
    CenterCrop(256),
    ToTensor(),
    Normalize([.485, .456, .406], [.229, .224, .225]),
])'''

target_transform = Compose([
    CenterCrop(256),
    ToLabel(),
    Relabel(255, 21),
])

class TransformImages(object):

    def __init__(self, enc, augment=True, height=512):
        self.enc = enc
        self.augment = augment
        self.height = height
        pass

    def AssignLabel(self, target):
        road_label = 19
        max_label = 28
        for iter in range(1, road_label):
            target = Relabel(iter, 255)(target)
        for iter in range(road_label, max_label):
            target = Relabel(iter, (iter-road_label+1))(target)
        return target

    def __call__(self, input, target):
        # do something to both images
        #print("TransformImages input = ", np.array(input).shape)
        input = Scale(self.height, Image.BILINEAR)(input)
        target = Scale(self.height, Image.NEAREST)(target)
        #print("scaled input = ", np.array(input))
        #print("scaled input = ", np.array(input).shape)

        if (self.augment):
            # Random hflip
            hflip = random.random()
            if (hflip < 0.5):
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)
        return ToTensor()(input), self.AssignLabel(ToLabel()(target))


def colormap_cityscapes(n):
    cmap = np.zeros([n, 3]).astype(np.uint8)
    cmap[0, :] = np.array([128, 64, 128])
    cmap[1, :] = np.array([124, 0, 0])
    cmap[2, :] = np.array([0, 80, 255])
    cmap[3, :] = np.array([255, 160, 0])
    cmap[4, :] = np.array([255, 255, 0])
    cmap[5, :] = np.array([130, 110, 90])

    cmap[6, :] = np.array([80, 110, 120])
    cmap[7, :] = np.array([80, 200, 255])
    cmap[8, :] = np.array([157, 143, 123])
    cmap[9, :] = np.array([240, 160, 60])
    cmap[10, :] = np.array([0, 0, 0])

    cmap[11, :] = np.array([220, 20, 60])
    cmap[12, :] = np.array([255, 0, 0])
    cmap[13, :] = np.array([0, 0, 142])
    cmap[14, :] = np.array([0, 0, 70])
    cmap[15, :] = np.array([0, 60, 100])

    cmap[16, :] = np.array([0, 80, 100])
    cmap[17, :] = np.array([0, 0, 230])
    cmap[18, :] = np.array([119, 11, 32])
    cmap[19, :] = np.array([124, 0, 0])
    cmap[20, :] = np.array([64, 164, 223])
    cmap[21, :] = np.array([153, 76, 0])
    cmap[22, :] = np.array([128, 64, 128])
    cmap[23, :] = np.array([0, 0, 255])

    return cmap


def colormap(n):
    cmap = np.zeros([n, 3]).astype(np.uint8)

    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(8):
            r = r + (1 << (7 - j)) * ((i & (1 << (3 * j))) >> (3 * j))
            g = g + (1 << (7 - j)) * ((i & (1 << (3 * j + 1))) >> (3 * j + 1))
            b = b + (1 << (7 - j)) * ((i & (1 << (3 * j + 2))) >> (3 * j + 2))

        cmap[i, :] = np.array([r, g, b])


    return cmap

class Colorize:
    def __init__(self, n=24):
        #self.cmap = colormap(256)
        self.cmap = colormap_cityscapes(256)
        #print("cmap=", self.cmap)
        self.cmap[n] = self.cmap[-1]
        #print("cmap=", self.cmap)
        self.cmap = torch.from_numpy(self.cmap[:n])
        #print("cmap=", self.cmap)
        #print("cmap shape=", self.cmap.shape)

    def __call__(self, gray_image):
        size = gray_image.size()
        #print("Colorize size=", size)
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)
        #color_image = torch.ByteTensor(3, size[0], size[1]).fill_(0)

        #for label in range(1, len(self.cmap)):
        #print("len(self.cmap)=", len(self.cmap))
        for label in range(0, len(self.cmap)):
            mask = gray_image[0] == label
            #mask = gray_image == label
            #print("MAsk = ", mask.shape)

            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image

color_transform = Colorize(NUM_CLASSES)

