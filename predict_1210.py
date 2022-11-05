from PIL import Image
import requests
import io
import math
import matplotlib.pyplot as plt
import cv2

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision
import torchvision.transforms as T
import numpy
numpy.set_printoptions(threshold=numpy.inf)

import glob
from hubconf import *
from util.misc import nested_tensor_from_tensor_list

torch.set_grad_enabled(False);


# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model, postprocessor = torch.hub.load('', 'detr_mine', pretrained=True,
                                      return_postprocessor=True, num_classes=2, source='local')
model.eval()


def img_read():
    # path = 'data/coco/val2017/'
    # im = path + "IMG_9755.jpg"
    # path = 'data/large/'
    # im = path + "IMG_8903.jpg"
    im = 'C:/CloudMusic/12345.jpg'
    # path = 'C:/ima/'
    # im = path + 'IMG_0398.jpg'
    # img_list = glob.glob(path + '*.jpg')
    # for i in img_list:
    #     im = Image.open(i)
    #     im.show()

    img = Image.open(im)
    return img


def predict(im, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)
    out = model(img)
    return img, out


def predict_total_mask_instance(img, out, ori_img, img_name):
    # the post-processor expects as input the target size of the predictions (which we set here to the image size)
    # size.shape = batch_size * 2
    # type(size) = torch.Tensor
    target_size = torch.as_tensor(img.shape[-2:]).unsqueeze(0)
    w, h = target_size.squeeze().numpy().tolist()
    result = list()
    result = postprocessor(result, out, target_size, target_size)

    total = torch.zeros((w, h), dtype=torch.bool)
    tar = numpy.asarray(result[0]['masks'], dtype=numpy.float)
    for i in range(tar.shape[0]):
        x = tar[i][0] != 0
        total = total | x
        # plt.imshow(tar[i][0], cmap='cividis')
        # plt.show()
    # plt.imshow(total, cmap='cividis')
    plt.subplot(121)
    plt.imshow(ori_img)
    plt.subplot(122)
    plt.imshow(total, cmap='cividis')
    # plt.savefig(img_name, dpi=300)
    plt.show()


def predict_instance(img, out):
    # the post-processor expects as input the target size of the predictions (which we set here to the image size)
    # size.shape = batch_size * 2
    # type(size) = torch.Tensor
    target_size = torch.as_tensor(img.shape[-2:]).unsqueeze(0)
    w, h = target_size.squeeze().numpy().tolist()
    result = list()
    result = postprocessor(result, out, target_size, target_size)

    total = torch.zeros((w, h), dtype=torch.bool)
    tar = numpy.asarray(result[0]['masks'], dtype=numpy.float)
    for i in range(tar.shape[0]):
        x = tar[i][0] != 0
        total = total | x
        # plt.imshow(tar[i][0], cmap='cividis')
        # plt.show()
    # plt.imshow(total, cmap='cividis')
    return total


def plot(img, pre):
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(pre, cmap='cividis')
    plt.show()


if __name__ == '__main__':
    im = img_read()
    img, out = predict(im, model, transform)
    # predict_total_mask_instance(img, out, im, 'IMG_9755.jpg')
    pre = predict_instance(img, out)
    plot(im, pre)
