from __future__ import print_function

import argparse
import os
import numpy as np
import shutil
import time
import random
from PIL import Image

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
from math import cos, pi

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def get_recognizer_model(checkpoint_path):
    model = models.__dict__['resnet50'](pretrained=False)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def recognize_attributes(face_image, model):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize,])
    face_image = Image.fromarray(face_image)
    input = data_transforms(face_image)
    input = input.unsqueeze(0)
    output = model(input)
    res = [F.softmax(item)[0][1].item() for item in output]
    return res


