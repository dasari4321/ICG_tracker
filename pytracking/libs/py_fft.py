#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 11:57:03 2022

@author: dasari
"""
from pytracking.libs.tensorlist import tensor_operation
import torch
from torchvision.models.resnet import resnet18
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


def fft_feats(A):
#A = torch.reshape(torch.arange(1,25),(2,3,4)) # 2x3x4
#A = img_tensor
    Y1 = torch.permute(torch.fft.fftshift(torch.fft.fft(torch.permute(A,(0,1,2,3)))),(0,1,2,3)) #  A.sum(2) 2x3 across width 
    Y2 = torch.permute(torch.fft.fftshift(torch.fft.fft(torch.permute(A,(0,3,1,2)))),(0,2,3,1)) #  A.sum(1) 2x4 across height
    Y3 = torch.permute(torch.fft.fftshift(torch.fft.fft(torch.permute(A,(0,2,3,1)))),(0,3,1,2)) #  A.sum(0) 3x4 across channels
    Y4 = torch.permute(torch.fft.fftshift(torch.fft.fft2(torch.permute(A,(0,1,2,3)))),(0,1,2,3)) # A.sum(1).sum(1) or A.sum(2).sum(1)
    Y5 = torch.permute(torch.fft.fftshift(torch.fft.fft2(torch.permute(A,(0,3,1,2)))),(0,2,3,1)) # A.sum(1).sum(0) or A.sum(0).sum(0)
    Y6 = torch.permute(torch.fft.fftshift(torch.fft.fft2(torch.permute(A,(0,2,3,1)))),(0,3,1,2)) # A.sum(0).sum(1) or A.sum(2).sum(0)
    Y7 = torch.fft.fftshift(torch.fft.fftn(A))
    return (Y1+Y2+Y3+Y4+Y5+Y6+Y7)/7

image = Image.open("/home/dasari/demo/one_x.jpg")
transform = transforms.Compose([
    transforms.ToTensor()])
  
img_tensor = transform(image)
net = resnet18(True)
FE = torch.nn.Sequential(*list(net.children())[:-2])
clss = torch.nn.Sequential(*list(net.children())[-2:])
A = img_tensor.unsqueeze(0)
A1 = FE(A)
A2 = abs(fft_feats(A1))
pred = clss[1](torch.flatten(clss[0](A2),1))

