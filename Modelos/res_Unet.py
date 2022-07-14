# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 17:10:00 2022

@author: MarioPC
"""

import torch
import torch.nn as nn
import cv2
from torchvision import transforms
import numpy as np


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, dilation):
        
        super(ResidualConv, self).__init__()
        padding = dilation * (3 - 1) // 2
        
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, 
                padding=padding, dilation=dilation),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)

class ResUnet(nn.Module):
    def __init__(self, channel, filters=[32, 64, 128, 256]):
        super(ResUnet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 3, 1, 1),
            nn.BatchNorm2d(3),
            #nn.Softmax(dim=1),
            # nn.ReLU(),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)

        return output



# model = ResUnet(3).cuda()
# img = cv2.imread('F:/cv_bootcamp/Proyecto/data/images/34.png')
# convert_tensor = transforms.ToTensor()
# # resized = cv2.resize(img, (64,64), interpolation = cv2.INTER_AREA)
# #test = convert_tensor(resized)
# test = convert_tensor(img)
# test = torch.unsqueeze(test, dim=0)
# aux = model(test.cuda())

# aux2 = np.uint8(aux.cpu().detach().numpy())[0][0]
# cv2.imshow("Test",aux2)


# # Example of target with class indices
# loss = nn.CrossEntropyLoss()
# # input = torch.randn(3, 5, requires_grad=True)
# # target = torch.empty(3, dtype=torch.long).random_(5)
# # output = loss(input, target)
# # output.backward()
# # Example of target with class probabilities

# im_target = cv2.imread('F:/cv_bootcamp/Proyecto/data/masks/34.png')

# input = torch.randn(3, 5, requires_grad=True)
# # target = torch.randn(3, 5).softmax(dim=1)
# target  = convert_tensor(im_target).softmax(dim=1)

# out1 = aux[0]
# output = loss(out1.cuda(), target.cuda())

# output.backward()


# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# batch_size = 4

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                           shuffle=True, num_workers=2)


import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import os
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
from PIL import Image
import random

from torchmetrics import JaccardIndex

# open method used to open different extension image file



class ImageDataset(Dataset):
    def __init__(self, targets_names, images_names,target_dir,images_dir,
                 transform=None, target_transform=None):
        self.targets_names = targets_names
        self.images_names = images_names
        self.transform = transform
        self.target_transform = target_transform
        self.target_dir = target_dir
        self.images_dir = images_dir

    def __len__(self):
        return len(self.targets_names)

    def __getitem__(self, idx):
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # image = read_image(img_path)
        # label = self.img_labels.iloc[idx, 1]
        # image = cv2.imread(self.images_dir+self.images_names[idx])
        image = Image.open(self.images_dir+self.images_names[idx]) 
        # target = cv2.imread(self.target_dir+self.targets_names[idx])
        target = Image.open(self.target_dir+self.targets_names[idx])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        if random.random() > 0.5:
            image = TF.hflip(image)
            target = TF.hflip(target)
    
        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            target = TF.vflip(target)
        
        return image, target

# =============================================================================
# def IoU_loss(output, target):
#     pred_labels = torch.argmax(outputs, dim=0)
#     target_labels = torch.argmax(targets, dim=0)
#     
#     return 1-jaccard(pred_labels ,target_labels)
# =============================================================================

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#       transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))])

transform = transforms.Compose(
    [transforms.ToTensor()])
mask_array = os.listdir('F:/cv_bootcamp/Proyecto/data/masks/')
images_array = os.listdir('F:/cv_bootcamp/Proyecto/data/images/')

trainset = ImageDataset(targets_names=mask_array, images_names=images_array,
                        target_dir='F:/cv_bootcamp/Proyecto/data/masks/',
                        images_dir='F:/cv_bootcamp/Proyecto/data/images/',
                        transform=transform, target_transform=transform)

train_dataloader = DataLoader(trainset, batch_size=5, shuffle=True)

from torch.optim.lr_scheduler import StepLR
model = ResUnet(3).cuda()
weights = [1,1,1.04]
class_weights = torch.FloatTensor(weights).cuda()
criterion = nn.CrossEntropyLoss(weight=class_weights)
criterion2 = nn.BCEWithLogitsLoss()
# criterion = nn.BCEWithLogitsLoss() + abs(nn.CrossEntropyLoss())
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.1)
scheduler = StepLR(optimizer, step_size=15, gamma=0.1)
for epoch in range(50):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, targets = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs.cuda())
        loss = (criterion(outputs, targets.cuda())+criterion2(outputs, targets.cuda()))
        # loss = criterion2(outputs, targets.cuda())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # if i % 2000 == 1999:    # print every 2000 mini-batches
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss}')
        running_loss = 0.0
    scheduler.step()

print('Finished Training')

# inv_transform =  transforms.Compose(
#     [transforms.Normalize((1/0.5, 1/0.5, 1/0.5), (1/0.5, 1/0.5, 1/0.5))])

# o = inv_transform(outputs)
# # aux =  np.uint8(o.cpu().detach().numpy())[0]
# aux =  np.uint8(outputs.cpu().detach().numpy())[0]
# cv2.imshow("Test",aux.T)


# MEAN = torch.tensor([0.5, 0.5, 0.5]).cuda()
# STD = torch.tensor([0.25, 0.25, 0.25]).cuda()

# x = outputs[0] * STD[:, None, None] + MEAN[:, None, None]
# plt.imshow(np.uint8(outputs[0].cpu().detach().numpy()*255).T)
# plt.imshow(np.uint8(targets[0].cpu().detach().numpy()*255).T)
# plt.xticks([])
# plt.yticks([])
torch.save(model.state_dict(), "FirstModel_6.ph")

# =============================================================================
# Test
# =============================================================================

model = ResUnet(3).cuda()
model.load_state_dict(torch.load("F:/cv_bootcamp/Proyecto/FirstModel.ph"))
model.eval()


model2 = ResUnet(3).cuda()
model2.load_state_dict(torch.load("F:/cv_bootcamp/Proyecto/FirstModel_6.ph"))
model2.eval()


model3 = ResUnet(3).cuda()
model3.load_state_dict(torch.load("F:/cv_bootcamp/Proyecto/FirstModel_4.ph"))
model3.eval()

jaccard = JaccardIndex(num_classes=3).cuda()
test_dataloader = DataLoader(trainset, batch_size=1, shuffle=False)
r1 = []
r2 = []
r3 = []
r_m = []
for i, data in enumerate(test_dataloader, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, targets = data

    outputs = model(inputs.cuda())
    outputs2 = model2(inputs.cuda())
    outputs3 = model3(inputs.cuda())
    
    pred_labels = torch.argmax(outputs[0], dim=0)
    target_labels = torch.argmax(targets[0], dim=0).cuda()
    
    pred_labels2 = torch.argmax(outputs2[0], dim=0)
    target_labels2 = torch.argmax(targets[0], dim=0).cuda()
    
    pred_labels3 = torch.argmax(outputs3[0], dim=0)
    target_labels3 = torch.argmax(targets[0], dim=0).cuda()
    
    r1.append(jaccard(pred_labels ,target_labels).cpu().detach().numpy())
    r2.append(jaccard(pred_labels2 ,target_labels2).cpu().detach().numpy())
    r3.append(jaccard(pred_labels3 ,target_labels3).cpu().detach().numpy())
    
    
    
    pred_labels_ = torch.argmax(torch.mean(torch.cat((outputs,outputs2,outputs3), 0), 0), dim=0)
    target_labels_m = torch.argmax(targets[0], dim=0).cuda()
    r_m.append(jaccard(pred_labels_ ,target_labels_m).cpu().detach().numpy())

target_dir='F:/cv_bootcamp/Proyecto/data/masks/'
images_dir='F:/cv_bootcamp/Proyecto/data/images/'

test_img =  Image.open(images_dir+"1.png")
target_img =  Image.open(target_dir+"1.png")

test_img=test_img.convert('RGB')
target_img =  target_img.convert('RGB')
 
resized = test_img.resize((512,512))
aux = torch.unsqueeze(transform(resized), dim=0)
result = model(aux.cuda())


# pred = np.argmax((result[0].cpu().detach().numpy()),0)

pred_labels = torch.argmax(result[0], dim=0)
target_labels = torch.argmax(transform(target_img), dim=0).cuda()
jaccard(pred_labels ,target_labels)


aux2  =transforms.ToPILImage(mode ="RGB")(result[0])
plt.imshow(np.uint8(result[0].cpu().detach().numpy()*255))
plt.imshow(np.uint8(resized*255))


# =============================================================================
# from torch.autograd import Variable
# 
# dummy_input = Variable(torch.randn(1,3,512,512)).cuda()
# torch.onnx.export(model3, dummy_input, "trained_model3.onnx")
# 
# =============================================================================
