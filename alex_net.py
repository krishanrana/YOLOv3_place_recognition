#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 12:24:54 2018

@author: Krishan
"""

import torch.nn as nn
import torch
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
from matplotlib import pyplot as plt
import time
import json


__all__ = ['AlexNet', 'alexnet']


f = open('labels.json', 'r')
data = json.load(f).items()

#Download the image and create a pillow image
img_pil = Image.open("objects/single_t_light.png")


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
        
    def forward(self, x):
        
         #Features
         
        layers = [] 
        
        self.cnn1 =  self.features[0]
        self.relu1 = self.features[1]
        self.maxpool1 = self.features[2]
        
        self.cnn2 = self.features[3]
        self.relu2 = self.features[4]
        self.maxpool2 = self.features[5]
        
        self.cnn3 = self.features[6]
        self.relu3 = self.features[7]
        
        self.cnn4 = self.features[8]
        self. relu4 = self.features[9]
        
        self.cnn5 = self.features[10]
        self.relu5 = self.features[11]
        self.maxpool5 = self.features[12]
            
        
        #Classifier
        
        self.dropout1 = self.classifier[0]
        self.linear1 = self.classifier[1]
        self.relu6 = self.classifier[2]
        self.dropout2 = self.classifier[3]
        self.linear2 = self.classifier[4]
        self.relu7 = self.classifier[5]
        self.linear3 = self.classifier[6]
        
        
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        
        layers.append(out)
        
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        
        layers.append(out)
        
        out = self.cnn3(out)
        out = self.relu3(out) 
        
        layers.append(out)
        
        out = self.cnn4(out)
        out = self. relu4(out)
        
        layers.append(out)
        
        out = self.cnn5(out)
        out = self.relu5(out)
        out = self.maxpool5(out)
        
        layers.append(out)
        
        out = out.view(out.size(0), 256 * 6 * 6)
        
        out = self.dropout1(out)
        out = self.linear1(out)
        out = self.relu6(out)
        out = self.dropout2(out)
        out = self.linear2(out)
        out = self.relu7(out)
        out = self.linear3(out)
        
        
        return out, layers


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(torch.load('alexnet_weights.pth'))
    return model




cnn = alexnet(True)
print(cnn)

start = time.time()
#Defining the preprocessing transform

normalize = transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
        )

preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
        ])


#Preprocess the image
img_tensor = preprocess(img_pil)
img_tensor.unsqueeze_(0)
img_tensor = img_tensor[:,0:3,:,:]

#Run a forward pass with the network
img_variable = Variable(img_tensor) #Input to the network must be autograd variable
fc_out, layers = cnn(img_variable)

end = time.time()
print(end - start)

#Download the labels
labels = {int(key):value for (key, value) in data}

print(labels[fc_out.data.numpy().argmax()])


try:
    plt.close('all')
    plt.figure(), plt.set_cmap('hot'), plt.imshow(plt.imshow(layers[2][0,10,:,:].detach().numpy()))
    
except TypeError:
    pass
