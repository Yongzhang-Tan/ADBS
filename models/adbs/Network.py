import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.resnet18_encoder import *
from models.resnet12 import *

class MYNET(nn.Module):

    def __init__(self, args, mode=None, trans=1):
        super().__init__()

        self.mode = mode
        self.args = args
        if self.args.dataset in ['cifar100']: 
            self.encoder = resnet12() 
            self.num_features = 640
        if self.args.dataset in ['mini_imagenet']: 
            self.encoder = resnet12()
            self.num_features = 640
        if self.args.dataset == 'cub200':
            self.encoder = resnet18(True, args)
            self.num_features = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.num_features, self.args.num_classes*trans, bias=False)

        # adaptive decision boundary 
        if self.args.margin:
            self.margin = nn.Parameter(torch.ones(self.args.num_classes * trans))
    
    def forward_metric(self, x):
        x, _ = self.encode(x)
        if 'cos' in self.mode:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            if self.args.margin:
                x = x * torch.abs(self.margin)
            x = self.args.temperature * x

        elif 'dot' in self.mode:
            x = self.fc(x)

        return x # joint, contrastive

    def encode(self, x):
        x, y = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x, y

    def forward(self, im_cla):
        if self.mode != 'encoder':
            x = self.forward_metric(im_cla)
            return x
        elif self.mode == 'encoder':
            x, _ = self.encode(im_cla)
            return x
        else:
            raise ValueError('Unknown mode')
    
    
    def update_fc(self,dataloader,class_list,transform,session):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            b = data.size()[0]
            data = transform(data)
            m = data.size()[0] // b 
            labels = torch.stack([label*m+ii for ii in range(m)], 1).view(-1)
            data, _ =self.encode(data)
            data.detach()

        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list)*m, self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, labels, class_list, m)
            

    def update_fc_avg(self,data,labels,class_list, m):
        new_fc=[]
        for class_index in class_list:
            for i in range(m):
                index = class_index*m + i
                data_index=(labels==index).nonzero().squeeze(-1)
                embedding=data[data_index]
                proto=embedding.mean(0)
                new_fc.append(proto)
                self.fc.weight.data[index]=proto
                if self.args.margin:
                    # Initialize the decision boundary of the new class
                    self.margin.data[index]=torch.mean(self.margin.data[:index])
        new_fc=torch.stack(new_fc,dim=0)
        return new_fc

    def get_logits(self,x,fc):
        if 'dot' in self.args.new_mode:
            return F.linear(x,fc)
        elif 'cos' in self.args.new_mode:
            return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))