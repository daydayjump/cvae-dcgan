#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: daydayjump
@contact: newlifestyle2014@126.com
@file: cvae-dcgan.py
@time: 18-6-29 上午10:56
'''
from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torchvision import datasets, transforms, models

parser = argparse.ArgumentParser(description='vae_dcgan')
parser.add_argument('--imageSize', type=int, default=224,
                    help='the height / width of the input image to network')
parser.add_argument('--labelDim', type=int, default=13,
                    help='the dimension of the label to network')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for traning(default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train(default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enable CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                   help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--outf', default='result/', help='folder to output images and model checkpoints')
parser.add_argument('--hidden-size', type=int, default=200, help='size of z')
parser.add_argument('--num-gpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--kernel',metavar='N', type=int, default=64, help='the size of origin kernel')

args = parser.parse_args()
print(args)

try:
    os.makedirs(args.outf)
except OSError:
    pass

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device('cuda' if args.cuda else 'cpu')
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

'''
model
'''
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        n = 16
        self.vgg_16_bn = models.vgg16_bn(pretrained=True)
        self.vgg_16_bn.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout()
        )
        self.fc11 = nn.Linear(4096, args.hidden_size)
        self.fc12 = nn.Linear(4096, args.hidden_size)

    def encoder(self, x):
            # input: noise
            # output: mu and sigma
            # input: args.batch_size x 3 x 224 x 224
            out = self.vgg_16_bn(x)
            return self.fc11(out.view(out.size(0), -1)), self.fc12(out.view(out.size(0), -1))

    def reparameterize(self, mu, logvar):
            if self.training:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return eps.mul(std).add_(mu)
            else:
                return mu

    def forward(self,x):
            mu, logvar = self.encoder(x)
            out = self.reparameterize(mu, logvar)
            return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        n = 64
        self.encoder = Encoder()
        self.fc0 = nn.Linear(args.hidden_size, n * 8 * 8)
        self.fc1_1 = nn.Linear()
        self.fc1_2 = nn.Linear()
        self.fc2_1 = nn.Linear(args.batch_size,)
        self.fc2_2 = nn.Linear()


        #code and label use fc rather than deconv
        # Use ReLU activation in generator for all layers except for the output, which uses Tanh
        self.deconv0_1 = nn.Sequential(nn.ConvTranspose2d(n*8*8, n * 4, kernel_size=5, stride=1, padding=0),
                                       nn.BatchNorm2d(n * 4),
                                       nn.ReLU())
        self.deconv0_2 = nn.Sequential(nn.ConvTranspose2d(13, n * 4, kernel_size=5, stride=1, padding=0),
                                       nn.BatchNorm2d(n * 4),
                                       nn.ReLU())
        # 5 x 5
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(n * 8, n * 4, kernel_size=5, stride=2, padding=0,output_padding=1),
                                     nn.BatchNorm2d(n * 4),
                                     nn.ReLU())
        # 14 x 14
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(n * 4, n * 2, kernel_size=3, stride=2, padding=1,output_padding=1),
                                     nn.BatchNorm2d(n * 2),
                                     nn.ReLU())
        # 28 x 28
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(n * 2, n, kernel_size=3, stride=2, padding=1,output_padding=1),
                                     nn.BatchNorm2d(n),
                                     nn.ReLU())
        # 56 x 56
        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(n , n, kernel_size=3, stride=2, padding=1, output_padding=1),
                                     nn.BatchNorm2d(n),
                                     nn.ReLU())
        # 112 x 112
        self.deconv5 = nn.Sequential(nn.ConvTranspose2d(n, 3, kernel_size=3, stride=2, padding=1,output_padding=1),
                                     nn.Tanh())
        # 224 x 224 using Tanh() as activations is better than sigmoid

    def forward(self,x, label):
        n = 64
        out = self.encoder(x)
        # input: args.batch_size x args.hidden_size
        out = self.fc2(out)
        # args.batch_size x (n*8*8 x 1 x 1),to make out add label achieved
        out = self.deconv0_1(out.view(x.size(0), n*8*8,1,1))
        # add label
        y = self.deconv0_2(label)
        out = torch.cat([out,y],1)
        # args.batch_size x n*8 x 5 x 5
        out = self.deconv1(out)
        # args.batch_size x n*4 x 14 x 14
        out = self.deconv2(out)
        # args.batch_size x n*2 x 28 x 28
        out = self.deconv3(out)
        # args.batch_size x n x 56 x 56
        out = self.deconv4(out)
        # args.batch_size x n x 112 x 112
        out = self.deconv5(out)
        # args.batch_size x 3 x 224 x 224
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        n = 64
        self.input_1 = nn.Sequential(
            # input args.batch_size x 3 x 224 x 224
            nn.Conv2d(3, n/2, kernel_size=3, stride=2,padding=1),
            nn.LeakyReLU(0.2,inplace=True))
            # Use LeakyReLU activation in the discriminator for all layers
        self.input_2 = nn.Sequential(
            # input args.batch_size x 13 x 224 x 224
            nn.Conv2d(13, n/2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True))
        self.main = nn.Sequential(
            # args.batch_size x n x 112 x 112
            nn.Conv2d(n, n*2, kernel_size=3, stride=2,padding=1),
            nn.BatchNorm2d(n*2),
            nn.LeakyReLU(),
            # args.batch_size x n*2 x 56 x 56
            nn.Conv2d(n*2, n*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(n*4),
            nn.LeakyReLU(0.2, inplace=True),
            # args.batch_size x n*4 x 28 x 28
            nn.Conv2d(n*4, n*8,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(n*8),
            nn.LeakyReLU(0.2, inplace=True),
            # args.batch_size x n*8 x 14 x 14
            nn.Conv2d(n*8,n*16,kernel_size=5,stride=2,padding=1),
            nn.BatchNorm2d(n*16),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.output = nn.Sequential(
            # args.batch_size x n*16 x 5 x 5
            nn.Conv2d(n*8,1, kernel_size=5,stride=1,padding=0),
            #nn.Linear(n*8*14*14,1),
            nn.Sigmoid()
        )
        # For the discriminator, the last convolution layer
        # is flattened and then fed into a single sigmoid output

    def forward(self,x,label):
        out = self.input_1(x)
        y = self.input_2(label)
        out = torch.cat([out,y],1)
        out = self.main(x)
        out = self.output(x).view(-1,1)
        return out

if __name__ == '__main__':
    netE = Encoder()
    netG = Generator()
    netD = Discriminator()
    with open('./model.txt',"r+w") as fm:
        print (netE,'\n',netG,'\n',netD,file = fm)


