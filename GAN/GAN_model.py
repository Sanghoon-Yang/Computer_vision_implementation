import torch
import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()
        self.noise = params['noise']
        self.img_size = params['size']



    def fc_layer(self, in_dim, out_dim, normalize = True):
        layers = nn.Sequential(nn.Linear(in_dim, out_dim))
        if normalize:
            layers.add(nn.BatchNorm1d(out_dim))
            layers.add(nn.LeakyReLU(0.2))
        else:
            layers.add(nn.LeakyReLU(0.2))

        return layers

    def forward(self,x):
        img = self.fc_layer(self.noise, 128, normalize=False)(x)
        img = self.fc_layer(128, 256)(img)
        img = self.fc_layer(256,512)(img)
        img = self.fc_layer(512, 1024)(img)
        img = self.fc_layer(1024, int(np.prod(self.img_size)))(img)
        img = nn.Tanh()(img)

        img = img.view(img.size(0), *self.img_size)

        return img


class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()
        self.img_size = params['size']

        self.discri = nn.Sequential(nn.Linear(np.prod(self.img_size), 512),
                                    nn.LeakyReLU(0.2),
                                    nn.Linear(512, 256),
                                    nn.LeakyReLU(0.2),
                                    nn.Linear(256,1),
                                    nn.Sigmoid())

    def forward(self, x):
        x = x.view(x.size(0),-1)
        x = self.discri(x)

        return x
