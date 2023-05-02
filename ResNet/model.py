import torch
import torch.nn as nn

batch_size = 1

class BottleNeck(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, activation_fn):
        super(BottleNeck,self).__init__()
        self.bottleneck = nn.Sequential(nn.Conv2d(in_dim, mid_dim, kernel_size=1, stride=1),
                                   activation_fn,
                                   nn.Conv2d(mid_dim, mid_dim, kernel_size=3, stride=1, padding=1),
                                   activation_fn,
                                   nn.Conv2d(mid_dim, out_dim, kernel_size=1, stride=2),
                                   activation_fn)
        self.downsample = nn.Conv2d(in_dim, out_dim, kernel_size=1,stride=1)

    def forward(self,x):
        sample = self.downsample(x)
        out = self.bottleneck(x)
        out += sample

        return out

class BottleNeck_without_downsample(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, activation_fn):
        super(BottleNeck_without_downsample, self).__init__()
        self.bottleneck_without = nn.Sequential(nn.Conv2d(in_dim, mid_dim, kernel_size=1, stride=1),
                                   activation_fn,
                                   nn.Conv2d(mid_dim, mid_dim, kernel_size=3, stride=1, padding=1),
                                   activation_fn,
                                   nn.Conv2d(mid_dim, out_dim, kernel_size=1, stride=2),
                                   activation_fn)

    def forward(self,x):
        out = self.bottleneck_without(x)
        out += x
        return out


class BottleNeck_stride(nn.Module):
    def __int__(self, in_dim, mid_dim, out_dim, activation_fn):
        super(BottleNeck_stride, self).__init__()
        self.bottleneck_stride = nn.Sequential(nn.Conv2d(in_dim,mid_dim, kernel_size=1, stride=2),
                                   activation_fn,
                                   nn.Conv2d(mid_dim, mid_dim, kernel_size=3, stride=1, padding=1),
                                   activation_fn,
                                   nn.Conv2d(mid_dim, out_dim, kernel_size=1, stride=2),
                                   activation_fn)
        self.downsample = nn.Conv2d(in_dim, out_dim, kernel_size=1,stride=2)

    def forward(self,x):
        sample = self.downsample(x)
        out = self.bottleneck_stride(x)
        out += sample

        return out

class ResNet(nn.Module):
    def __init__(self, dimension, num_class):
        super(ResNet,self).__init__()
        self.layer_1st = nn.Sequential(nn.Conv2d(3, dimension, 7,2,3),
                                       nn.ReLU(),
                                       nn.MaxPool2d(3,2,1))
        self.layer_2nd = nn.Sequential(BottleNeck(dimension, dimension, dimension*4, nn.ReLU()),
                                       BottleNeck_without_downsample(dimension*4, dimension, dimension*4, nn.ReLU()),
                                       BottleNeck_stride(dimension*4, dimension, dimension*4, nn.ReLU()))

        self.layer_3rd = nn.Sequential(BottleNeck(dimension*4, dimension*2, dimension*8, nn.ReLU()),
                                       BottleNeck_without_downsample(dimension*8, dimension*2, dimension*8, nn.ReLU()),
                                       BottleNeck_without_downsample(dimension * 8, dimension*2, dimension * 8,nn.ReLU()),
                                       BottleNeck_stride(dimension*8, dimension*2, dimension*8, nn.ReLU()))
        self.layer_4th = nn.Sequential(BottleNeck(dimension*8, dimension*4, dimension*16, nn.ReLU()),
                                       BottleNeck_without_downsample(dimension*16, dimension*4, dimension*16, nn.ReLU()),
                                       BottleNeck_without_downsample(dimension*16, dimension*4, dimension*16, nn.ReLU()),
                                       BottleNeck_without_downsample(dimension*16, dimension*4, dimension*16, nn.ReLU()),
                                       BottleNeck_without_downsample(dimension*16, dimension*4, dimension*16, nn.ReLU()),
                                       BottleNeck_stride(dimension*16, dimension*4, dimension*16, nn.ReLU()))
        self.layer_5th = nn.Sequential(BottleNeck(dimension*16, dimension*8, dimension*32, nn.ReLU()),
                                       BottleNeck_without_downsample(dimension*32, dimension*8, dimension*32, nn.ReLU()),
                                       BottleNeck_stride(dimension*32, dimension*8, dimension*32, nn.ReLU()))
        self.fc_layer = nn.Linear(dimension*32, num_class)

    def forward(self,x):
        out = self.layer_1st(x)
        out = self.layer_2nd(out)
        out = self.layer_3rd(out)
        out = self.layer_4th(out)
        out = self.layer_5th(out)
        out = nn.AvgPool2d(7,1)(out)
        out = out.view(batch_size, -1)
        out = self.fc_layer(out)

        return out
