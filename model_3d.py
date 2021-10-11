from layer_3d import *

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.utils.data
from torch.optim import lr_scheduler


class UNet3DEM(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, norm='bnorm'):
        super(UNet3DEM, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        """
        Encoder part
        """

        self.enc1_1 = nn.Sequential(nn.Conv3d(1, nch_ker, kernel_size=3, stride=2, padding=1)
                                 , nn.BatchNorm3d(nch_ker)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.enc2_1 = nn.Sequential(nn.Conv3d(nch_ker, nch_ker, kernel_size=3, stride=2, padding=1)
                                 , nn.BatchNorm3d(nch_ker)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.enc3_1 = nn.Sequential(nn.Conv3d(nch_ker, nch_ker, kernel_size=3, stride=2, padding=1)
                                 , nn.BatchNorm3d(nch_ker)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.enc4_1 = nn.Sequential( nn.Conv3d(nch_ker, nch_ker, kernel_size=3, stride=2, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )

        """
        Decoder part
        """

        self.dec3_1 = nn.Sequential( nn.Conv3d(2 * nch_ker, 2 * nch_ker, kernel_size=3, padding=1)
                                 , nn.BatchNorm3d(2*nch_ker)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv3d(2 * nch_ker, 2 * nch_ker, kernel_size=3, padding=1)
                                 , nn.BatchNorm3d(2 * nch_ker)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec2_1 = nn.Sequential( nn.Conv3d(3 * nch_ker, 2 * nch_ker, kernel_size=3, padding=1)
                                 , nn.BatchNorm3d(2 * nch_ker)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv3d(2 * nch_ker, 2 * nch_ker, kernel_size=3, padding=1)
                                 , nn.BatchNorm3d(2 * nch_ker)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.dec1_1 = nn.Sequential( nn.Conv3d(2 * nch_ker+1, nch_ker, kernel_size=3, padding=1)
                                 , nn.BatchNorm3d(nch_ker)
                                 , nn.LeakyReLU(0.1)
                                 , nn.Conv3d(nch_ker, 1, kernel_size=3, padding=1)
                                 )

    def forward(self, x):

        '''

        Args:
            x: input feature map

        Returns:
            prediction y
        '''

        """
        Encoder part
        """
        enc1 = self.enc1_1(x)
        enc2 = self.enc2_1(enc1)
        enc3 = self.enc3_1(enc2)
        enc4 = self.enc4_1(enc3)

        """
        Decoder part
        """
        unpool3 = F.interpolate(enc4, size=(enc2.size(2), enc2.size(3), enc2.size(4)), mode='nearest')
        cat3 = torch.cat([unpool3, enc2], dim=1)
        dec3 = self.dec3_1(cat3)

        unpool2 = F.interpolate(dec3, size=(enc2.size(2), enc2.size(3), enc2.size(4)), mode='nearest')
        cat2 = torch.cat([unpool2, enc2], dim=1)
        dec2 = self.dec2_1(cat2)

        unpool1 = F.interpolate(dec2, size=(x.size(2), x.size(3), x.size(4)), mode='nearest')
        cat1 = torch.cat([unpool1, x], dim=1)
        dec1 = self.dec1_1(cat1)

        y = dec1

        return y


class UNet3DEM_UP(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, norm='bnorm'):
        super(UNet3DEM_UP, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        """
        Encoder part
        """
        self.enc1_1 = nn.Sequential(nn.Conv3d(1, nch_ker, kernel_size=3, stride=2, padding=1)
                                 , nn.BatchNorm3d(nch_ker)
                                 , nn.Conv3d(nch_ker, nch_ker, kernel_size=3, stride=2, padding=1)
                                 , nn.BatchNorm3d(nch_ker)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.enc2_1 = nn.Sequential(nn.Conv3d(nch_ker, nch_ker, kernel_size=3, stride=2, padding=1)
                                 , nn.BatchNorm3d(nch_ker)
                                 , nn.Conv3d(nch_ker, nch_ker, kernel_size=3, stride=2, padding=1)
                                 , nn.BatchNorm3d(nch_ker)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.enc3_1 = nn.Sequential(nn.Conv3d(nch_ker, nch_ker, kernel_size=3, stride=2, padding=1)
                                 , nn.BatchNorm3d(nch_ker)
                                 , nn.Conv3d(nch_ker, nch_ker, kernel_size=3, stride=2, padding=1)
                                 , nn.BatchNorm3d(nch_ker)
                                 , nn.LeakyReLU(0.1)
                                 )
        self.enc4_1 = nn.Sequential( nn.Conv3d(nch_ker, nch_ker, kernel_size=3, stride=2, padding=1)
                                 , nn.LeakyReLU(0.1)
                                 )

        """
        Decoder part
        """
        self.dec3_1 = Up(2 * nch_ker, 2 * nch_ker, bilinear=True)
        self.dec2_1 = Up(3 * nch_ker, 2 * nch_ker, bilinear=True)
        self.dec1_1 = Up(2 * nch_ker, 1, bilinear=True)

    def forward(self, x):

        '''

        Args:
            x: input feature map

        Returns:
            prediction y
        '''

        """
        Encoder part
        """
        enc1 = self.enc1_1(x)
        enc2 = self.enc2_1(enc1)
        enc3 = self.enc3_1(enc2)
        enc4 = self.enc4_1(enc3)

        """
        Decoder part
        """
        k = self.dec3_1(enc4, enc3)
        k = self.dec2_1(k, enc2)
        k = self.dec1_1(k, enc1)

        y = k

        return y


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            # if you have padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
            x1 = torch.cat([x2, x1], dim=1)
        x = x1
        return self.conv(x)


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if gpu_ids:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

