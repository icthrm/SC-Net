'''

'''

import torch
import torch.nn as nn
import numpy as np


def gaussian_filter_3d(sigma, s=11):
    dim = s//2
    xx,yy,zz = np.meshgrid(np.arange(-dim, dim+1), np.arange(-dim, dim+1), np.arange(-dim,dim+1))
    d = xx**2 + yy**2 + zz**2
    f = np.exp(-0.5*d/sigma**2)
    return f


class GaussianDenoise3d(nn.Module):
    def __init__(self, sigma, scale=5):
        super(GaussianDenoise3d, self).__init__()
        width = 1 + 2*int(np.ceil(sigma*scale))
        f = gaussian_filter_3d(sigma, s=width)
        f /= f.sum()

        self.filter = nn.Conv3d(1, 1, width, padding=width//2)
        self.filter.weight.data[:] = torch.from_numpy(f).float()
        self.filter.bias.data.zero_()

    def forward(self, x):
        return self.filter(x)
