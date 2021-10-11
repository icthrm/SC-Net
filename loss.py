import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Gradient_Net(nn.Module):
  def __init__(self, batch_size = 1):
    super(Gradient_Net, self).__init__()
    kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    kernel_x = torch.FloatTensor(kernel_x).expand(3, 3, 3).unsqueeze(0).to(device)

    kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
    kernel_y = torch.FloatTensor(kernel_y).expand(3, 3, 3).unsqueeze(0).to(device)

    kernel_x = kernel_x.expand(batch_size, kernel_x.shape[0], kernel_x.shape[1], kernel_x.shape[2], kernel_x.shape[3])
    kernel_y = kernel_y.expand(batch_size, kernel_y.shape[0], kernel_y.shape[1], kernel_y.shape[2], kernel_y.shape[3])

    self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
    self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    # self.norm_layer = nn.BatchNorm3d(num_features=1)

  def forward(self, x):
    # x = self.norm_layer(x)
    grad_x = F.conv3d(x, self.weight_x)
    grad_y = F.conv3d(x, self.weight_y)
    gradient = torch.abs(grad_x) + torch.abs(grad_y)
    return gradient


def gradient(x, batch_size):
    gradient_model = Gradient_Net(batch_size=batch_size).to(device)
    g = gradient_model(x)
    return g


class Mean_Block(nn.Module):
  def __init__(self, batch_size = 1, kernel_size = 5):
    super(Mean_Block, self).__init__()
    kernel_means = [[1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.]
                    , [1., 1., 1., 1., 1.], [1., 1., 1., 1., 1.]]
    kernel_means = torch.FloatTensor(kernel_means).expand(kernel_size, kernel_size, kernel_size)
    kernel_means = kernel_means / (kernel_size*kernel_size*kernel_size)
    kernel_means = kernel_means.unsqueeze(0).to(device)
    kernel_means = kernel_means.expand(batch_size, kernel_means.shape[0], kernel_means.shape[1], kernel_means.shape[2],
                                       kernel_means.shape[3])
    self.weight_means = nn.Parameter(data=kernel_means, requires_grad=False)

  def forward(self, x):
    mean_block = F.conv3d(x, self.weight_means)
    return mean_block


def mean_by_blocks(x, batch_size):
    mean_model = Mean_Block(batch_size=batch_size).to(device)
    m = mean_model(x)
    return m
