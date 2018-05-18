import torch
import torch.nn as nn


class SinusoidNet(nn.Module):
    '''
        Net to Perform Sinusoid Meta-Learning
    '''

    def __init__(self):
        super(SinusoidNet, self).__init__()
        self.layers = []
