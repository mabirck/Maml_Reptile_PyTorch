from __future__ import print_function
import numpy as np
import torch
from torch import Tensor, nn


from net import SinusoidNet


class MAML:
    def __init__(self, dim_input=1, dim_output=1, test_num_updates=5, args=None):
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.update_lr = args.update_lr
        self.meta_lr = Tensor(args.meta_lr)
        self.classification = False
        self.test_num_updates = test_num_updates

        if args.datasource == "sinusoid":
            self.dim_hidden = [40, 40]
            self.loss_func = nn.MSELoss()
            self.forward = self.forward_fc
            self.construct_weights = self.construct_fc_weights
        else:
            raise ValueError('Unrecognized data source.')

    def build(self, inputs, prefix='metatrain_'):
        if inputs is None:
            self.inputa = Tensor(np.zeros([1]))
            self.inputb = Tensor(np.zeros([1]))
            self.labela = Tensor(np.zeros([1]))
            self.labelb = Tensor(np.zeros([1]))
        # else:
        #     self.inputa = input_tensors['inputa']
        #     self.inputb = input_tensors['inputb']
        #     self.labela = input_tensors['labela']
        #     self.labelb = input_tensors['labelb']

        self.net = SinusoidNet()

        if torch.cuda.is_available():
            self.net.cuda()
        
