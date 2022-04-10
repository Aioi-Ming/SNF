import numpy as np
import torch
import matplotlib.pyplot as plt

import sys

class Mueller(object):

    params_default = {'k' : 1.0,
                      'dim' : 2}


    aa = [-1, -1, -6.5, 0.7]
    bb = [0, 0, 11, 0.6]
    cc = [-10, -10, -6.5, 0.7]
    AA = [-200, -100, -170, 15]
    XX = [1, 0, -0.5, -1]
    YY = [0, 0.5, 1.5, 1]

    def __init__(self, params=None):
        # set parameters
        if params is None:
            params = self.__class__.params_default
        self.params = params

        # useful variables
        self.dim = self.params['dim']

    def energy(self, x):
        """Muller potential

        Returns
        -------
        potential : {float, np.ndarray}
            Potential energy. Will be the same shape as the inputs, x and y.

        Reference
        ---------
        Code adapted from https://cims.nyu.edu/~eve2/ztsMueller.m
        """
        x1 = x[:, 0]
        x2 = x[:, 1]
        value = 0
        for j in range(0, 4):
            value += self.AA[j] * np.exp(self.aa[j] * (x1 - self.XX[j])**2 +
                                         self.bb[j] * (x1 - self.XX[j]) * (x2 - self.YY[j]) +
                                         self.cc[j] * (x2 - self.YY[j])**2)
        # redundant variables
        if self.dim > 2:
            value += 0.5 * np.sum(x[:, 2:] ** 2, axis=1)

        return self.params['k'] * value

    def energy_torch(self, x):
        """Muller potential

        Returns
        -------
        potential : {float, np.ndarray}
            Potential energy. Will be the same shape as the inputs, x and y.

        Reference
        ---------
        Code adapted from https://cims.nyu.edu/~eve2/ztsMueller.m
        """
        x1 = x[:, 0]
        x2 = x[:, 1]
        batchsize = x.shape[0]
        value = torch.zeros(batchsize)
        for j in range(0, 4):
            value += self.AA[j] * torch.exp(self.aa[j] * (x1 - self.XX[j])**2 +
                                         self.bb[j] * (x1 - self.XX[j]) * (x2 - self.YY[j]) +
                                         self.cc[j] * (x2 - self.YY[j])**2)
        # redundant variables
        if self.dim > 2:
            value += 0.5 * torch.reduce_sum(x[:, 2:] ** 2, axis=1)

        return self.params['k'] * value
