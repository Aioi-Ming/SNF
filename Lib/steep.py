import numpy as np
import torch
import matplotlib.pyplot as plt

import sys

class Steep(object):

    params_default = {'w': 1,
                      'k' : 1.0,
                      'c' : 5.0,
                      'dim' : 2}
    def __init__(self, params=None, w=None):
        # set parameters
        if params is None:
            params = self.__class__.params_default
            if w is not None:
                params['w']=w
        self.params = params

        # useful variables
        self.dim = self.params['dim']
        self.max_energy = 1e6
        self.T=1/self.params['w']
        self.T2=self.T/2

    def energy(self, x):
        out_range = np.logical_or.reduce( [x[:,0] < -1, x[:,0] > 1]).astype('float32')
        
        x1=np.remainder(x[:,0],self.T)
        
        dimer_energy = (np.abs(x1*4/self.T-2)-1)* self.params['c']
        
        oscillator_energy = 0.0
        if self.dim == 2:
            oscillator_energy = (self.params['k'] / 2.0) * x[:, 1] ** 2
        if self.dim > 2:
            oscillator_energy = np.sum((self.params['k'] / 2.0) * x[:, 1:] ** 2, axis=1)
        return  (dimer_energy + oscillator_energy) *(1- out_range) + out_range*self.max_energy

    def energy_torch(self, x):
        
        out_range =  torch.logical_or( x[:,0] < -1, x[:,0] > 1)
        
        x1=torch.remainder(x[:,0],self.T)
        
        dimer_energy = (torch.abs(x1*4/self.T-2)-1)* self.params['c']
        
        oscillator_energy = 0.0
        if self.dim == 2:
            oscillator_energy = (self.params['k'] / 2.0) * x[:, 1] ** 2
        if self.dim > 2:
            oscillator_energy = torch.sum((self.params['k'] / 2.0) * x[:, 1:] ** 2, axis=1)
        return  (dimer_energy + oscillator_energy) *~out_range + out_range*self.max_energy
    
    
    def plot_dimer_energy(self, axis=None, temperature=1.0):
        """ Plots the dimer energy to the standard figure """
        x_grid = np.linspace(-1, 1, num=200)
        if self.dim == 1:
            X = x_grid[:, None]
        else:
            X = np.hstack([x_grid[:, None], np.zeros((x_grid.size, self.dim - 1))])
        energies = self.energy(X) / temperature

        if axis is None:
            axis = plt.gca()
        #plt.figure(figsize=(5, 4))
        axis.plot(x_grid, energies, linewidth=3, color='black')
        axis.set_xlabel('x / a.u.')
        axis.set_ylabel('Energy / kT')
        axis.set_ylim(energies.min() - 2.0, energies[int(energies.size / 2)] + 2.0)

        return x_grid, energies
    
    def plot_sample_energy(self,x):
        x1=x[:,0]
        counts, bins = np.histogram(x1, bins = 200 )
        anchors = (bins[1:] + bins[:-1]) / 2
        probs = counts / np.sum(counts)

        anchors = anchors[np.where(probs > 0.000001)]
        probs = probs[np.where(probs > 0.000001)]

        f = -np.log(probs)
        fn = f - np.min(f)
        plt.scatter(anchors, fn)         
        x_grid = np.linspace(-1, 1, num=200)
        if self.dim == 1:
            X = x_grid[:, None]
        else:
            X = np.hstack([x_grid[:, None], np.zeros((x_grid.size, self.dim - 1))])
        energies = self.energy(X)
        energies = energies -energies.min()
        axis = plt.gca()
        #plt.figure(figsize=(5, 4))
        axis.plot(x_grid, energies, linewidth=3, color='black')
        axis.set_xlabel('x / a.u.')
        axis.set_ylabel('Energy / kT')
        axis.set_ylim(energies.min() - 2.0, energies[int(energies.size / 2)] + 2.0)
        return x_grid, energies
