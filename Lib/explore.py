import numpy as np
import torch
from torch import distributions
from torch import nn
from torch.utils import data

class Configuration(object):
  def __init__(self, bg, X0, capacity, batch_size=1024):
    self.bg=bg
    self.lr=1e-4
    self.batch_size=batch_size

    self.I = np.arange(capacity)
    I_X0=np.arange(X0.shape[0])
    I_select=np.random.choice(I_X0, size=capacity, replace=True)
    self.X=X0[I_select]

    self.loss_train = []
    self.acceptance_rate = []
    self.stepsize = []

  def configure(self, epochs, stepsize=None, iter=300, lr=1e-4, start_step=0.5):
    if stepsize is None:  # initialize stepsize when called for the first time
        if len(self.stepsize) == 0:
            self.stepsize.append(1)
    else:
        self.stepsize = [start_step]



    for e in range(epochs):
        ax=plt.gca()
        ax.hist2d(self.X[:,0],self.X[:,1],bins=100,norm=matplotlib.colors.LogNorm())
        plt.show()
        plot_energy(self.X[:,0]) #plot energy
      
        #sample batch
        I_select=np.random.choice(self.I,size=self.batch_size, replace=True)
#        x_batch=self.X[I_select]
        x_batch=torch.from_numpy(self.X[I_select])

        #train 
#        loss1=self.bg.train_ML(x_batch, iter=iter, lr=lr)
#        print('iter %s:' % e, 'loss = %.3f' % loss1[-1]) 
#        loss2=self.bg.train_KL(iter=iter,lr=lr)     
#        print('iter %s:' % e, 'loss = %.3f' % loss2[-1]) 
        
        loss=self.bg.train_mix(x_batch,iter=iter,lr=lr)
        print('iter %s:' % e, 'loss = %.3f' % loss[-1])
        
        z_batch, Jxz_batch = self.bg.backward_flow(x_batch)
        
        #
#        E0=0.5*torch.linalg.norm(z_batch,dim=1)**2
#        z_batch_new = z_batch + self.stepsize[-1] * torch.randn(z_batch.shape[0], z_batch.shape[1])
#        E1=0.5*torch.linalg.norm(z_batch_new,dim=1)**2
#        x_batch_new, Jzx_batch_new = self.bg.forward_flow(z_batch_new)

#        rand = -torch.log(torch.rand(self.batch_size))
#        Iacc = rand >= E1-E0
#        x_acc = x_batch_new[Iacc]

#        self.X[I_select[Iacc]] = x_acc.detach().numpy()

#        pacc = float(np.count_nonzero(Iacc)) / float(self.batch_size)
#        self.acceptance_rate.append(pacc)
        

        #methopolis
        E0 = self.bg.target_energy(x_batch) + Jxz_batch
        z_batch_new = z_batch + self.stepsize[-1] * torch.randn(z_batch.shape[0], z_batch.shape[1])

        x_batch_new, Jzx_batch_new = self.bg.forward_flow(z_batch_new)
        E1 = self.bg.target_energy(x_batch_new) - Jzx_batch_new

        #accept and replace
        rand = -torch.log(torch.rand(self.batch_size))
        Iacc = rand >= E1-E0

        x_acc = x_batch_new[Iacc]
        self.X[I_select[Iacc]] = x_acc.detach().numpy()

        pacc = float(np.count_nonzero(Iacc)) / float(self.batch_size)
        self.acceptance_rate.append(pacc)

        #update stepsize
        if stepsize is None:
          if len(self.acceptance_rate) > 2:  # update stepsize
              mean_acceptance_rate = np.mean(self.acceptance_rate[-2:])
              if mean_acceptance_rate < 0.3:
                  self.stepsize.append(max(self.stepsize[-1] - 0.1, 0.001))
              elif mean_acceptance_rate > 0.7:
                  self.stepsize.append(min(self.stepsize[-1] + 0.1, 1.0))
              else:
                  self.stepsize.append(self.stepsize[-1])  # just copy old stepsize
          else:
              self.stepsize.append(self.stepsize[-1])  # just copy old stepsize