import numpy as np
import torch
from torch import distributions
from torch import nn
from torch.utils import data

class BG_RealNVP(nn.Module):
    def __init__(self, target, dim, stochastic=False, step_size=0.25, nsteps=10, n_hidden=256, n_block=8, masks=None,
                 nets=None, nett=None):
        super(BG_RealNVP, self).__init__()
        self.stochastic = stochastic
        self.step_size = step_size
        self.nsteps = nsteps
        self.target_model = target
        self.n_hidden = n_hidden
        self.n_block = n_block
        self.dim = dim

        if nets == None:
            nets = lambda: nn.Sequential(nn.Linear(dim, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_hidden), nn.ReLU(), nn.Linear(n_hidden, dim), nn.Tanh())
        if nett == None:
            nett = lambda: nn.Sequential(nn.Linear(dim, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_hidden), nn.ReLU(), nn.Linear(n_hidden, dim))

        self.prior = distributions.MultivariateNormal(torch.zeros(self.dim), torch.eye(self.dim))

        if masks == None:
            self.masks = nn.Parameter(torch.from_numpy(np.array(
                [np.tile([0, 1], int(self.dim / 2)), np.tile([1, 0], int(self.dim / 2))] * self.n_block).astype(
                np.float32)), requires_grad=False)

        self.nett = torch.nn.ModuleList([nett() for _ in range(len(self.masks))])  # translation function (net)
        self.nets = torch.nn.ModuleList([nets() for _ in range(len(self.masks))])  # scaling function (net)

    def MCMC_forward(self, x ):
        stepsize=self.step_size
        nsteps=self.nsteps
        E0 = self.target_energy(x).reshape((x.shape[0],1))
        Et = E0
        for i in range(nsteps):
            # proposal step
            dx = stepsize * torch.zeros_like(x).normal_()
            xprop = x + dx
            Eprop = self.target_energy(xprop).reshape((x.shape[0],1))
            # acceptance step
            acc = (torch.rand(x.shape[0],1) < torch.exp(-(Eprop - Et))).float()  # selection variable: 0 or 1.
            x = (1-acc) * x + acc * xprop
            Et = (1-acc) * Et + acc * Eprop

        dW = (Et - E0).reshape(x.shape[0],)
        return x, dW     
    
    def MCMC_backward(self, z):
        stepsize=self.step_size
        nsteps=self.nsteps
        E0 = self.prior_energy(z).reshape((z.shape[0],1))
        Et = E0
        for i in range(nsteps):
            # proposal step
            dz = stepsize * torch.zeros_like(z).normal_()
            zprop = z + dz
            Eprop = self.prior_energy(zprop).reshape((z.shape[0],1))
            # acceptance step
            acc = (torch.rand(z.shape[0],1) < torch.exp(-(Eprop - Et))).float()  # selection variable: 0 or 1.
            z = (1-acc) * z + acc * zprop
            Et = (1-acc) * Et + acc * Eprop

        dW = (Et - E0).reshape(z.shape[0],)
        return z, dW    
    
    def target_energy(self, x):
#        return self.prior_energy(x)
        return self.target_model.energy_torch(x)
    

    def prior_energy(self, z):
        return 0.5 * torch.linalg.norm(z, dim=1) ** 2

    def forward_flow(self, z):

        log_R_zx, x = z.new_zeros(z.shape[0]), z

        for i in range(len(self.masks)):
            x1 = x * self.masks[i]

            s = self.nets[i](x1) * (1 - self.masks[i])
            t = self.nett[i](x1) * (1 - self.masks[i])

            x = x1 + (1 - self.masks[i]) * (x * torch.exp(s) + t)
            log_R_zx += torch.sum(s, -1)
            if self.stochastic==True:
                x,dw=self.MCMC_forward(x)
                log_R_zx+=dw
        return x, log_R_zx

    def backward_flow(self, x):
        log_R_xz, z = x.new_zeros(x.shape[0]), x

        for i in reversed(range(len(self.masks))):
            if self.stochastic==True:
                z, dw=self.MCMC_backward(z)
                log_R_xz+=dw
            z1 = z * self.masks[i]

            s = self.nets[i](z1) * (1 - self.masks[i])
            t = self.nett[i](z1) * (1 - self.masks[i])

            z = z1 + (1 - self.masks[i]) * (z - t) * torch.exp(-s)
            log_R_xz -= torch.sum(s, -1)

        return z, log_R_xz

    def sample(self, batchSize):
        z = self.prior.sample((batchSize,))
        #      logp = self.prior.log_prob(z)
        x, log_R_zx = self.forward_flow(z)
        return z.detach().numpy(), x.detach().numpy(), log_R_zx.detach().numpy()

    def loss(self, batch, w_ml=1.0, w_kl=0.0, w_rc=0.0):
        return w_ml * self.loss_ml(batch) + w_kl * self.loss_kl(batch) + w_rc * self.loss_rc(batch)

    def loss_ml(self, batch_x):
        z, log_R_xz = self.backward_flow(batch_x)
        energy = self.prior_energy(z)
        return torch.mean(energy - log_R_xz)

    def loss_kl(self, batch_z):
        x, log_R_zx = self.forward_flow(batch_z)
        energy = self.target_energy(x)
        e_high = 1e10
        for i in range(len(energy)):
            if abs(energy[i]) == float('inf'):
                print("energy overflow detected")
            elif energy[i] > e_high:
                energy[i] = e_high + torch.log(energy[i] - e_high + 1.0)

        return torch.mean(energy - log_R_zx)

    def loss_mix(self, batch_x, batch_z=None, w_kl=0.5, w_ml=0.5):
        if batch_z == None:
            batch_z = self.prior.sample((batch_x.shape[0],))
        return w_kl * self.loss_kl(batch_z) + w_ml * self.loss_ml(batch_x)

    def train_mix(self, x_training_set, z_training_set=None, iter=200, lr=1e-3, batch_size=1024, w_kl=0.5, w_ml=0.5):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        trainloader = data.DataLoader(dataset=x_training_set, batch_size=batch_size,shuffle=True)
        losses = []
        t = 0  # iteration count
        while t < iter:
            for batch_x in trainloader:
                # Custom ML loss function
                loss = self.loss_mix(batch_x=batch_x, w_kl=w_kl, w_ml=w_ml)
                losses.append(loss.item())  # save values for plotting later

                # Training
                optimizer.zero_grad()  # Set grads to zero, else PyTorch will accumulate gradients on each backprop
                loss.backward(retain_graph=True)
                optimizer.step()
                t = t + 1

        return losses

    def train_ML(self, training_data, iter=200, lr=1e-3, batch_size=1024):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        trainloader = data.DataLoader(dataset=training_data, batch_size=batch_size,shuffle=True)

        losses = []
        t = 0  # iteration count
        while t < iter:
            for batch in trainloader:
                # Custom ML loss function
                loss = self.loss_ml(batch)
                losses.append(loss.item())  # save values for plotting later

                # Training
                optimizer.zero_grad()  # Set grads to zero, else PyTorch will accumulate gradients on each backprop
                loss.backward(retain_graph=True)
                optimizer.step()
                t = t + 1

        return losses

    def train_KL(self, training_data=None, training_data_size=1000, iter=200, lr=1e-3, batch_size=1024):
        if training_data == None:
            training_data = self.prior.sample((training_data_size,))
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        trainloader = data.DataLoader(dataset=training_data, batch_size=batch_size,shuffle=True)
        losses = []
        t = 0  # iteration count
        while t < iter:
            for batch in trainloader:
                # Custom ML loss function
                loss = self.loss_kl(batch)
                losses.append(loss.item())  # save values for plotting later

                # Training
                optimizer.zero_grad()  # Set grads to zero, else PyTorch will accumulate gradients on each backprop
                loss.backward(retain_graph=True)
                optimizer.step()
                t = t + 1

        return losses