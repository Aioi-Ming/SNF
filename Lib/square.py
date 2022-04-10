import numpy as np
import torch
import matplotlib.pyplot as plt

class Square(object):
    
    
    def __init__(self, n_square = 2, energy_array = None, xlim=[-90,90], ylim=[-90,90], dx=0.5, dy=0.5 ):
        self.n_square=n_square
        self.dim=2*(n_square-1)
        self.energy_array= energy_array
        self.energy_array_torch=torch.tensor(energy_array)
        self.xlim=xlim
        self.ylim=ylim
        self.dx=dx
        self.dy=dy
        self.max_energy = 1e10
        assert len(energy_array) - 1 == int((xlim[1]-xlim[0])/dx), f"Size error"
        
    def energy_dimer(self, z):
        assert np.ndim(z)==2
        x=z[:,0]
        y=z[:,1]
        
        dxdy_inv=1/self.dx/self.dy
        out_box=np.logical_or.reduce( [x < self.xlim[0], x >self.xlim[1], y<self.ylim[0], y>self.ylim[1]]).astype('float32')
        
        energy=np.zeros(len(z))
        
        i_x = ((x - self.xlim[0])/ self.dx).astype('int')
        i_y = ((y - self.ylim[0])/ self.dy).astype('int')
        i_x-=(x==self.xlim[1]).astype('int')
        i_y-=(y==self.ylim[1]).astype('int')
        x_i = self.xlim[0] + i_x * self.dx 
        y_i = self.ylim[0] + i_y * self.dy
        w11 = (x_i + self.dx - x)* (y_i + self.dy -y)*dxdy_inv
        w12 = (x_i + self.dx - x)* (y - y_i) *dxdy_inv
        w21 = (x- x_i)* (y_i + self.dy -y) *dxdy_inv
        w22 = (x- x_i)* (y - y_i) *dxdy_inv
        
        i_x = i_x * (1-out_box) + out_box * 44
        i_y = i_y * (1-out_box) + out_box * 44
        
        energy= w11 * self.energy_array[i_x,i_y]+w12 * self.energy_array[i_x,i_y+1] + w21* self.energy_array[i_x+1,i_y] + w22 * self.energy_array[i_x+1,i_y+1]
        
        return energy*(1-out_box)+ out_box*self.max_energy
        
        
    def energy_dimer_torch(self,z):
        assert z.dim()==2
        x=z[:,0]
        y=z[:,1]
        
        dxdy_inv=1/self.dx/self.dy
        
        out_box_x=torch.logical_or( x < self.xlim[0], x >self.xlim[1])
        out_box_y=torch.logical_or( y < self.ylim[0], y >self.ylim[1])
        out_box =torch.logical_or( out_box_x, out_box_y)
        
        energy=torch.zeros(len(z))

        i_x = ((x - self.xlim[0])/ self.dx).int()
        i_y = ((y - self.ylim[0])/ self.dy).int()
        i_x-=(x==self.xlim[1]).int()
        i_y-=(y==self.ylim[1]).int()
        
        i_x = i_x *~out_box + out_box * 44
        i_y = i_y *~out_box + out_box * 44
        
        x_i = self.xlim[0] + i_x * self.dx 
        y_i = self.ylim[0] + i_y * self.dy
        w11 = (x_i + self.dx - x)* (y_i + self.dy -y)*dxdy_inv
        w12 = (x_i + self.dx - x)* (y - y_i) *dxdy_inv
        w21 = (x- x_i)* (y_i + self.dy -y) *dxdy_inv
        w22 = (x- x_i)* (y - y_i) *dxdy_inv
        
        
        energy=w11 * self.energy_array_torch[i_x,i_y]+w12 * self.energy_array_torch[i_x,i_y+1] + w21 * self.energy_array_torch[i_x+1,i_y] + w22* self.energy_array_torch[i_x+1,i_y+1]    

        return energy*~out_box+ out_box*self.max_energy 
     
    def energy(self,z):
        
        assert np.ndim(z)==2
        
        k= int((self.n_square-1)*self.n_square/2)
        energy = np.zeros(len(z))
        for i in range(self.n_square):
            if i==0:
                for j in range(1, self.n_square):
                    energy += self.energy_dimer(z[:,2*j-2:2*j])
            else:
                for j in range(i + 1 , self.n_square ):
                    energy += self.energy_dimer(z[:,2*j-2:2*j]-z[:,2*i-2:2*i])
        return energy

    def energy_torch(self,z):
        assert z.dim()==2
        
        k= int((self.n_square-1)*self.n_square/2)
        energy=torch.zeros(len(z))
        for i in range(self.n_square):
            if i==0:
                for j in range(1, self.n_square):
                    energy += self.energy_dimer_torch(z[:,2*j-2:2*j])
            else:
                for j in range(i + 1 , self.n_square ):
                    energy += self.energy_dimer_torch(z[:,2*j-2:2*j]-z[:,2*i-2:2*i])
        return energy    


    
