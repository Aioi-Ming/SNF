import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib

def plot_dimer_energy_with_data(model, x, axis=None):
    counts, bins = np.histogram(x, bins = 200 )
    anchors = (bins[1:] + bins[:-1]) / 2
    probs = counts / np.sum(counts)

    anchors = anchors[np.where(probs > 0.0001)]
    probs = probs[np.where(probs > 0.0001)]

    f = -np.log(probs)
    fn = f - np.min(f)
    plt.scatter(anchors, fn)
    """ Plots the dimer energy to the standard figure """
    d_scan = np.linspace(0.5, 2.5, 100)
    E_scan = model.dimer_energy_distance(d_scan)
    E_scan -= E_scan.min()

    if axis is None:
        axis = plt.gca()
    #plt.figure(figsize=(5, 4))
    axis.plot(d_scan, E_scan, linewidth=2)
    axis.set_xlabel('x / a.u.')
    axis.set_ylabel('Energy / kT')
    axis.set_ylim(E_scan.min() - 2.0, E_scan[int(E_scan.size / 2)] + 2.0)


    return d_scan, E_scan


def plot_forward_backward_2d(model, x, num_samples=10000):
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize = (12,10))

    # plot data sampled in real space 
    plt.subplot(221)
    ax=plt.gca()
    ax.hist2d(x[:,0],x[:,1],bins=100,norm=matplotlib.colors.LogNorm())
    plt.title(r'$a) x \sim \mu_X$')

    # sample from x and transform to z 
    zb=model.backward_flow(torch.from_numpy(x))[0].detach().numpy()
    plt.subplot(222)
    ax=plt.gca()
    ax.hist2d(zb[:,0],zb[:,1],bins=100,norm=matplotlib.colors.LogNorm())
    plt.title(r'$b) z = f(x)$')

    # sampling from gaussian and transform to x
    z, x, _ = model.sample(num_samples)
    #plot gaussian
    plt.subplot(223)
    ax=plt.gca()
    ax.hist2d(z[:,0],z[:,1],bins=100,norm=matplotlib.colors.LogNorm())
    #plt.scatter(z[:, 0], z[:, 1])

    plt.title(r'$c) z \sim \mu_Z$')

    # plot x transformed from gaussian
    plt.subplot(224)
    ax=plt.gca()
    ax.hist2d(x[:,0],x[:,1],bins=100,norm=matplotlib.colors.LogNorm())
    plt.title(r'$d) x = g(z)$')

    plt.show()