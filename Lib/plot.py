import numpy as np
import matplotlib.pyplot as plt

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