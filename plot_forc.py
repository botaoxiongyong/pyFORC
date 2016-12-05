# /usr/bin/env python
# --*--coding:UTF-8--*--
'''
this is for load raw FORC measurement file by Micromag
'''
import itertools
import numpy as np
import scipy.interpolate
from matplotlib import pyplot as plt
from scipy.optimize import leastsq, curve_fit
from lmfit.models import GaussianModel, PolynomialModel
from lmfit import minimize, Parameters, Parameter, report_fit
from pylab import pcolor, matplotlib
from mpl_toolkits.mplot3d import Axes3D

def load_file():
    path = '/Users/pro/Documents/python/FORC_project/'
    file = 'grid_data.dat'
    data = np.loadtxt(path+file)
    H_a = [i[1] for i in data]
    H_b = [i[0] for i in data]
    M = [i[2] for i in data]
    p = [i[3] for i in data]
    return H_a, H_b, p, M

def draw_fig():
    H_a, H_b, p, M = load_file()

    #plt.scatter(H_b,H_a,c=M)
    #plt.show()

    p = [-0.5*i/np.max(p) for i in p]
    bounds = np.arange(-0.001, 0.003, 0.0001)
    norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    print np.mean(p)
    x = (np.array([b-a for a,b in zip(H_a, H_b)]))/2
    y = (np.array([a+b for a,b in zip(H_a, H_b)]))/2
    #sc = plt.scatter(x, y, c=p,cmap='rainbow',norm=norm,linewidths=0.1, s=5)
    #plt.colorbar(sc)
    z=np.array(p)
    xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate
    rbf = scipy.interpolate.Rbf(x, y, z, function='linear')
    zi = rbf(xi, yi)

    #plt.scatter(H_b,H_a,c=M)
    plt.imshow(zi, vmin=z.min(), vmax=z.max(), origin='lower',cmap='rainbow', norm=norm,
               extent=[x.min(), x.max(), y.min(), y.max()])
    plt.show()


def main():
    draw_fig()


if __name__ == '__main__':
    main()