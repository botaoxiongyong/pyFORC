# /usr/bin/env python
# --*--coding:UTF-8--*--
'''
this is for load raw FORC measurement file by Micromag
'''
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import leastsq, curve_fit

def load_file():
    path = '/Users/pro/Documents/python/FORC_project/'
    file = 'MSM33-55-1_d151_5.forc'
    data = np.genfromtxt(path+file, skip_footer=1, skip_header=34, delimiter=',')
    H = [data[i][0] for i in np.arange(len(data))]
    M = [data[i][1] for i in np.arange(len(data))]
    return H, M

def plot_raw_froc():
    global a

    H, M = load_file()
    data_interval_H=[]
    data_interval_M=[]
    #H_a = []
    H_b = []
    for i in np.arange(1,len(H)):
        data_interval_H.append(H[i])
        data_interval_M.append(M[i])
        if abs(H[i]-H[0])<=0.001:
            H_a = data_interval_H[0]
            a = H_a
            #H_a.append(data_interval_H[0])
            data_interval_H.pop(-1)
            data_interval_M.pop(-1)
            H_b= data_interval_H
            if len(H_b) >6:
                y_poly=poly_value(H_b,data_interval_M)
                plt.plot(H_b, y_poly)
                #params = curve_data(H_b, data_interval_M)
                #M_fit = func(H_b, params)
                #plt.scatter([a]*len(data_interval_H), data_interval_H, s=0.1)
                #plt.plot(data_interval_H,M_fit,c='k',linewidth=0.1)
                '''
                for t in H_b:
                    print a,t
                    plt.scatter((t-a)/2, (a+t)/2, c=params[5], cmap='coolwarm')
                '''
            data_interval_H=[]
            data_interval_M=[]
    #plt.ylim(-0.0000002, 0.0000002)
    #print len(H_a)
    #plt.scatter([H_a]*len(data_interval_H), data_interval_H, s=0.1)
    #for i in np.arange(len(H_a)):

    plt.show()

def poly_value(H_b,data_interval_M):
    b = H_b
    popt = poly_fit(H_b,data_interval_M)
    x1, x2, x3, x4, x5, x6 = popt
    print x1, x2, x3, x4, x5, x6
    y=[]
    for i in np.array(b):
        value =x1+x2*a+ x3*sqr(a)+ x4*i+ x5*sqr(i)+ x6*a*i
        y.append(value)
    return y


def poly_fit(H_b,data_interval_M):
    b = H_b
    M = data_interval_M
    popt, pcov = curve_fit(poly_func, b, M)
    return popt

def poly_func(b, x1, x2, x3, x4, x5, x6):
    b2 = [i*i for i in b]
    b3 = [a*i for i in b]
    return x1+x2*a+ x3*sqr(a)+ x4*b+ x5*b2+ x6*b3





def curve_data(H_b,data_interval_M):
    b = H_b
    M = data_interval_M
    params = [1,1,1,1,1,1]
    plsqr = leastsq(residual, params, args=(b, M), maxfev=500)
    print plsqr[0]
    return plsqr[0]

def func(b,params):
    x1, x2, x3, x4, x5, x6 = params
    b2 = [i*i for i in b]
    b3 = [a*i for i in b]
    return x1+x2*a+ x3*sqr(a)+ x4*b+ x5*b2+ x6*b3

def residual(params,b,M):
    return M-func(b, params)

def sqr(x):
    return x*x

def main():
    plot_raw_froc()

if __name__ == '__main__':
    main()
