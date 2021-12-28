# /usr/bin/env python
# --*--coding:UTF-8--*--
'''
this is for load raw FORC measurement file by Micromag
'''
import itertools
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import leastsq, curve_fit
from lmfit.models import GaussianModel, PolynomialModel
from lmfit import minimize, Parameters, Parameter, report_fit
from pylab import pcolor, matplotlib
from mpl_toolkits.mplot3d import Axes3D

def load_file():
    path = '/Users/pro/Documents/python/FORC_project/'
    file = 'MSM33-55-1_d151_5.forc'
    data = np.genfromtxt(path+file, skip_footer=1, skip_header=34, delimiter=',')
    H = [data[i][0] for i in np.arange(len(data))]
    M = [data[i][1] for i in np.arange(len(data))]
    return H, M

def plot_raw_froc():
    #global a
    bounds = np.arange(-50, 50, 10)
    norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=100)
    H, M = load_file()
    data_interval_H=[]
    data_interval_M=[]
    #H_a = []
    H_b = []
    x=[]
    y=[]
    z=[]
    for i in np.arange(1,len(H)):
        data_interval_H.append(H[i])
        data_interval_M.append(M[i])
        if abs(H[i]-H[0])<=0.001:

            if len(data_interval_H) >=0 and len(data_interval_H) <=200:
                H_a = data_interval_H[0]
                a = H_a
                #H_a.append(data_interval_H[0])
                data_interval_H.pop(-1)
                data_interval_M.pop(-1)
                H_b= data_interval_H[1:-1]
                H_M= data_interval_M[1:-1]
                #y_poly=poly_value(H_b,data_interval_M)
                #plt.plot(H_b, y_poly)
                #params = lmfit_curve(H_b, H_M)
                #M_fit = lm_func(params, H_b)
                #plt.plot(H_b,M_fit,c='k',linewidth=0.1)
                #plt.scatter(H_b, data_interval_M,linewidths=0, s=1)

                #print params['x6'].value
                for t in np.arange(len(H_b)):
                    x.append('%.3f'%(H_b[t]))
                    y.append('%.3f'%(H_a))
                    z.append(H_M[t])
                    #Axes3D.plot_surface((t-a)/2, (a+t)/2, params['x6'].value)
                    #plt.scatter((t-a)/2, (a+t)/2, c=params['x6'].value, cmap='coolwarm', norm=norm, linewidths=0.1, s=5)
                    #im = pcolor((t-a)/2, (a+t)/2,params['x6'].value, cmap='RdBu' )


            data_interval_H=[]
            data_interval_M=[]

    #plt.scatter(x, y,c=z, linewidths=0.1, s=5)
    #plt.show()
    return x, y, z

def matrix():
    x, y, z = plot_raw_froc()
    data = np.array(zip(x, y, z))
    data = data[np.argsort(data[:,0])]
    #print [i for i in data]
    x_range=list(sorted(set(x)))
    y_range=list(sorted(set(y)))

    matrix_z = np.zeros(shape=(len(x_range),len(y_range)))
    for line in data:
        m = x_range.index(line[0])
        n = y_range.index(line[1])
        matrix_z[m,n]=line[2]
    #print matrix_z


    return matrix_z, data, x_range,y_range

def new_fit():
    matrix_z, data, x_range,y_range = matrix()

    SF=1

    y_array = np.array(y_range).astype(float)
    point_1 = [float(data[-1-2*SF][0]), float(data[-1-2*SF][1])]                                    #the most right data point
    a_1_2 = float(point_1[1]-point_1[0])                                                  #the line between point1 and point 2
    point_2 = [y_array.min()-a_1_2, y_array.min()]                                        #the lower right data point
    point_4 = [y_array.max(),y_array.max()]                             #the upper most data point
    a_1_4 = (point_1[1]-point_4[1])/(point_1[0]-point_4[0])
    b_1_4 =point_1[1]-point_1[0]*a_1_4


    #data_dict={}
    f = open('/Users/pro/Documents/python/FORC_project/grid_data.dat', 'w')
    for m in np.arange(len(x_range)):
        for n in np.arange(len(y_range)):
            if np.float(y_range[n])-np.float(x_range[m])<=-0.01 and np.float(y_range[n])-np.float(x_range[m])-a_1_2>=-0.01 and \
                    np.float(y_range[n])-a_1_4*np.float(x_range[m])-b_1_4<=-0.01 and np.float(y_range[n])-point_2[1]>=-0.01:#m>=SF and n>=SF and m<=len(x_range)-SF and m<=len(y_range)-SF: #matrix_z.item(m,n) !=0 and
                grid_data=[]
                for i in np.arange((2*SF+1)):
                    for j in np.arange((2*SF+1)):
                        print m-SF+i,n-SF+j
                        grid_data.append([x_range[m-SF+i],y_range[n-SF+j],matrix_z.item(m-SF+i,n-SF+j)])
                #print len(grid_data)
                p = lmfit_curve(grid_data)
                print >>f,x_range[m],y_range[n],matrix_z.item(m,n),p




def SF_fit():
    x, y, z = plot_raw_froc()
    data = np.array(zip(x, y, z))
    data = data[np.argsort(data[:,0])]
    #print [i for i in data]
    x_range=list(sorted(set(x)))
    y_range=list(sorted(set(y)))


    SF=3
    data_dict={}
    f = open('/Users/pro/Documents/python/FORC_project/grid_data.dat', 'w')
    count=0
    for line in data:
        grid_data=[]
        print len(data), count
        P_x = line[0]
        P_y = line[1]
        P_z = line[2]
        n = x_range.index(P_x)
        m = y_range.index(P_y)
        if n>SF and m>SF:
            grid_data.append([x_range[n], y_range[m-1], P_zvalue(x_range[n],y_range[m-1],data)])
            grid_data.append([P_x, P_y, P_z])
            grid_data.append([x_range[n], y_range[m+1], P_zvalue(x_range[n], y_range[m+1],data)])
            grid_data.append([x_range[n-1], y_range[m-1], P_zvalue(x_range[n-1],y_range[m-1],data)])
            grid_data.append([x_range[n-1], y_range[m], P_zvalue(x_range[n-1],y_range[m],data)])
            grid_data.append([x_range[n-1], y_range[m+1], P_zvalue(x_range[n-1],y_range[m+1],data)])
            grid_data.append([x_range[n+1], y_range[m-1], P_zvalue(x_range[n+1],y_range[m-1],data)])
            grid_data.append([x_range[n+1], y_range[m], P_zvalue(x_range[n+1],y_range[m],data)])
            grid_data.append([x_range[n+1], y_range[m+1], P_zvalue(x_range[n+1],y_range[m+1],data)])

            p = lmfit_curve(grid_data)
            #density.append([P_x,P_y, P_z,p])
            print >>f, P_x,P_y, P_z,p
            count+=1



def lmfit_curve(grid_data):
    a = np.array([i[1] for i in grid_data], dtype=np.float64)
    b = np.array([i[0] for i in grid_data], dtype=np.float64)
    M = np.array([i[2] for i in grid_data], dtype=np.float64)
    params = Parameters()
    for i in np.arange(1,7):
        params.add('x'+str(i), value=0.1)
    result = minimize(lm_residual, params, args=(a,b, M), method='leastsq')
    #report_fit(result)
    params = result.params
    #x = [params['x'+str(i)].value for i in np.arange(1,7)]
    return params['x6'].value

def lm_residual(params,a, b, M):
    return M - lm_func(params,a,b)

def lm_func(params,a, b):
    x = [params['x'+str(i)].value for i in np.arange(1,7)]
    x1,x2,x3,x4,x5,x6=x
    a2 = np.array([i**2 for i in a], dtype=np.float64)
    b2 = np.array([i**2 for i in b], dtype=np.float64)
    ab = np.array([m*n for m,n in zip(a,b)], dtype=np.float64)
    value =x1+x2*a+ x3*a2+x4*b+x5*b2+x6*ab
    return value



def P_zvalue(x,y,data):
    P_z=0
    for line in data:
        if line[0]==x and line[1]==y:
            #print 'true'
            P_z = line[2]
    return P_z




def sqr(x):
    return x*x

def main():
    #plot_raw_froc()
    #SF_fit()
    new_fit()

if __name__ == '__main__':
    main()
