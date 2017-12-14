#/usr/bin/env python3
#--*--coding:utf-8--*--
import numpy as np
from numba import jit
from matplotlib import pyplot as plt
from lmfit import minimize, Parameters
import pandas as pd
import time
import threading
import multiprocessing
import forcFortran as f
class dataLoad(object):
    def __init__(self):
        self.rawData()
        self.matrix()
        self.initial()
    def rawDataPlot(self):
        plt.scatter(self.x, self.y,c=self.z,cmap=plt.cm.rainbow,linewidths=0.1, s=5)
        plt.show()
    def initial(self):
        return self.matrix_z,self.data,self.x_range,self.y_range
    def rawData(self):
        path = '/Users/pro/Documents/python/FORC_project/'
        file = 'MSM33-55-1_d151_5.forc'
        rawdat = np.genfromtxt(path+file, skip_footer=1, skip_header=34, delimiter=',')
        H = [rawdat[i][0] for i in np.arange(len(rawdat))]
        M = [rawdat[i][1] for i in np.arange(len(rawdat))]
        dataInterval_H=[]
        dataInterval_M=[]
        #print(H)
        self.x,self.y,self.z=[[],[],[]]
        for i in np.arange(1,len(H)):
            dataInterval_H.append(H[i])
            dataInterval_M.append(M[i])
            if abs(H[i]-H[0])<=0.001:
                if len(dataInterval_H)>=0 and len(dataInterval_H)<=200:
                    #print(dataInterval_H)
                    Ha=dataInterval_H[0]
                    dataInterval_H.pop(-1)
                    dataInterval_M.pop(-1)
                    Hb=dataInterval_H[1:-1]
                    Hm=dataInterval_M[1:-1]
                    for t in np.arange(len(Hb)):
                        self.x.append('%.3f'%(Hb[t]))
                        self.y.append('%.3f'%(Ha))
                        self.z.append(Hm[t])
                        #print(Ha)
                dataInterval_H=[]
                dataInterval_M=[]
        #plt.scatter(self.x, self.y,c=self.z, linewidths=0.1, s=5) # plot raw data
        #plt.show()
    def matrix(self):
        '''
        transfer the data set to matrix as len(x)*len(y) with z value
        :return:
        '''
        data = list((zip(self.x,self.y,self.z)))
        self.data=np.asarray(data)
        self.x_range=list(sorted(set(self.x)))
        self.y_range=list(sorted(set(self.y)))
        self.matrix_z = np.zeros(shape=(len(self.x_range),len(self.y_range)))
        #plt.scatter(self.x_range, self.y_range, linewidths=0.1, s=5)
        #plt.show()
        for line in data:
            m = self.x_range.index(line[0])
            n = self.y_range.index(line[1])
            self.matrix_z[m,n]=line[2]
@jit
def process(mlist, SF, x_range, y_range, data, matrix_z):
    y_array = np.array(y_range).astype(float)
    point_1 = [float(data[-1-2*SF][0]), float(data[-1-2*SF][1])]                                    #the most right data point
    a_1_2 = float(point_1[1]-point_1[0])                                                  #the line between point1 and point 2
    point_2 = [y_array.min()-a_1_2, y_array.min()]                                        #the lower right data point
    point_4 = [y_array.max(),y_array.max()]                             #the upper most data point
    a_1_4 = float((point_1[1]-point_4[1])/(point_1[0]-point_4[0]))
    b_1_4 = float(point_1[1]-point_1[0]*a_1_4)
    for m in mlist:
        for n in range(len(y_range)):
            x=float(x_range[m])
            y=float(y_range[n])
            #----------------------------
            #线性规划取有效值范围
            if y-x<=-0.01:# and \
                if y-x-a_1_2>=-0.01:# and \
                    if y-a_1_4*x-b_1_4<=-0.01: #and\
                        if y-point_2[1]>=-0.01:
                            grid_data=[]
                            for i in np.arange((2*SF+1)):
                                for j in np.arange((2*SF+1)):
                                    #print(m-SF+i, n-SF+i)
                                    if m-SF+i <= matrix_z.shape[0]-1 and n-SF+j<=matrix_z.shape[1]-1:
                                        grid_data.append([x_range[m-SF+i],y_range[n-SF+j],matrix_z.item(m-SF+i,n-SF+j)])
                                    else:
                                        grid_data.append([10**-6,10**-6,10**-6])
                                        #pass
                            p = lmFit(grid_data).initial()
                            #print(self.x_range[m],self.y_range[n],self.matrix_z[m,n],p)
                            #fitFile.write(str(x_range[m])+' '+str(y_range[n])+' '+str(matrix_z[m,n])+' '+str(p))
                            #fitFile.write('\n')
@jit
def process_2(mlist, SF, x_range, y_range, data, matrix_z):
    for m in mlist:
        for n in range(len(y_range)):
            x=float(x_range[m])
            y=float(y_range[n])
            #----------------------------
            #线性规划取有效值范围
            if y-x<=-0.01:# and \
                grid_data=[]
                for i in np.arange((2*SF+1)):
                    for j in np.arange((2*SF+1)):
                        #print(m-SF+i, n-SF+i)
                        if m-SF+i <= matrix_z.shape[0]-1 and n-SF+j<=matrix_z.shape[1]-1:
                            grid_data.append([x_range[m-SF+i],y_range[n-SF+j],matrix_z.item(m-SF+i,n-SF+j)])
                        else:
                            grid_data.append([10**-6,10**-6,10**-6])
                            #pass
                p = lmFit(grid_data).initial()
                    #print(self.x_range[m],self.y_range[n],self.matrix_z[m,n],p)
                    #fitFile.write(str(x_range[m])+' '+str(y_range[n])+' '+str(matrix_z[m,n])+' '+str(p))
                    #fitFile.write('\n')
def testFortran(SF, x_range, y_range, data, matrix_z, point_1,a_1_2,point_2,point_4,a_1_4,b_1_4):
    out,k = f.forcfortran(SF, x_range, y_range, matrix_z,a_1_2,point_2,a_1_4,b_1_4)
    print(len(out),k)
    t=k-1
    grid_data = out[0:t]
    #print(grid_data.shape)
    #grid_data = np.resize(grid_data, ((int(t/9),9)))
    data=[]
    for i in np.arange(0,t,step=9):
        end = i+9
        data.append(grid_data[i:end])
    data=np.array(data)
    #np.ndarray.reshape(grid_data,((int(t/9),3)))
    #print(grid_data)
    pp=[]
    for grid in data:
        p = lmFit(grid).initial()
        pp.append(p)
    print(pp)

    #fitFile = open('/Users/pro/Documents/python/FORC_project/'+'tempt.dat', 'w+')
    #fitFile.write(str(x_range[m])+' '+str(y_range[n])+' '+str(matrix_z[m,n])+' '+str(p))
                            #fitFile.write('\n')

class mythread(threading.Thread):
    def __init__(self, mlist, SF, x_range, y_range, data, matrix_z):
        threading.Thread.__init__(self)
        self.mlist, self.SF, self.x_range, self.y_range, self.data, self.matrix_z = mlist, SF, x_range, y_range, data, matrix_z
    def run(self):
        process(self.mlist, self.SF, self.x_range, self.y_range, self.data, self.matrix_z)

class mul_prcocess(multiprocessing.Process):
    def __init__(self, mlist, SF, x_range, y_range, data, matrix_z):
        self.mlist, self.SF, self.x_range, self.y_range, self.data, self.matrix_z = mlist, SF, x_range, y_range, data, matrix_z
    def run(self):
        process(self.mlist, self.SF, self.x_range, self.y_range, self.data, self.matrix_z)

def test(SF, x_range, y_range, data, matrix_z):
    x_range=np.array(x_range, dtype=np.float64)
    y_array = np.array(y_range, dtype=np.float64)
    y_range = np.array(y_range, dtype=np.float64)
    point_1 = [np.float64(data[-1-2*SF][0]), np.float64(data[-1-2*SF][1])]                                    #the most right data point
    a_1_2 = np.float64(point_1[1]-point_1[0])                                               #the line between point1 and point 2
    point_2 = [y_array.min()-a_1_2, y_array.min()]                                        #the lower right data point
    point_4 = [y_array.max(),y_array.max()]                             #the upper most data point
    a_1_4 = np.float64((point_1[1]-point_4[1])/(point_1[0]-point_4[0]))
    b_1_4 =np.float64(point_1[1]-point_1[0]*a_1_4)
    testFortran(SF, x_range, y_range, data, matrix_z, point_1,a_1_2,point_2,point_4,a_1_4,b_1_4)


def wast():
    mlist = []
    n = int(len(x_range)/10)
    for i in range(0,n):
        mlist.append(np.arange(10*i, 10+10*i))
    mlist.append(np.arange(10+10*n, len(x_range)))
    ps = []
    #for mmlist in mlist:
    #    print(mmlist)
    for m,i in zip(mlist[0],range(0,10)):
        print(m,i)
        args = ([m], SF, x_range, y_range, data, matrix_z)
        #print(m)
        p = multiprocessing.Process(target = process_2, args =args)
        ps.append(p)
        #p.start()
        #p.join()
    mlist_1 = range(10)
    mlist_2 = range(10, 20)
    #print(mlist_1, mlist_2)
    #p_1 = mul_prcocess(mlist_1, SF, x_range, y_range, data, matrix_z)
    #p_1.start()
    p_1 = multiprocessing.Process(target = process_2, args = (mlist_1, SF,
                                                              x_range, y_range,
                                                              data, matrix_z))
    p_1.start()
    p_2 = multiprocessing.Process(target = process_2, args = (mlist_2, SF,
                                                              x_range, y_range,
                                                              data, matrix_z))
    p_2.start()
    p_1.join()
    p_2.join()



    thread_1 = mythread(mlist_1, SF, x_range, y_range, data, matrix_z)
    thread_2 = mythread(mlist_2, SF, x_range, y_range, data, matrix_z)
    thread_1.start()
    thread_2.start()
    thread_1.join()
    thread_2.join()
    treads = []
    for m in range(len(x_range)):
        treads=threading.append(process(m, SF, x_range, y_range, data, matrix_z))
    for t in treads:
        t.start()


@jit
def Fit_fit(SF, x_range, y_range, data, matrix_z):
    y_array = np.array(y_range).astype(float)
    point_1 = [float(data[-1-2*SF][0]), float(data[-1-2*SF][1])]                                    #the most right data point
    a_1_2 = float(point_1[1]-point_1[0])                                                  #the line between point1 and point 2
    point_2 = [y_array.min()-a_1_2, y_array.min()]                                        #the lower right data point
    point_4 = [y_array.max(),y_array.max()]                             #the upper most data point
    a_1_4 = (point_1[1]-point_4[1])/(point_1[0]-point_4[0])
    b_1_4 =point_1[1]-point_1[0]*a_1_4
    #fitFile = open('/Users/pro/Documents/python/FORC_project/'+'tempt.dat', 'w+')
    treads = []
    for m in range(len(x_range)):
        treads.append()
        for n in range(len(y_range)):
            x=float(x_range[m])
            y=float(y_range[n])
            #----------------------------
            #线性规划取有效值范围
            if y-x<=-0.01:# and \
                if y-x-a_1_2>=-0.01:# and \
                    if y-a_1_4*x-b_1_4<=-0.01: #and\
                        if y-point_2[1]>=-0.01:
                            grid_data=[]
                            for i in np.arange((2*SF+1)):
                                for j in np.arange((2*SF+1)):
                                    #print(m-SF+i, n-SF+i)
                                    if m-SF+i <= matrix_z.shape[0]-1 and n-SF+j<=matrix_z.shape[1]-1:
                                        grid_data.append([x_range[m-SF+i],y_range[n-SF+j],matrix_z.item(m-SF+i,n-SF+j)])
                                    else:
                                        grid_data.append([10**-6,10**-6,10**-6])
                                        #pass
                            p = lmFit(grid_data).initial()
                            #print(self.x_range[m],self.y_range[n],self.matrix_z[m,n],p)
                            #fitFile.write(str(x_range[m])+' '+str(y_range[n])+' '+str(matrix_z[m,n])+' '+str(p))
                            #fitFile.write('\n')
            '''
            try:
                del grid_data
                del x,y
            except:
                pass
            '''
    print ('end')
    #fitFile.close()

class Fit(object):
    def __init__(self):
        self.matrix_z,self.data,self.x_range,self.y_range=dataLoad().initial()
        self.fit()
    def fit(self):
        SF=1
        print(self.matrix_z.shape)
        test(SF = SF, x_range = self.x_range, y_range = self.y_range,
                data=self.data, matrix_z = self.matrix_z)
    def polyFit(self,data):
        return lmFit(data).initial()
#@jit
def lm_func(params,a, b):
    x = []
    for i in np.arange(1, 7):
        x.append(params['x'+str(i)].value)
    #x = [params['x'+str(i)].value for i in np.arange(1,7)]
    x1,x2,x3,x4,x5,x6=x
    a2 = []
    b2 = []
    ab = []
    for i in a:
        a2.append(i**2)
    for i in b:
        b2.append(i**2)
    for m,n in zip(a,b):
        ab.append(m*n)
    a2 = np.array(a2, dtype=np.float64)#np.array([i**2 for i in a], dtype=np.float64)
    b2 = np.array(b2, dtype=np.float64) #np.array([i**2 for i in b], dtype=np.float64)
    ab = np.array(ab, dtype=np.float64) #np.array([m*n for m,n in zip(a,b)], dtype=np.float64)
    value =x1+x2*a+ x3*a2+x4*b+x5*b2+x6*ab
    return value
#@jit
def lmfit_curve(data):
    a=[]
    b=[]
    M=[]
    for i in data:
        a.append(i[1]) #np.array([i[1] for i in data], dtype=np.float64)
        b.append(i[0])#np.array([i[0] for i in data], dtype=np.float64)
        M.append(i[2])#np.array([i[2] for i in data], dtype=np.float64)
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    M = np.array(M, dtype=np.float64)
    params = Parameters()
    for i in np.arange(1,7):
        params.add('x'+str(i), value=0.1)
    return params, a, b, M

class lmFit(object):
    def __init__(self,data=None):
        self.data=data
        self.lmfit_curve()
        del self.data
    def initial(self):
        p= self.params['x6'].value
        del self.params
        return p
    def lmfit_curve(self):
        data = self.data
        params, a, b, M = lmfit_curve(data)
        result = minimize(self.lm_residual, params, args=(a,b, M), method='leastsq')
        self.params = result.params
        #self.params = lmfit_curve(data)
    def lm_residual(self,params,a, b, M):
        return M - self.lm_func(params,a,b)
    def lm_func(self,params,a, b):
        return lm_func(params, a, b)
        #return value
def main():
    start_time = time.time()
    Fit()
    end_time = time.time()
    print(end_time - start_time)

if __name__ == '__main__':
    main()
