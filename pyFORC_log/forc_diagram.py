#/usr/bin/env python
#--*--coding:utf-8--*--
'''
#=================================================
/this is for process and plot the forc diagrams,
/icluding the conventional and irregualar forc.

/author: Jiabo
/GFZ potsdam
#=================================================
'''
import sys
import numpy as np
import itertools
from matplotlib import pyplot as plt
import pandas as pd
from scipy.interpolate import griddata
import time

class Forc(object):
    def __init__(self,irData=None,fileAdres=None,SF=None):
        '''
        #=================================================
        /process the raw data
        /do the fit
        #=================================================
        '''
        self.rawData = dataLoad(fileAdres)
        #self.matrix_z,self.x_range,self.y_range=dataLoad(fileAdres).initial()
        if irData !=None:
            self.rawData = irData#dataLoad(fileAdres)
        else:
            self.rawData = dataLoad(fileAdres)

        self.fit(SF = SF,
                 x_range = self.rawData.x_range,
                 y_range = self.rawData.y_range,
                 matrix_z = self.rawData.matrix_z)
    def fit(self,SF, x_range, y_range, matrix_z):

        '''
        #=================================================
        /the main fitting process
        /xx,yy,zz = Hb,Ha,p
        /p is the FORC distribution
        /m0,n0 is the index of values on Ha = Hb
        /then loop m0 and n0
        /based on soomth factor(SF)
        /select data grid from the matrix_z for curve fitting
        #=================================================
        '''
        xx,yy,zz=[],[],[]
        m0,n0 = [],[]
        for m,n in itertools.product(np.arange(0,len(x_range),step=SF),np.arange(0,len(y_range),step=SF)):
            if x_range[m]>y_range[n]: # Ha nearly equal Hb
                m0.append(m)
                n0.append(n)

        aa,bb,cc = [],[],[]
        for m,n in zip(m0,n0):
            s=0
            try:
                grid_data = []
                a_ = x_range[m+s]
                b_ = y_range[n-s]
                for i,j in itertools.product(np.arange(3*SF+1), np.arange(3*SF+1)):
                    try:
                        grid_data.append([x_range[m+s+i],y_range[n-s-j],matrix_z.item(n-s-j,m+s+i)])
                    except:
                        try:
                            for i,j in itertools.product(np.arange(3), np.arange(3)):
                                grid_data.append([x_range[m+i],y_range[n-j],matrix_z.item(n-j,m+i)])
                        except:
                            pass
                #print(grid_data)
                '''
                #=================================================
                /when SF = n
                /data grid as (2*n+1)x(2*n+1)
                /grid_list: convert grid to list
                /every grid produce on FORC distritution p
                /the poly fitting use d2_func
                #=================================================
                '''
                x,y,z = grid_list(grid_data)
                try:
                    p = d2_func(x,y,z)
                    #print(p)
                    xx.append((a_-b_)/2)
                    yy.append((a_+b_)/2)
                    zz.append(p)
                except Exception as e:
                    #print(e)
                    pass
            except:
                pass

        '''
        #=================================================
        /the data will be save as pandas dataframe
        /all the data with nan values will be delete be dropna()
        #=================================================
        '''
        #print(zz)
        df = pd.DataFrame({'x':xx,'y':yy,'z':zz})
        #df = df.replace(0,np.nan)
        df = df.dropna()
        '''
        #=================================================
        /due to the space near Bc = zero
        /the Bi values when Bc <0.003 will be mirrored to -Bc
        #=================================================
        '''
        df_negative = df[(df.x<0.03)].copy()
        df_negative.x = df_negative.x*-1
        df = df.append(df_negative)
        df = df.drop_duplicates(['x','y'])
        df = df.sort_values('x')
        #plt.scatter(df.x,df.y,c=df.z)
        #plt.show()
        '''
        #=================================================
        /reset the Bc and Bi range by X,Y
        /use linear interpolate to obtain FORC distribution
        #=================================================
        '''
        xrange = [0,int((np.max(df.x)+0.05)*10)/10]
        yrange = [int((np.min(df.y)-0.05)*10)/10,int((np.max(df.y)+0.05)*10)/10]
        X = np.linspace(xrange[0],xrange[1],200)
        Y = np.linspace(yrange[0],yrange[1],200)
        self.yi,self.xi = np.mgrid[yrange[0]:yrange[1]:200j,xrange[0]:xrange[1]:200j]

        #self.xi,self.yi = np.mgrid[0:0.2:400j,-0.15:0.15:400j]
        z = df.z/np.max(df.z)
        z = np.asarray(z.tolist())
        self.zi = griddata((df.x,df.y),z,(self.xi,self.yi), method='cubic')

    def plot(self):
        fig = plt.figure(figsize=(6,5), facecolor='white')
        fig.subplots_adjust(left=0.18, right=0.97,
                        bottom=0.18, top=0.9, wspace=0.5, hspace=0.5)
        #ax = fig.add_subplot(1,1,1)
        plt.contour(self.xi*1000,self.yi*1000,self.zi,9,colors='k',linewidths=0.5)#mt to T
        #plt.pcolormesh(X,Y,Z_a,cmap=plt.get_cmap('rainbow'))#vmin=np.min(rho)-0.2)
        plt.pcolormesh(self.xi*1000,self.yi*1000,self.zi,cmap=plt.get_cmap('rainbow'))#vmin=np.min(rho)-0.2)
        plt.colorbar()
        #plt.xlim(0,0.15)
        #plt.ylim(-0.1,0.1)
        plt.xlabel('B$_{c}$ (mT)',fontsize=12)
        plt.ylabel('B$_{i}$ (mT)',fontsize=12)

        plt.show()

class dataLoad(object):
    '''
    #=================================================
    /process the measured forc data.
    /converte the raw data into matrix
    /with x range and y range
    /empty postion replaced with np.nan
    #=================================================
    '''
    def __init__(self,fileAdres=None):
        self.rawData(fileAdres)

    def rawData(self,fileAdres=None):
        #skip skiprows
        skiprows = None
        skip_from = '    Field         Moment   '
        with open(fileAdres,'rb') as fr:
            #f = fr.read()
            for i,line in enumerate(fr,1):
                #print(line)
                if skip_from in str(line):
                    skiprows=i+2
                    break
                #else:
                #    print('file format wrong, cannot find the data row.')
        skiprows = 34 if skiprows==None else skiprows
        df = pd.read_csv(fileAdres, skiprows=skiprows, sep='\s+',
                         delimiter=',', names=['H','M'], skipfooter=1,
                         engine='python')

        H = df.H    #measured field
        M = df.M    #measured magnetic moment
        '''
        #=================================================
        /datainterval_H/_M
        /slice the measured data into pieces
        /for every measured FORC
        #=================================================
        '''
        dataInterval_H=[]
        dataInterval_M=[]
        #print(H)
        cretia = df.H.mean()##edge of linear programing for selecting data
        H0 = df.H.max() # the maximum field
        self.x,self.y,self.z=[[],[],[]]
        for i in np.arange(1,len(H)):
            dataInterval_H.append(H[i])
            dataInterval_M.append(M[i])
            if abs(H[i]-H0)<=0.001: #when the filed reach the max, a new forc
                if len(dataInterval_H)>=0 and len(dataInterval_H)<=200:
                    #print(dataInterval_H)
                    Ha=dataInterval_H[0]
                    dataInterval_H.pop(-1)
                    dataInterval_M.pop(-1)
                    Hb=dataInterval_H[1:-1]
                    Hm=dataInterval_M[1:-1]
                    for t in np.arange(len(Hb)):
                        self.x.append(Hb[t])
                        self.y.append(Ha)
                        self.z.append(Hm[t])
                        #print(Ha)
                dataInterval_H=[]
                dataInterval_M=[]
        self.rawdf = df
        '''
        #=================================================
        transfer the data set to matrix as len(x)*len(y) with z value
        /mesh up the rawdata
        /select the data area by X,Y ranges
        /obtain regular spaced data potins by np.linspace
        /use interplote to caculate the Hm values
        /loop Ha(Y),Hb(X)
        /fill every position with Hm, else with np.nan
        #=================================================
        '''
        self.z = self.z/np.max(self.z)
        #print(int(np.min(self.x)*100)/100,np.max(self.x))
        xrange = [int((np.min(self.x)-0.1)*10)/10,int((np.max(self.x)+0.1)*10)/10]
        yrange = [int((np.min(self.y)-0.1)*10)/10,int((np.max(self.y)+0.1)*10)/10]
        X = np.linspace(xrange[0],xrange[1],200)
        Y = np.linspace(yrange[0],yrange[1],200)
        yi,xi = np.mgrid[yrange[0]:yrange[1]:200j,xrange[0]:xrange[1]:200j]

        #X = np.linspace(-0.2,0.3,200)
        #Y = np.linspace(-0.2,0.3,200)
        #xi,yi = np.mgrid[-0.2:0.3:200j,-0.2:0.3:200j]

        zi=griddata((self.x,self.y),self.z,(xi,yi),method='linear') #!!! must linear
        self.matrix_z = zi
        self.x_range=X
        self.y_range=Y

def d2_func(x, y, z):
    '''
    #=================================================
    /poly fit for every SF grid data
    #=================================================
    '''
    X, Y = np.meshgrid(x, y, copy=False)
    X = X.flatten()
    Y = Y.flatten()
    A = np.array([np.ones(len(X)), X, X**2, Y, Y**2, X*Y]).T
    Z = np.array(z)
    B = Z.flatten()
    #print(A.shape,B.shape)
    coeff, r, rank, s = np.linalg.lstsq(A, B, rcond=None)
    return -coeff[5]

def grid_list(data):
    '''
    #=================================================
    /process the grid data
    /convert to list data for poly fitting
    #=================================================
    '''
    a=[]
    b=[]
    M=[]
    for i in data:
        a.append(i[0]) #np.array([i[1] for i in data], dtype=np.float64)
        b.append(i[1])#np.array([i[0] for i in data], dtype=np.float64)
        M.append(i[2])#np.array([i[2] for i in data], dtype=np.float64)
    a = np.array(a, dtype=np.float64).tolist()
    b = np.array(b, dtype=np.float64).tolist()
    M = np.array(M, dtype=np.float64).tolist()
    a = list(set(a))
    b = list(set(b))
    return a, b, M

def main():
    #start_time = time.time()
    fileAdres = sys.argv[1]
    SF = int(sys.argv[2])
    SF = SF if isinstance(SF,int) else 5 #defualt SF=5
    #fileAdres='./ps97-085-3-d472_9.irforc'
    #Fit(dataLoad(fileAdres),SF).plot()
    if fileAdres!='':
        try:
            Forc(fileAdres=fileAdres,SF=SF).plot()
            pass
        except Exception as e:
            print(e)
            pass
    else:
        print('!input filename and soomth_factor\npyFORC /data_path/forc_file_name.text 5')
    #end_time = time.time()
    #print(end_time - start_time)

if __name__ == '__main__':
    main()
