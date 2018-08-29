#/usr/bin/env python3
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

class Fit(object):
    def __init__(self,fileAdres=None,SF=None):
        '''
        #=================================================
        /process the raw data
        /do the fit
        #=================================================
        '''
        self.matrix_z,self.x_range,self.y_range=dataLoad(fileAdres).initial()
        self.fit(SF)
    def fit(self,SF=None):
        #SF=5
        #print(self.matrix_z.shape)
        #test
        test_fit(SF = SF, x_range = self.x_range, y_range = self.y_range,
                 matrix_z = self.matrix_z)

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
        self.matrix()
        #self.rawDataPlot()
        self.initial()
    def rawDataPlot(self):
        '''
        /plot the measured data
        '''
        #plt.scatter(self.x, self.y,c=self.z,cmap=plt.cm.rainbow,linewidths=0.1, s=5)
        plt.scatter(self.matrix_z)
        plt.show()
    def initial(self):
        '''
        /to transfer the data for fitting
        '''
        return self.matrix_z,self.x_range,self.y_range
    def rawData(self,fileAdres=None):
        #skip skiprows
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
        skiprows = skiprows if isinstance(skiprows,int) else 1
        df = pd.read_csv(fileAdres, skiprows=skiprows, sep='\s+',
                         delimiter=',', names=['H','M'], skipfooter=1,
                         engine='python')
        #print(df)
        #plt.scatter(df['H'],df['M']/df['M'].max())
        #plt.show()
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
        '''
        /x = Hb
        /y = Ha
        /z = Hm
        '''
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
        #plt.scatter(self.x,self.y,c=self.z)
        #plt.show()
    def matrix(self):
        '''
        #=================================================
        transfer the data set to matrix as len(x)*len(y) with z value
        :return:
        #=================================================
        '''
        #df = pd.DataFrame({'x':self.x,'y':self.y,'z':self.z},dtype=np.float)
        #df = df.sort_values(by=['x','y'])
        '''
        #=================================================
        /this is another method for construct matrix
        /since irregular FORC in log space
        #=================================================
        #x_duplicate = df['x'].drop_duplicates().tolist()
        #y_duplicate = df['y'].drop_duplicates().tolist()
        #self.matrix_z = np.zeros(shape=(int(len(x_duplicate)), int(len(y_duplicate))))
        #for i in np.arange(len(x_duplicate)):
        #    dx = df[df.x == x_duplicate[i]]
        #    for j in np.arange(len(y_duplicate)):
        #        if y_duplicate[j] in dx.y.tolist():
        #            dxy = dx[dx.y == y_duplicate[j]]
        #            self.matrix_z[i,j]=dxy['z'].values[0]
        #        else:
        #            self.matrix_z[i,j]=0
        #            pass
        #self.x_range = x_duplicate
        #self.y_range = y_duplicate
        #plt.scatter(df.x,df.y,c=df.z)
        #plt.show()
        #=================================================
        '''
        #df = df.drop_duplicates(['x','y'])
        '''
        #=================================================
        /mesh up the rawdata
        /select the data area by X,Y ranges
        /obtain regular spaced data potins by np.linspace
        /use interplote to caculate the Hm values
        /loop Ha(Y),Hb(X)
        /fill every position with Hm, else with np.nan
        #=================================================
        '''
        X = np.linspace(-0.2,0.3,200)
        Y = np.linspace(-0.2,0.3,200)
        xi,yi = np.mgrid[-0.2:0.3:200j,-0.2:0.3:200j]
        #Z = matplotlib.mlab.griddata(df.x,df.y,df.z,X,Y,interp='linear')
        #zi = griddata((df.x,df.y),df.z,(xi,yi), method='linear')#!!! must linear

        zi=griddata((self.x,self.y),self.z,(xi,yi),method='linear')
        #plt.pcolormesh(xi,yi,zi,cmap=plt.get_cmap('rainbow'))#vmin=np.min(rho)-0.2)
        #plt.show()
        '''
        #=================================================
        /abandon method to creat matrix_z
        /due to matplotlib.mlab.gridata expiered
        /the scipy.gridata can creat matrix,
        /but note the method used have to be 'linear'
        #=================================================
        self.matrix_z = np.zeros(shape=(len(xi),len(yi)))
        for m in np.arange(0,len(xi)):
            for n in np.arange(0,len(yi)):
                if isinstance(zi[m][n],np.NaN):
                    self.matrix_z[n,m]=zi[m][n]
                else:
                    self.matrix_z[n,m]=np.nan
        #=================================================
        '''
        self.matrix_z = zi
        self.x_range=X
        self.y_range=Y

def test_fit(SF, x_range, y_range, matrix_z):
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
    for m in np.arange(0,len(x_range),step=6):
        for n in np.arange(0,len(y_range)):
    for m,n in itertools.product(np.arange(0,len(x_range),step=6),np.arange(0,len(y_range))):
        a_=float(x_range[m]) # = Ha
        b_=float(y_range[n]) # = Hb
        if abs(a_-b_) < 0.0001: # Ha nearly equal Hb
            m0.append(m)
            n0.append(n)

    #for m in m0:
    #    for n in n0:
    for m,n,s in itertools.product(m0,n0,[-1,0,1]):
        #for s in [-1,0,1]: #forc to select data around
        try:
            grid_data = []
            a_ = x_range[m+s]
            b_ = y_range[n-s]
            #for i in np.arange(2*SF+1):
            #    for j in np.arange(2*SF+1):
            for i,j in itertools.product(np.arange(2*SF+1, np.arange(2*SF+1))
                try:
                    grid_data.append([x_range[m+s+i],y_range[n-s-j],matrix_z.item(m+s+i,n-s-j)])
                except Exception as e:
                    pass
            #print(grid_data)
            '''
            #=================================================
            /when SF = n
            /data grid as (2*n+1)x(2*n+1)
            /every grid produce on FORC distritution p
            /the fitting use d2_func
            /test_lmf for process data
            #=================================================
            '''
            x,y,z = test_lmf(grid_data)
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
    df = pd.DataFrame({'x':xx,'y':yy,'z':zz})
    #df = df.replace(0,np.nan)
    df = df.dropna()
    '''
    #=================================================
    /due to the space near Bc = zero
    /the Bi values when Bc <0.003 will be mirrored to -Bc
    #=================================================
    '''
    df_negative = df[(df.x<0.003)].copy()
    #df_negative['x'] = df_negative['x'].apply(lambda x: x*-1)
    df_negative.x = df_negative.x*-1
    df = df.append(df_negative)
    df = df.drop_duplicates(['x','y'])
    df = df.sort_values('x')
    #plt.scatter(df.x,df.y,c=df.z)
    '''
    #=================================================
    /reset the Bc and Bi range by X,Y
    /use linear interpolate to obtain FORC distribution
    #=================================================
    '''
    #X = np.linspace(0,0.15,400)
    #Y = np.linspace(-0.1,0.1,400)
    xi,yi = np.mgrid[0:0.15:400j,-0.1:0.1:400j]
    z = df.z/np.max(df.z)
    z = np.asarray(z.tolist())
    #Z = matplotlib.mlab.griddata(df.x,df.y,z,X,Y,interp='linear')
    #matplotlib 2.2 was expiered
    Z = griddata((df.x,df.y),z,(xi,yi), method='cubic')
    '''
    #=================================================
    /if using log space for X,Y
    /the scipy.gridata has to be used
    #X = np.logspace(np.log10(0.001),np.log10(0.15),100)
    #Y = np.logspace(np.log10(np.min(yy)),np.log10(np.max(yy)),100)
    #X, Y = np.meshgrid(X,Y)
    #points = np.column_stack((df.x.tolist(), df.y.tolist()))
    #Z = griddata(points, z, (X,Y),method='linear')
    #print(Z)
    #=================================================
    '''
    plt.contour(xi,yi,Z,9,colors='k',linewidths=0.5)
    #plt.pcolormesh(X,Y,Z_a,cmap=plt.get_cmap('rainbow'))#vmin=np.min(rho)-0.2)
    plt.pcolormesh(xi,yi,Z,cmap=plt.get_cmap('rainbow'))#vmin=np.min(rho)-0.2)
    plt.colorbar()
    plt.xlim(0,0.15)

    plt.show()
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

def test_lmf(data):
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
    #print(set(a),set(b))
    #print(a.shape,b.shape,M.shape)
    #params = [0.1,0.1,0.1,0.1,0.1,0.1]
    return a, b, M

def main():
    #start_time = time.time()
    fileAdres = sys.argv[1]
    SF = int(sys.argv[2])
    SF = SF if isinstance(SF,int) else 5 #defualt SF=5
    #SF=5
    #Fit(fileAdres,SF)
    #fileAdres='./ps97-085-3-d472_9.irforc'
    if fileAdres!='':
        try:
            Fit(fileAdres,SF)
        except Exception as e:
            print(e)
            pass
    else:
        print('!input filename and soomth_factor\npyFORC /data_path/forc_file_name.text 5')
    #end_time = time.time()
    #print(end_time - start_time)

if __name__ == '__main__':
    main()
