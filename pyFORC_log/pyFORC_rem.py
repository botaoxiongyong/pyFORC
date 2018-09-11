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
#from pyFORC_log import Fit

class dataLoad_rem(object):
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

        df['H'] = df['H']*1000
        df['H'] = df['H'].astype(np.int)
        index = []
        for i in np.arange(1,len(df['H'])):
            #print(df['H'][i],df['H'][i-1])
            if df['H'][i] == df['H'][i-1]:
                #print('true')
                index.append(i)
            else:
                pass

        df = df.drop(df.index[index])
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
        H,M = df['H'].values,df['M'].values
        satField = df['H'][0]
        H_tf_seg,H_rem_seg,M_tf_seg,M_rem_seg,M_seg = [],[],[],[],[]
        #print(satField)
        self.x,self.y,self.z=[[],[],[]]
        for i in np.arange(1,len(H)-1):
            if H[i] == 0:
                H_rem_seg.append(H[i-1])
                M_rem_seg.append(M[i])
            else:
                H_tf_seg.append(H[i])
                M_tf_seg.append(M[i])
            if H[i] == satField:
                H_rem_a = H_rem_seg[1] if len(H_rem_seg)>1 else H_rem_seg[0]
                #print(H_rem_seg)
                for t in range(len(H_rem_seg)):
                    self.x.append(H_rem_seg[t]/1000)
                    self.y.append(H_rem_a/1000)
                    self.z.append(M_rem_seg[t])
                    #self.z.append(M_seg)

                H_tf_a = H_tf_seg[0]
                H_tf_seg,H_rem_seg,M_tf_seg,M_rem_seg,M_seg = [],[],[],[],[]

        self.rawdf = df


    def matrix(self):
        '''
        #=================================================
        transfer the data set to matrix as len(x)*len(y) with z value
        :return:
        #=================================================
        '''

        X = np.linspace(-0.2,0.3,200)
        Y = np.linspace(-0.2,0.3,200)
        xi,yi = np.mgrid[-0.2:0.3:200j,-0.2:0.3:200j]

        zi=griddata((self.x,self.y),self.z,(xi,yi),method='cubic')
        self.matrix_z =zi
        self.x_range=X
        self.y_range=Y

        #print(xi)

        #plt.matshow(zi)
        #plt.show()

class Fit(object):
    def __init__(self,irData=None,SF=None):
        '''
        #=================================================
        /process the raw data
        /do the fit
        #=================================================
        '''
        self.rawData = irData#dataLoad(fileAdres)
        #self.matrix_z,self.x_range,self.y_range=dataLoad(fileAdres).initial()
        if self.rawData !=None:
            self.fit(SF)
        else:
            pass
    def fit(self,SF=None):
        #SF=5
        #print(self.matrix_z.shape)
        #test
        self.test_fit(SF = SF, x_range = self.rawData.x_range, y_range = self.rawData.y_range,
                 matrix_z = self.rawData.matrix_z)
    def test_fit(self,SF, x_range, y_range, matrix_z):
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
        #for m in np.arange(0,len(x_range),step=6):
        #    for n in np.arange(0,len(y_range)):
        for m,n in itertools.product(np.arange(0,len(x_range),step=2*SF+1),np.arange(0,len(y_range),step=2*SF+1)):
            a_=float(x_range[m]) # = Ha
            b_=float(y_range[n]) # = Hb
            if abs(a_-b_) < 0.0001: # Ha nearly equal Hb
                m0.append(m)
                n0.append(n)
        #print(len(m0))

        #print(x_range[0],y_range[0],matrix_z.item(1,1))

        #for m in m0:
        #    for n in n0:
        for m,n in itertools.product(m0,n0):
            #for s in [-1,0,1]: #forc to select data around
            try:
                grid_data = []
                a_ = x_range[m]
                b_ = y_range[n]
                #for i in np.arange(2*SF+1):
                #    for j in np.arange(2*SF+1):
                for i,j in itertools.product(np.arange(2*SF+1), np.arange(2*SF+1)):
                    try:
                        grid_data.append([x_range[m+i],y_range[n-j],matrix_z.item(m+i,n-j)])
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
        df_negative = df[(df.x<0.003)].copy()
        #df_negative['x'] = df_negative['x'].apply(lambda x: x*-1)
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
        #X = np.linspace(0,0.15,400)
        #Y = np.linspace(-0.1,0.1,400)

        self.xi,self.yi = np.mgrid[0:0.2:400j,-0.15:0.15:400j]
        z = df.z/np.max(df.z)
        z = np.asarray(z.tolist())
        #Z = matplotlib.mlab.griddata(df.x,df.y,z,X,Y,interp='linear')
        #matplotlib 2.2 was expiered
        #plt.scatter(df.x,df.y,c=z)
        #plt.show()
        self.Z = griddata((df.x,df.y),z,(self.xi,self.yi), method='cubic')
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
    def plot(self):
        fig = plt.figure(figsize=(6,5), facecolor='white')
        fig.subplots_adjust(left=0.18, right=0.97,
                        bottom=0.18, top=0.9, wspace=0.5, hspace=0.5)
        #ax = fig.add_subplot(1,1,1)
        plt.contour(self.xi*1000,self.yi*1000,self.Z,9,colors='k',linewidths=0.5)#mt to T
        #plt.pcolormesh(X,Y,Z_a,cmap=plt.get_cmap('rainbow'))#vmin=np.min(rho)-0.2)
        plt.pcolormesh(self.xi*1000,self.yi*1000,self.Z,cmap=plt.get_cmap('rainbow'))#vmin=np.min(rho)-0.2)
        plt.colorbar()
        #plt.xlim(0,0.15)
        #plt.ylim(-0.1,0.1)
        plt.xlabel('B$_{c}$ (mT)',fontsize=12)
        plt.ylabel('B$_{i}$ (mT)',fontsize=12)

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
    #print(coeff)
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
    #fileAdres='./ps97-085-3-d472_9.irforc'
    #dataLoad_rem(fileAdres)
    Fit(dataLoad_rem(fileAdres),SF).plot()
    if fileAdres!='':
        try:
            pass
            #Fit(dataLoad(fileAdres),SF).plot()
        except Exception as e:
            print(e)
            pass
    else:
        print('!input filename and soomth_factor\npyFORC /data_path/forc_file_name.text 5')
    #end_time = time.time()
    #print(end_time - start_time)

if __name__ == '__main__':
    main()
