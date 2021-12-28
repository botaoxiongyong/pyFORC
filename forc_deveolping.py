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
import matplotlib.colors as mcolors
import pandas as pd
from scipy.interpolate import griddata
import time

class Fit(object):
    def __init__(self,irData=None,SF=None,fileAdres=None):
        '''
        #=================================================
        /process the raw data
        /do the fit
        #=================================================
        '''
        self.dfraw = dataLoad(fileAdres).rawData()
        #print(self.dfraw)
        self.SF = SF

        self.fit()

    def fit(self):
        #SF=5
        #print(self.matrix_z.shape)
        #test
        #self.test_fit(SF = SF, )


        dfmatrix = self.dfraw.pivot(index="ha",
                                    columns="hb",
                                    values="hm")
        #print(dfmatrix.iloc[0])
        #dfmatrix = dfmatrix.loc[:,~(dfmatrix.iloc[0]==np.nan).any()]
        s = dfmatrix.iloc[0]

        si = [i for i in range(s.shape[0]) if np.isnan(s.iloc[i])]

        dfmatrix = dfmatrix.drop(dfmatrix.columns[si],axis=1)

        #print(dfmatrix)
        #"""
        dfmatrix = dfmatrix.interpolate(method='linear',
                                        limit_direction='forward',
                                        limit=20,
                                        axis=0,
                                        )
        #"""
        dfmatrix = dfmatrix.iloc[::-1]

        #dfmatrix.to_csv("dfmatrix.csv")
        matrix_z = dfmatrix.values
        #print(matrix_z.shape)
        plt.imshow(matrix_z)
        plt.show()
        #print(matrix_z)

        """
        X = np.linspace(-150,150,100)
        Y = np.linspace(-150,150,100)
        yi,xi = np.mgrid[-150:150:100j,-150:150:100j]
        #Z = matplotlib.mlab.griddata(df.x,df.y,df.z,X,Y,interp='linear')
        #zi = griddata((df.x,df.y),df.z,(xi,yi), method='linear')#!!! must linear

        zi=griddata((self.dfraw.hb,self.dfraw.ha),self.dfraw.hm,(xi,yi),method='linear')
        plt.pcolormesh(yi.T,xi.T,zi)
        plt.show()
        """


        #self.test_fit(self.SF,X,Y,zi)
        self.test_fit(self.SF,dfmatrix.columns.tolist(),dfmatrix.index.tolist(),matrix_z)







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

        for m,n in itertools.product(np.arange(len(y_range),step=1),np.arange(len(x_range),step=1)):
            #print(m,n)
            if ~np.isnan(matrix_z.item(m,n)):

                #for s in [-1,0,1]: #forc to select data around
                try:
                    s=0
                    grid_data = []
                    a_ = x_range[n]
                    b_ = y_range[m]
                    #for i in np.arange(2*SF+1):
                    #    for j in np.arange(2*SF+1):
                    #if ~np.isnan(matrix_z.item(m+SF,n+SF)) and ~np.isnan(matrix_z.item(m+SF,n-SF)):
                    for i,j in itertools.product(np.arange(-SF,SF+1,step=1), np.arange(-SF,SF+1,step=1)):
                        #print(i,j)
                        try:
                            if ~np.isnan(matrix_z.item(m+i,n+j)):
                                grid_data.append([y_range[m+j],x_range[n+i],matrix_z.item(m+i,n+j)])
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
                    #print(len(grid_data))
                    x,y,z = test_lmf(grid_data)
                    try:
                        p = d2_func(x,y,z)
                        #print(p)
                        if ~np.isnan(p):
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
        #df = df[df['x']>=0]
        '''
        #=================================================
        /due to the space near Bc = zero
        /the Bi values when Bc <0.003 will be mirrored to -Bc
        #=================================================
        '''
        df_negative = df[(df.x<3)].copy()
        df_negative['x'] = df_negative['x'].apply(lambda x: x*-1)
        df_negative.x = df_negative.x*-1
        df = df.append(df_negative)
        df = df.drop_duplicates(['x','y'])
        df = df.sort_values('x')

        #df.z = reject_outliers(data=df.z)

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
        self.xi,self.yi = np.mgrid[0:np.max(df.x):300j,np.min(df.y):np.max(df.y):300j]
        #self.xi,self.yi = np.mgrid[0:0.2:400j,-0.15:0.15:400j]
        z = df.z/np.max(df.z)
        z = np.asarray(z.tolist())

        df['z'] = [abs(i) for i in z]#z
        #df = df[(df['z'] > df['z'].mean()-df['z'].std()) & (df['z'] < df['z'].mean()+df['z'].std())]

        #plt.hist(z,bins=100)
        plt.hist(df.z,bins=100)
        plt.show()
        #print(df.z)
        plt.scatter(df.x,df.y,c=df.z)
        plt.show()

        df = df[df['x']>0]
        

        #zi = griddata((df.x,df.y),df.z,(self.xi,self.yi), method='cubic')
        dfm = df.pivot(index="y",columns="x",values="z")
        dfm.to_csv('./dfm.csv')

        plt.imshow(dfm.values)
        plt.show()

        plt.pcolormesh(dfm.columns.tolist(),dfm.index.tolist(),dfm.values)#vmin=np.min(rho)-0.2)
        plt.show()
        #self.Z = zi
        """

        for m in np.arange(5,len(self.xi)-5):
            xarray = zi[:,m]
            xnum = np.count_nonzero(~np.isnan(zi[:,m]))
            print(self.xi[:,0][xnum:-1],zi[:,m][xnum:-1])
            f = np.polyfit(self.xi[:,0][xnum:-1],zi[:,m][xnum:-1],30)
            for i in np.arange(len(self.xi[:,0][xnum:-1])):
                fx = np.polyval(f,self.xi[:,0][xnum:-1][i])
                zi[xnum+i,m] = fx
        """

        self.Z = zi
        #self.plot()
        #self.Z = griddata((df.x,df.y),z,(self.xi,self.yi), method='cubic')
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
    #def plot(self):
        fig = plt.figure(figsize=(6,5), facecolor='white')
        fig.subplots_adjust(left=0.18, right=0.97,
                        bottom=0.18, top=0.9, wspace=0.5, hspace=0.5)
        #ax = fig.add_subplot(1,1,1)
        norm = mcolors.Normalize(vmin=df['z'].mean()-df['z'].std(),vmax=df['z'].mean()+df['z'].std())
        plt.contour(self.xi,self.yi,self.Z,9,colors='k',linewidths=0.5,norm=norm)#mt to T
        #plt.pcolormesh(X,Y,Z_a,cmap=plt.get_cmap('rainbow'))#vmin=np.min(rho)-0.2)
        plt.pcolormesh(self.xi,self.yi,self.Z,cmap=plt.get_cmap('rainbow'),norm=norm)#vmin=np.min(rho)-0.2)
        plt.colorbar()
        #plt.xlim(0,0.15)
        #plt.ylim(-0.1,0.1)
        plt.xlabel('B$_{c}$ (mT)',fontsize=12)
        plt.ylabel('B$_{i}$ (mT)',fontsize=12)

        plt.show()

def reject_outliers(data, m=1):
    list = []
    for i in data:
        if abs(i-np.mean(data))<m*np.std(data):
            list.append(i)
        else:
            list.append(0)
    return list





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
        #self.rawData()
        self.fileAdres = fileAdres
        #self.matrix()
        #self.rawDataPlot()
        #self.initial()
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
    def rawData(self):
        fileAdres = self.fileAdres
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
        H0 = df.H.max() # the maximum field
        self.x,self.y,self.z=[[],[],[]]
        for i in np.arange(0,len(H)):
            dataInterval_H.append(int(H[i]*10000)/10)
            dataInterval_M.append(M[i])
            if abs(H[i]-H0)<0.002 and len(dataInterval_M)>1:
                #print(dataInterval_H[1:])
                Ha=dataInterval_H[1]

                self.x.extend(dataInterval_H[1::])
                self.y.extend([Ha]*len(dataInterval_H[1::]))
                self.z.extend(dataInterval_M[1::])

                dataInterval_H=[]
                dataInterval_M=[]

        dfraw = pd.DataFrame({"hb":self.x,
                              "ha":self.y,
                              "hm":self.z}).drop_duplicates(
                                  subset=["hb","ha"],
                                  )
        #dfmatrix = dfraw.pivot(index="ha",columns="hb",values="hm")
        #print(dfraw.ha,dfraw.hb)
        #pd.plotting.scatter_matrix(dfmatrix)
        #plt.show()
        #dfraw.to_csv("./dfraw.csv")

        dfraw = dfraw[dfraw["ha"] != dfraw["hb"]]

        #dfraw.to_csv("./dfraw.csv")

        plt.scatter(self.x,self.y,c=self.z)
        plt.show()




        return dfraw


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
        M.append(i[2])#np.arrayi([i[2] for i in data], dtype=np.float64)
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
    start_time = time.time()
    #dataLoad("./example/MSM33-60-1-d416_2.irforc")
    Fit(fileAdres="./example/MSM33-60-1-d416_2.irforc",SF=4)
    end_time = time.time()
    print(end_time - start_time)

if __name__ == '__main__':
    main()
