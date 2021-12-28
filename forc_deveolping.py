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
from scipy import interpolate,ndimage
from scipy.interpolate import griddata
import time

class Fit(object):
    def __init__(self,irData=None,SF=None,fileAdres=None,kind=None):
        '''
        #=================================================
        /process the raw data
        /do the fit
        #=================================================
        '''
        if kind in ['rem']:
            self.dfraw = dataLoad_rem(fileAdres).rem()
        elif kind in ['i']:
            self.dfraw = dataLoad(fileAdres).rawData()
        #print(self.dfraw)
        self.SF = SF

        self.fit()

    def fit(self):
        #------------------------------------------------s-
        ##convert dfraw into matrix

        dfmatrix = self.dfraw.pivot(index="ha",
                                    columns="hb",
                                    values="hm")
        #----------------------end-

        #------------------------------------------------s-
        ##convert dfraw into matrix by using pandas
        n = 100
        nj = 100j
        X = np.linspace(self.dfraw.hb.min(),self.dfraw.hb.max(),n)
        Y = np.linspace(self.dfraw.ha.min(),self.dfraw.ha.max(),n)
        yi,xi = np.mgrid[self.dfraw.ha.min():self.dfraw.ha.max():nj,
                         self.dfraw.hb.min():self.dfraw.hb.max():nj]
        #zi=griddata((self.dfraw.hb,self.dfraw.ha),self.dfraw.hm,(xi,yi),method='linear')
        #interplotate into new matrix
        #dfnew = pd.DataFrame(data=zi,
        #                     index=X,
        #                     columns=Y,
        #                     dtype=np.float64)

        rbfi = interpolate.Rbf(self.dfraw.hb, self.dfraw.ha, self.dfraw.hm,
                               function='thin_plate')
        zi = rbfi(xi,yi)

        #plt.pcolormesh(xi,yi,zi)
        #plt.scatter(self.dfraw.hb, self.dfraw.ha, s=0.1)
        #plt.contour(X,Y,dfnew.values,20,colors='k')

        #plt.show()
        #----------------------end-

        """
        #------------------------------------------------s-
        #test# read rawdata into fit ,without interpolate

        #print(dfmatrix.iloc[0])
        #dfmatrix = dfmatrix.loc[:,~(dfmatrix.iloc[0]==np.nan).any()]
        s = dfmatrix.iloc[0]

        si = [i for i in range(s.shape[0]) if np.isnan(s.iloc[i])]

        dfmatrix = dfmatrix.drop(dfmatrix.columns[si],axis=1)

        #print(dfmatrix)

        #dfmatrix = dfmatrix.interpolate(method='linear',
        #                                limit_direction='forward',
        #                                limit=10,
        #                                axis=0,
        #                                )
        #interpolate pandas dataframe along row

        dfmatrix = dfmatrix.iloc[::-1]
        #reverse along yaxis

        #dfmatrix.to_csv("dfmatrix.csv")
        matrix_z = dfmatrix.values
        #print(matrix_z.shape)
        #plt.imshow(matrix_z)

        plt.pcolormesh(dfmatrix.columns.tolist(),dfmatrix.index.tolist(),dfmatrix)
        plt.show()
        #print(matrix_z)
        #----------------------end-
        """
        """
        #------------------------------------------------s-
        #test# interpolate polynormial by very column

        for column in dfnew:
            s = dfnew[column].dropna()
            #print(s.values)
            if len(s.values)>2:
                f = interpolate.interp1d(s.index.tolist(),
                                         s.values,kind='previous',fill_value='extrapolate')
                #print(X)
                #print(X[0:len(s.values)+5])
                dfnew[column]=f(X)

        plt.pcolormesh(X,Y,dfnew.values)
        plt.contour(X,Y,dfnew.values,20,colors='k')
        plt.show()
        #----------------------end-
        """

        """
        #------------------------------------------------s-
        #test# rotate matrix 45degree,then do interpolate

        xnew,ynew=[],[]
        for i,j in zip (self.dfraw.ha,self.dfraw.hb):
            ynew.append((i+j)/2)
            xnew.append((j-i)/2)

        plt.scatter(xnew,ynew,c=self.dfraw.hm.values)
        plt.show()

        xii,yii = np.mgrid[-20:np.max(xnew):50j,np.min(ynew):np.max(ynew):50j]

        zii = griddata((xnew,ynew),self.dfraw.hm.values,(xii,yii))
        dfnew = pd.DataFrame(data=zii,
                             dtype=np.float64)
        for i, row in dfnew.itercolumns():
            row.interpolate(method='polynomial', order=2)
        zii = dfnew.values
        plt.contour(xii,yii,zii,20,colors='k',linewidths=0.5)
        plt.pcolormesh(xii,yii,zii)

        plt.colorbar()

        plt.show()
        #----------------------end-
        """

        self.test_fit(self.SF,X,Y,zi)
        #self.test_fit(self.SF, dfmatrix.columns.tolist(),
        #              dfmatrix.index.tolist(), dfmatrix.values)
        #self.test_fit(self.SF,xii,yii,zii)

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

        #------------------------------------------------s-
        #calculate forc distribution by every point in
        # matrix have values

        xx,yy,zz=[],[],[]
        pmatrix = np.full((len(y_range),len(x_range)),np.nan)
        #pmagrix recording original forc p values
        for m,n in itertools.product(np.arange(SF,len(y_range)-SF,step=1),
                                     np.arange(SF,len(x_range)-SF,step=1)):
            if ~np.isnan(matrix_z.item(m,n)):
            # skip position have no values
                try:
                    grid_data = []
                    for i,j in itertools.product(np.arange(-SF,SF+1,step=1),
                                             np.arange(-SF,SF+1,step=1)):
                    #iterate in a 3*3 grid
                        try:
                            if ~np.isnan(matrix_z.item(m+i,n+j)):
                                grid_data.append([y_range[m+j],x_range[n+i],matrix_z.item(m+i,n+j)])
                                #grud_data collecting ha,hb,hm from matrix for
                                # one 3x3 grid
                        except Exception as e:
                            pass
                    #------------------------------------------------s-
                    #this is the main func for calculate p values
                    # if p is not nan, then recorded in pmatrix
                    try:
                        p = self.cal_p(data=grid_data)
                        if abs(p)>0:
                            #pmatrix[m][n] = -1*p
                            a_ = x_range[n]
                            b_ = y_range[m]
                            xx.append((a_-b_)/2)
                            yy.append((a_+b_)/2)
                            #xx.append(a_)
                            #yy.append(b_)
                            zz.append(-1*p)

                    except Exception as e:
                        pass
                    #----------------------end-
                except:
                    pass

        #----------------------end-
        #check pmatrix
        #norm = mcolors.Normalize(vmin=-1*10**-11,vmax=10**-11)
        #plt.pcolormesh(x_range,y_range,pmatrix)
        #plt.colorbar()
        #plt.show()

        """
        #------------------------------------------------s-
        # smoothing part based on the give SF factor
        npmatrix = np.full((len(y_range),len(x_range)),np.nan)
        #npmatrix recording the smoothed p
        for m,n in itertools.product(np.arange(len(y_range),step=1),
                                     np.arange(len(x_range),step=1)):
            #print(m,n)
            if ~np.isnan(pmatrix.item(m,n)):
                pgrid = []
                                #if m+SF<len(y_range) and m-SF>0 and n+SF<len(x_range) and n-SF>0:

                for i,j in itertools.product(np.arange(-2*SF,2*SF+1,step=1),
                                             np.arange(-2*SF,2*SF+1,step=1)):
                #iterate 3x3 grid if SF=1, 5x5 grid if SF=2
                    try:
                        if ~np.isnan(pmatrix.item(m+i,n+j)):
                            pgrid.append(pmatrix.item(m+i,n+j))
                            #collecting p values
                    except:
                        pass
                npmatrix[m][n] = np.mean(pgrid)
                #for every point, record them in npmatrix
                a_ = x_range[n]
                b_ = y_range[m]
                xx.append((a_-b_)/2)
                yy.append((a_+b_)/2)
                #xx.append(a_)
                #yy.append(b_)
                zz.append(np.mean(pgrid))
                #xx,yy, rotate distribution for 45 degree
                # and save ha,hb,hm into three lists
        #----------------------end-
        """

        #------------------------------------------------s-
        #plotting for developping
        #print(npmatrix)
        #plt.imshow(npmatrix)
        #plt.show()
        #norm = mcolors.Normalize(vmin=0,vmax=5*10**-11)

        #plt.pcolormesh(x_range,y_range,npmatrix)
        #plt.contour(x_range,y_range,npmatrix,colors='k',linewidths=0.5,norm=norm)
        #plt.colorbar()
        #plt.show()

        #newz = ndimage.filters.gaussian_filter(npmatrix,[0.3,0.3])
        #plt.pcolormesh(x_range,y_range,newz)
        #plt.contour(x_range,y_range,npmatrix,colors='k',linewidths=0.5,norm=norm)
        #plt.colorbar()
        #plt.show()
        #----------------------end-

        '''
        #=================================================
        /the data will be save as pandas dataframe
        /all the data with nan values will be delete be dropna()

        /reset the Bc and Bi range by X,Y
        /use linear interpolate to obtain FORC distribution
        #=================================================
        '''
        df = pd.DataFrame({'x':xx,'y':yy,'z':zz})
        #df = df[(df['z'] > df['z'].mean()-df['z'].std()) & (df['z'] < df['z'].mean()+df['z'].std())]

        """
        #------------------------------------------------s-
        #test for developping
        #df['z'] = [abs(i) for i in z]#z
        #df = df[(df['z'] > df['z'].mean()-df['z'].std()) & (df['z'] < df['z'].mean()+df['z'].std())]
        # for reject abnormal values

        #plt.hist(z,bins=100)
        #plt.hist(df.z,bins=100)
        #plt.show()
        #print(df.z)

        df = df[df['y']<100]
        plt.hist(df.z,bins=100)
        plt.show()

        df = df[(df['z']<0.01) & (df['z']>-0.01)]
        plt.hist(df.z,bins=100)
        plt.show()

        plt.scatter(df.x,df.y,c=df.z)
        plt.xlim(0,df.x.max())
        plt.colorbar()
        plt.show()
        #----------------------end-
        """

        #------------------------------------------------s-
        #using scipy rbf to interpolate empty values 
        # between 0 and ~5mT
        # solve the boundary problem
        #??? occupy memery
        df = df[df['y']<df['y'].max()*0.8]
        #df = df[df['y']>-100]

        # cut off
        n = 100
        nj = 100j
        X = np.linspace(0,np.max(df.x),n)
        Y = np.linspace(np.min(df.y),np.max(df.y),n)
        xi,yi = np.mgrid[0:np.max(df.x):nj,np.min(df.y):np.max(df.y):nj]
        zi = griddata((df.x,df.y),df.z,(xi,yi),method='cubic')

        #rbfi = interpolate.Rbf(df.x,df.y,df.z)
        #zi = rbfi(xi,yi)
        #----------------------end-

        """
        #------------------------------------------------s-
        # test for developping
        # polynomial interpolate for boundary
        # interpolate for every rows

        dfnew = pd.DataFrame(data=zi,
                             index=X,
                             columns=Y,
                             dtype=np.float64)
        plt.pcolormesh(Y,X,dfnew.values)
        plt.show()

        for column in dfnew:
            #dfnew[column]=dfnew[column].interpolate(method='linear',
            #                          limit_direction='backward',)
            s = dfnew[column].dropna()
            #print(dfnew[column][0])
            if len(s.values)>2:
                f = interpolate.interp1d(s.index.tolist(),
                                         s.values,fill_value='extrapolate')
                #print(s.index[2])
                #print(X[0:len(s.index)])
                #print(s.values)
                inp = f(X[0:10])
                #dfnew[column] = f(X)
                for i in np.arange(0,10):
                    dfnew[column][i] = inp[i]

        plt.pcolormesh(xi,yi,dfnew.values)
        plt.contour(xi,yi,dfnew.values,20,colors='k')
        plt.show()
        #----------------------end-
        """
        #zi = ndimage.filters.gaussian_filter1d(zi,1,axis=0)
        #gaussian smooth again
        self.plot(xi,yi,zi)

    def plot(self,x,y,z):
        fig = plt.figure(figsize=(6,5), facecolor='white')
        fig.subplots_adjust(left=0.18, right=0.97,
                        bottom=0.18, top=0.9, wspace=0.5, hspace=0.5)
        ax = fig.add_subplot(1,1,1)
        #norm = mcolors.Normalize(vmin=df['z'].mean()-df['z'].std(),vmax=df['z'].mean()+df['z'].std())
        #norm = mcolors.Normalize(vmin=df['z'].mean()-df['z'].std(),vmax=df['z'].mean()+df['z'].std())
        #norm = mcolors.LogNorm(vmin=df['z'].min(),vmax=df['z'].max())
        #norm=mcolors.SymLogNorm(linthresh=0.03, linscale=0.03,
        #                        vmin=df['z'].min(),vmax=df['z'].max())
        #bounds = [-0.01,-0.005,-0.003,0,0.003,0.005,0.01]#np.linspace(-0.01, 0.01, 10)
        #norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=50)

        plt.contour(x,y,z,9,colors='k',linewidths=0.5)#mt to T
        plt.pcolormesh(x,y,z)#vmin=np.min(rho)-0.2)
        plt.colorbar()
        #plt.xlim(0,0.15)
        #plt.ylim(-0.1,0.1)
        plt.xlabel('B$_{c}$ (mT)',fontsize=12)
        plt.ylabel('B$_{i}$ (mT)',fontsize=12)
        plt.show()
    def cal_p(self, data=None):
        '''
        #=================================================
        /process the grid data
        /convert to list data for poly fitting
        #=================================================
        '''

        """
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
        """
        data = np.array(data)
        a,b,M = data[:,0],data[:,1],data[:,2]
        #print(a.shape,b.shape,M.shape)


        '''
        #=================================================
        /poly fit for every SF grid data
        #=================================================
        '''
        #X, Y = np.meshgrid(a, b, copy=False)
        #X = X.flatten()
        #Y = Y.flatten()
        X,Y = a,b
        A = np.array([np.ones(len(X)), X, X**2, Y, Y**2, X*Y]).T
        Z = np.array(M)
        B = Z.flatten()
        #print(A.shape,B.shape)
        coeff, r, rank, s = np.linalg.lstsq(A, B, rcond=None)
        #coeff = np.linalg.solve(A,B)
        #print(coeff)
        return coeff[5]

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
            dataInterval_H.append(int(H[i]*10000)/10)#T to mt
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

        #plt.scatter(self.x,self.y,c=self.z)
        #plt.show()

        return dfraw

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

    def rawData(self,fileAdres=None):
        #skip skiprows
        skip_from = '    Field         Moment   '
        with open(fileAdres,'rb') as fr:
            for i,line in enumerate(fr,1):
                if skip_from in str(line):
                    skiprows=i+1
                    break
                else:
                    pass
        skiprows = skiprows if isinstance(skiprows,int) else 1
        df = pd.read_csv(fileAdres, skiprows=skiprows, sep='\s+',
                         delimiter=',', names=['H','M'], skipfooter=1,
                         engine='python',
                         skip_blank_lines=False,
                         dtype=np.float)

        #print(df.iloc[12:33])
        df['H'] = df['H']*1000 #T to mt

        rows = len(df.index)
        seg_index=[]
        for index,row in df.iterrows():
            if index<rows-2  and np.isnan(row['H']) and np.isnan(df.loc[index+2]['H']):
                seg_index.append(index+1)


        H_tf_seg,H_rem_seg,M_tf_seg,M_rem_seg = [],[],[],[]
        tf_a,tf_b,tf_m = [],[],[]
        rem_a,rem_b,rem_m = [],[],[]
        #h_a,h_b,h_m=[[],[],[]]

        for i in np.arange(len(seg_index)-1):
            df_seg = df.iloc[seg_index[i]:seg_index[i+1]+1]
            #print(df_seg.index)
            points = ((seg_index[i+1]+1 - seg_index[i])/3 - 1)/2
            #print(points)
            for j in np.arange(points):
                H_tf_seg.append(df_seg.loc[seg_index[i]+5+j*6]['H'])
                M_tf_seg.append(df_seg.loc[seg_index[i]+5+j*6]['M'])

                H_rem_seg.append(df_seg.loc[seg_index[i]+5+j*6]['H'])
                M_rem_seg.append(df_seg.loc[seg_index[i]+8+j*6]['M'])

            #print (H_tf_seg,H_rem_seg,M_tf_seg,M_rem_seg)
            for j in np.arange(points,dtype=np.int):
                tf_a.append(H_tf_seg[j])
                tf_b.append(H_tf_seg[0])
                tf_m.append(M_tf_seg[j])

                rem_a.append(H_rem_seg[j])
                rem_b.append(H_rem_seg[0])
                rem_m.append(M_rem_seg[j])

            H_tf_seg,H_rem_seg,M_tf_seg,M_rem_seg = [],[],[],[]

        self.remdf = self.todf(rem_a,rem_b,rem_m)
        self.tfdf = self.todf(tf_a,tf_b,tf_m)

    def todf(self,x,y,z):
        dfraw = pd.DataFrame({"hb":x,
                              "ha":y,
                              "hm":z}).drop_duplicates(
                                  subset=["hb","ha"],
                                  )
        dfraw = dfraw[dfraw["ha"] != dfraw["hb"]]
        #plt.scatter(x,y,c=z)
        #plt.show()

        return dfraw
    def rem(self):
        return self.remdf
    def tf(self):
        return self.tfdf

def main():
    start_time = time.time()
    #dataLoad("./example/MSM33-60-1-d416_2.irforc")
    #Fit(fileAdres="./example/MSM33-60-1-d416_2.irforc",SF=2)
    #Fit(fileAdres="./example/ps97-085-3-d472_9.irforc",SF=2,kind='i')
    #Fit(fileAdres="./example/MSM33-53-1-d400_5.irforc",SF=2,kind='i')
    Fit(fileAdres="./example/MSM33-55-1-d380.remforc",SF=3,kind="rem")

    end_time = time.time()
    print(end_time - start_time)

if __name__ == '__main__':
    main()
