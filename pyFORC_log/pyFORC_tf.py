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
from pyFORC_log import Fit

class dataLoad_tf(object):
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
            for i,line in enumerate(fr,1):
                if skip_from in str(line):
                    skiprows=i+1
                    break
                else:
                    pass
        skiprows = skiprows if isinstance(skiprows,int) else 1
        df = pd.read_csv(fileAdres, skiprows=skiprows, sep='\s+|,',
                        names=['H','M'], skipfooter=1,
                         engine='python',encoding='ISO-8859-15',
                         skip_blank_lines=False,
                         dtype=np.float)

        #print(df.iloc[12:33])

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

        self.x = [int(i*1000) for i in tf_a]
        self.y = [int(i*1000) for i in tf_b]
        self.z = tf_m


    def matrix(self):
        '''
        #=================================================
        transfer the data set to matrix as len(x)*len(y) with z value
        :return:
        #=================================================
        '''

        self.z = self.z/np.max(self.z)

        plt.scatter(self.x,self.y,c=self.z)
        plt.show()

        #xi = np.array([self.x,]*200).transpose()
        #yi = np.array([self.y,]*200)'
        df = pd.DataFrame({'x':self.x,'y':self.y,'z':self.z})
        df = df.set_index(['x','y','z'],append=True)
        df = df.pivot(index='y',columns='x',values='z')
        print(df.columns)

        plt.pcolormesh(self.x,self.y,self.z,cmap=plt.get_cmap('rainbow'))
        plt.show()

        #X = np.linspace(np.min(self.x),np.max(self.x),200)
        #Y = np.linspace(np.min(self.y),np.max(self.y),200)


        '''
        x1 = np.logspace(-3,np.log10(np.max(self.x)),num=100)
        x2 = -1*x1
        x = np.concatenate([x2[::-1],x1])
        xi = np.array([x,]*200).transpose()
        #print(x1)
        y1 = np.logspace(-3,np.log10(np.max([np.max(self.y),abs(np.min(self.y))])),num=100)
        y2 = -1*y1
        y = np.concatenate([y2[::-1],y1])
        yi = np.array([y,]*200)
        #print(yi)
        '''
        #xi = np.array([self.x,]*len(self.x)).transpose()
        #yi = np.array([self.y,]*len(self.y))
        '''
        xi,yi = np.mgrid[np.min(self.x):np.max(self.x):500j,np.min(self.y):np.max(self.y):500j]
        self.x_range=[i[0] for i in xi]
        self.y_range=yi[0]
        zi=griddata((self.x,self.y),self.z,(xi,yi),method='linear')
        self.matrix_z = zi
        #xt,yi = np.mgrid[-0.159:0.159:200j,-0.159:0.133:200j]
        '''

        '''

        for m in np.arange(0,len(xi)):
            #print(m)
            xarray = zi[:,m]
            xnum = np.count_nonzero(np.isnan(zi[:,m]))-1
            f = np.polyfit(self.x_range[xnum:-1],zi[:,m][xnum:-1],1)
            fx = np.poly1d(f)
            for j in np.arange(2,10,step=1):
                if xnum-j >0:
                    zi[xnum-j,m] = fx(self.x_range[xnum-j])

        '''
        '''
        f = np.polyfit(self.y_range[0:2],zi[1,:][0:2],1)
        fx = np.poly1d(f)

        #zi[1,4] = fx(self.y_range[2])

        for m in np.arange(1,len(xi)-1):
            #print(m)
            yarray = zi[m,:]
            ynum = np.count_nonzero(~np.isnan(zi[m,:]))
            f = np.polyfit(self.y_range[0:ynum],zi[m,:][0:ynum],2)
            fx = np.poly1d(f)
            for j in np.arange(0,100,step=1):
                if ynum+j < len(self.y_range):
                    #pass
                    if j==0 and np.isnan(zi[m,ynum]):
                        zi[m,ynum] = fx(self.y_range[ynum+j])
                        pass
                    else:
                        zi[m,ynum+j] = fx(self.y_range[ynum+j])


        '''
        plt.pcolormesh(xi,yi,zi,cmap=plt.get_cmap('rainbow'))
        plt.show()


def main():
    #start_time = time.time()
    fileAdres = sys.argv[1]
    SF = int(sys.argv[2])
    SF = SF if isinstance(SF,int) else 5 #defualt SF=5
    #fileAdres='./ps97-085-3-d472_9.irforc'
    #Fit(dataLoad_tf(fileAdres),SF).plot()
    dataLoad_tf(fileAdres)
    if fileAdres!='':
        try:
            pass
            #dataLoad(fileAdres)
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
