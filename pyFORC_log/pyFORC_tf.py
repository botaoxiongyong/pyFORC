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
                    skiprows=i+2
                    break
                else:
                    pass
        skiprows = skiprows if isinstance(skiprows,int) else 1
        df = pd.read_csv(fileAdres, skiprows=skiprows, sep='\s+',
                         delimiter=',', names=['H','M'], skipfooter=1,
                         engine='python')
        df['H'] = df['H']*1000
        df['H'] = df['H'].astype(np.int)
        index = []
        for i in np.arange(1,len(df['H'])):
            if df['H'][i] == df['H'][i-1]:
                index.append(i)
            else:
                pass

        df = df.drop(df.index[index])
        H,M = df['H'].values,df['M'].values
        satField = df['H'][0]
        H_tf_seg,H_rem_seg,M_tf_seg,M_rem_seg = [],[],[],[]
        self.x,self.y,self.z=[[],[],[]]

        for i in np.arange(1,len(H)-1):
            if abs(H[i]) <= 1:
                H_rem_seg.append(H[i-1])
                M_rem_seg.append(M[i])
            else:
                H_tf_seg.append(H[i])
                M_tf_seg.append(M[i])
            if H[i] >= satField-2:
                H_tf_a = H_tf_seg[0]
                #print(H_tf_seg)
                for t in range(len(H_tf_seg)):
                    self.x.append(H_tf_seg[t]/1000)
                    self.y.append(H_tf_a/1000)
                    self.z.append(M_tf_seg[t])

                H_tf_seg,H_rem_seg,M_tf_seg,M_rem_seg = [],[],[],[]

        self.rawdf = df
        #plt.scatter(self.x,self.y,c=self.z/np.max(self.z))
        #plt.show()
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

        zi=griddata((self.x,self.y),self.z,(xi,yi),method='linear')
        #plt.scatter(xi,yi,c=zi)#vmin=np.min(rho)-0.2)
        #plt.show()
        self.matrix_z = zi
        self.x_range=X
        self.y_range=Y

def main():
    #start_time = time.time()
    fileAdres = sys.argv[1]
    SF = int(sys.argv[2])
    SF = SF if isinstance(SF,int) else 5 #defualt SF=5
    #fileAdres='./ps97-085-3-d472_9.irforc'
    Fit(dataLoad_tf(fileAdres),SF).plot()
    #dataLoad(fileAdres)
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
