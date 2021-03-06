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
from pyFORC_log import Fit,dataLoad
from pyFORC_tf import dataLoad_tf
from pyFORC_rem import dataLoad_rem

class dataLoad_t(object):
    '''
    #=================================================
    /process the measured forc data.
    /converte the raw data into matrix
    /with x range and y range
    /empty postion replaced with np.nan
    #=================================================
    '''
    def __init__(self,fileAdres=None, tfraw=None, remraw=None, irraw=None):
        if tfraw==None:
            self.tf_rawData = dataLoad_tf(fileAdres.replace('.irforc','.remforc'))
            #self.rem_rawData = dataLoad_rem(fileAdres)
            self.ir_rawData = dataLoad(fileAdres)
        else:
            self.tf_rawData = tfraw
            #self.rem_rawData = remraw
            self.ir_rawData = irraw
        self.x_range = self.tf_rawData.x_range
        self.y_range = self.tf_rawData.y_range

        #print(self.ir_rawData.matrix_z)
        self.matrix_z = np.subtract(self.ir_rawData.matrix_z,self.tf_rawData.matrix_z)

        #plt.scatter(self.ir_rawData.x,self.ir_rawData.y,c=self.ir_rawData.z)
        #plt.show()
class dataLoad_int(object):
    '''
    #=================================================
    /process the measured forc data.
    /converte the raw data into matrix
    /with x range and y range
    /empty postion replaced with np.nan
    #=================================================
    '''
    def __init__(self,fileAdres=None, tfraw=None, remraw=None, irraw=None):
        if irraw==None:
            self.tf_rawData = dataLoad_tf(fileAdres.replace('.irforc','.remforc'))
            #self.rem_rawData = dataLoad_rem(fileAdres)
            self.ir_rawData = dataLoad(fileAdres)
        else:
            self.rem_rawData = remraw
            #self.rem_rawData = remraw
            self.ir_rawData = irraw
        self.x_range = self.rem_rawData.x_range
        self.y_range = self.rem_rawData.y_range

        #print(self.ir_rawData.matrix_z)
        self.matrix_z = np.subtract(self.ir_rawData.matrix_z,self.rem_rawData.matrix_z)

        #plt.scatter(self.ir_rawData.x,self.ir_rawData.y,c=self.ir_rawData.z)
        #plt.show()

def main():
    #start_time = time.time()
    fileAdres = sys.argv[1]
    SF = int(sys.argv[2])
    SF = SF if isinstance(SF,int) else 5 #defualt SF=5
    #fileAdres='./ps97-085-3-d472_9.irforc'
    Fit(dataLoad_t(fileAdres),SF).plot()
    if fileAdres!='':
        try:
            #Fit(dataLoad_tf(fileAdres),SF).plot()
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
