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
import os
import codecs
import numpy as np
import itertools
import pandas as pd
from scipy.interpolate import griddata
import time
import matplotlib
matplotlib.use('Qt5Agg')
from PyQt5 import QtCore
from PyQt5.QtWidgets import (QApplication,QMainWindow,QGridLayout,QSizePolicy,
                             QWidget,QAction,QFileDialog,QPushButton,QTextEdit,
                             QLabel,QLineEdit,QVBoxLayout,QHBoxLayout)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import pyplot as plt
from pyFORC_log import dataLoad,Fit
from pyFORC_rem import dataLoad_rem
from pyFORC_tf import dataLoad_tf
from pyFORC_in import dataLoad_in
from py_FORC_dataLoad import dataLoad_t,dataLoad_int


class Mainwindow(QMainWindow):
    '''
    #====================================================================
    this is PyQt5 GUI
    #====================================================================
    '''
    def __init__(self):
        super().__init__()
        introducion='''pyIRM
        this is for rock magnetic irm acquisition curves decompose

        New features: Now you can manually adjust all the parameters and see
        the results immediately, and afterwards, you could try 'refit' button
        to better refine your fittings.
        '''
        #self.clickCount=0

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.main_widget = QWidget(self)
        self.grid = QGridLayout(self.main_widget)
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)


        self.layout()
        self.connections()
        self.initUI()
        self.show()
    def layout(self):

        self.figure = plt.figure(figsize=(5,4),dpi=100,facecolor='white')
        self.figure.subplots_adjust(left=0.18, right=0.97,
                        bottom=0.2, top=0.9, wspace=0.5, hspace=0.5)

        hbox_RawData = QHBoxLayout()
        hbox1_Forcs = QHBoxLayout()
        hbox2_Forcs = QHBoxLayout()
        vbox_Forcs = QVBoxLayout()
        vbox_irraw = QVBoxLayout()
        vbox_remraw = QVBoxLayout()

        irraw_label = QLabel('iregualar FORC')
        self.load_irraw_PB = QPushButton('Load data file')
        vbox_irraw.addWidget(irraw_label)
        vbox_irraw.addWidget(self.load_irraw_PB)
        vbox_irraw.addStretch(1)
        vbox_irraw.setContentsMargins(10,10,50,10)

        remraw_label = QLabel('rem FORC')
        self.load_remraw_PB = QPushButton('Load data file')
        vbox_remraw.addWidget(remraw_label)
        vbox_remraw.addWidget(self.load_remraw_PB)
        vbox_remraw.addStretch(1)
        vbox_remraw.setContentsMargins(10,10,50,10)


        self.irRaw_plot = FigureCanvas(self.figure)
        self.remRaw_plot = FigureCanvas(self.figure)
        hbox_RawData.addWidget(self.irRaw_plot)
        hbox_RawData.addLayout(vbox_irraw)
        hbox_RawData.addWidget(self.remRaw_plot)
        hbox_RawData.addLayout(vbox_remraw)
        hbox_RawData.setContentsMargins(10,10,10,10)

        self.irforc_plot = FigureCanvas(self.figure)
        vbox_irforc = QVBoxLayout()
        label = QLabel('irFORC\n\nSF=')
        self.ir_SF = QLineEdit('3')
        self.ir_plot_PB = QPushButton('plot')
        vbox_irforc.addWidget(label)
        vbox_irforc.addWidget(self.ir_SF)
        vbox_irforc.addWidget(self.ir_plot_PB)
        vbox_irforc.addStretch(1)

        self.remforc_plot = FigureCanvas(self.figure)
        vbox_remforc = QVBoxLayout()
        label = QLabel('remFORC\n\nSF=')
        self.rem_SF = QLineEdit('3')
        self.rem_plot_PB = QPushButton('plot')
        vbox_remforc.addWidget(label)
        vbox_remforc.addWidget(self.rem_SF)
        vbox_remforc.addWidget(self.rem_plot_PB)
        vbox_remforc.addStretch(1)

        self.inforc_plot = FigureCanvas(self.figure)
        vbox_inforc = QVBoxLayout()
        label = QLabel('inFORC\n\nSF=')
        self.in_SF = QLineEdit('3')
        self.in_plot_PB = QPushButton('plot')
        vbox_inforc.addWidget(label)
        vbox_inforc.addWidget(self.in_SF)
        vbox_inforc.addWidget(self.in_plot_PB)
        vbox_inforc.addStretch(1)

        hbox1_Forcs.addWidget(self.irforc_plot)
        hbox1_Forcs.addLayout(vbox_irforc)
        hbox1_Forcs.addWidget(self.remforc_plot)
        hbox1_Forcs.addLayout(vbox_remforc)
        hbox1_Forcs.addWidget(self.inforc_plot)
        hbox1_Forcs.addLayout(vbox_inforc)
        hbox_RawData.setContentsMargins(10,10,10,10)

        self.tfforc_plot = FigureCanvas(self.figure)
        vbox_tfforc = QVBoxLayout()
        label = QLabel('tfFORC\n\nSF=')
        self.tf_SF = QLineEdit('3')
        self.tf_plot_PB = QPushButton('plot')
        vbox_tfforc.addWidget(label)
        vbox_tfforc.addWidget(self.tf_SF)
        vbox_tfforc.addWidget(self.tf_plot_PB)
        vbox_tfforc.addStretch(1)

        self.tforc_plot = FigureCanvas(self.figure)
        vbox_tforc = QVBoxLayout()
        label = QLabel('tFORC\n\nSF=')
        self.t_SF = QLineEdit('3')
        self.t_plot_PB = QPushButton('plot')
        vbox_tforc.addWidget(label)
        vbox_tforc.addWidget(self.t_SF)
        vbox_tforc.addWidget(self.t_plot_PB)
        vbox_tforc.addStretch(1)

        self.intforc_plot = FigureCanvas(self.figure)
        vbox_intforc = QVBoxLayout()
        label = QLabel('intFORC\n\nSF=')
        self.int_SF = QLineEdit('3')
        self.int_plot_PB = QPushButton('plot')
        vbox_intforc.addWidget(label)
        vbox_intforc.addWidget(self.int_SF)
        vbox_intforc.addWidget(self.int_plot_PB)
        vbox_intforc.addStretch(1)

        hbox2_Forcs.addWidget(self.tfforc_plot)
        hbox2_Forcs.addLayout(vbox_tfforc)
        hbox2_Forcs.addWidget(self.tforc_plot)
        hbox2_Forcs.addLayout(vbox_tforc)
        hbox2_Forcs.addWidget(self.intforc_plot)
        hbox2_Forcs.addLayout(vbox_intforc)
        hbox_RawData.setContentsMargins(10,10,10,10)

        vbox_Forcs.addLayout(hbox_RawData)
        vbox_Forcs.addLayout(hbox1_Forcs)
        vbox_Forcs.addLayout(hbox2_Forcs)


        self.grid.addLayout(vbox_Forcs,1,1,1,1)
        #self.grid.addLayout(hbox1_Forcs,1,1,2,2)
    def connections(self):
        self.load_irraw_PB.clicked.connect(self.load_irraw)
        self.load_remraw_PB.clicked.connect(self.load_remraw)

    def initUI(self):
        self.statusBar()

        openfile = QAction('open',self)
        openfile.triggered.connect(self.showDialog)
        quitAction = QAction('quit',self)
        quitAction.triggered.connect(self.fileQuit)

        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        filename = menubar.addMenu('&File')
        filename.addAction(openfile)
        filename.addAction(quitAction)

        quitname = menubar.addMenu('&Help')
        menubar.addSeparator()
        quitname.addAction(quitAction)

        self.setGeometry(300,300,1200,900)
        self.setWindowTitle('pyFORC')
    def showDialog(self):
        filename=QFileDialog.getOpenFileName(self,'open file','/home/Documents/')
        if filename[0]:
            f = codecs.open(filename[0],'r',encoding='utf-8',errors='ignore')
            with f:
                data=f.read()
                #self.dataDisp.setText(data)
                filePath=filename[0]
                self.workPath=os.path.dirname(filename[0])
            f.close()
        return filePath
    def fileQuit(self):
        sys.exit(app.exec_())
    def load_irraw(self):
        self.ir_file_path = self.showDialog()
        self.ir_rawData = dataLoad(fileAdres=self.ir_file_path)
        self.ir_fitData = Fit(irData=self.ir_rawData,SF=3)
        #irForc = Fit(self.rawDf,SF=3)

        self.figure.clf()
        ax = self.figure.add_subplot(111)
        #ax1.plot(range(5),range(5))
        _plotRaw(ax=ax, data=self.ir_rawData)
        self.irRaw_plot.draw()

        self.ir_plot_buttom()

    def load_remraw(self):
        self.rem_file_path = self.showDialog()
        self.rem_rawData = dataLoad_rem(fileAdres=self.rem_file_path)
        self.rem_fitData = Fit(irData=self.rem_rawData,SF=3)

        self.tf_rawData = dataLoad_tf(fileAdres=self.rem_file_path)
        self.tf_fitData = Fit(irData=self.tf_rawData,SF=3)

        self.in_rawData = dataLoad_in(tfraw=self.tf_rawData,remraw=self.rem_rawData)
        self.in_fitData = Fit(irData=self.in_rawData,SF=3)

        self.t_rawData = dataLoad_t(tfraw=self.tf_rawData,irraw=self.ir_rawData)
        self.t_fitData = Fit(irData=self.t_rawData,SF=3)

        self.int_rawData = dataLoad_int(remraw=self.rem_rawData,irraw=self.ir_rawData)
        self.int_fitData = Fit(irData=self.int_rawData,SF=3)

        self.figure.clf()
        ax = self.figure.add_subplot(111)
        _plotRaw(ax=ax, data=self.rem_rawData)
        self.remRaw_plot.draw()

        self.rem_plot_buttom()
        self.tf_plot_buttom()
        self.in_plot_buttom()
        self.t_plot_buttom()
        self.int_plot_buttom()

    def ir_plot_buttom(self):
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        #ax.plot(range(10),range(10)
        _plotForc(ax=ax,data=self.ir_fitData)
        self.irforc_plot.draw()
    def rem_plot_buttom(self):
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        #ax.plot(range(10),range(10)
        _plotForc(ax=ax,data=self.rem_fitData)
        self.remforc_plot.draw()
    def in_plot_buttom(self):

        self.figure.clf()
        ax = self.figure.add_subplot(111)
        #ax.plot(range(10),range(10)
        _plotForc(ax=ax,data=self.in_fitData)
        self.inforc_plot.draw()
    def tf_plot_buttom(self):
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        #ax.plot(range(10),range(10)
        _plotForc(ax=ax,data=self.tf_fitData)
        self.tfforc_plot.draw()
    def t_plot_buttom(self):
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        #ax.plot(range(10),range(10)
        _plotForc(ax=ax,data=self.t_fitData)
        self.tforc_plot.draw()
    def int_plot_buttom(self):
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        #ax.plot(range(10),range(10)
        _plotForc(ax=ax,data=self.int_fitData)
        self.intforc_plot.draw()









def _plotRaw(ax=None,data=None):
    #ax.text(0.5,0.5,'raw')
    #ax.set_xlabel('Field')
    #ax.set_ylabel('Magnetization')
    ax.scatter(data.x,data.y,c=data.z, s=0.1)
    #plt.show()
def _plotForc(ax=None,data=None):

    ax.contour(data.xi*1000,data.yi*1000,data.Z,9,colors='k',linewidths=0.5)#mt to T
    #plt.pcolormesh(X,Y,Z_a,cmap=plt.get_cmap('rainbow'))#vmin=np.min(rho)-0.2)
    ax.pcolormesh(data.xi*1000,data.yi*1000,data.Z,cmap=plt.get_cmap('rainbow'))#vmin=np.min(rho)-0.2)
    #plt.colorbar()
    #plt.xlim(0,0.15)
    #plt.ylim(-0.1,0.1)
    ax.set_xlabel('B$_{c}$ (mT)',fontsize=12)
    ax.set_ylabel('B$_{i}$ (mT)',fontsize=12)


def main():
    app = QApplication(sys.argv)
    Mwindow = Mainwindow()
    sys.exit(app.exec_())

if __name__=='__main__':
    main()
