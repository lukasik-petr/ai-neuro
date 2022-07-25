#!/usr/bin/python3

#------------------------------------------------------------------------------
# ai-daemon
# (C) GNU General Public License,
# pro potreby projektu TM-AI vyrobil Petr Lukasik , 2022 ");
#------------------------------------------------------------------------------
# program pro projekt TM-AI v TAJMAC-ZPS,a.s.
# predikce vlivu teploty na presnost obrabeni
# Prerekvizity: linux Debian-11 nebo Ubuntu-20.04,);
#               miniconda3,
#               python 3.9,
#               tensorflow 2.8,
#               mathplotlib,
#               scikit-learn-intelex,
#               pandas,
#               numpy,
#               keras,
#
# Windows se pokud mozno vyhnete. Tak je doporuceno v manualech TensorFlow
# i kdyz si myslim ze by to take fungovalo. Ale proc si delat zbytecne
# starosti...
#
# Pozor pro instalaci je nutno udelat nekolik veci
#  1. instalace prostredi miniconda 
#       a. stahnout z webu miniconda3 v nejnovejsi verzi
#       b. chmod +x Miniconda3-py39_4.11.0-Linux-x86_64.sh
#       c. ./Miniconda3-py39_4.11.0-Linux-x86_64.sh
#
#  2. update miniconda
#       conda update conda
#
#  3. vyrobit behove prostredi 'tf' v miniconda
#       conda create -n tf python=3.8
#
#  4. aktivovat behove prostredi tf (preX tim je nutne zevrit a znovu
#     otevrit terminal aby se conda aktivovala.
#       conda activate  tf
#
#  5. instalovat tyto moduly (pomoci conda)
#       conda install tensorflow=2.8 
#       conda install mathplotlib
#       conda install scikit-learn-intelex
#       conda install pandas
#       conda install numpy
#       conda install keras
#
#  6. v prostredi tf jeste upgrade tensorflow 
#       pip3 install --upgrade tensorflow
#------------------------------------------------------------------------------
# import vseho co souvisi s demonem...
import sys, os, getopt; 
import traceback;
import time; 
import atexit; 
import signal; 
import grp;
import daemon; 
import lockfile;

import logging;
from signal import SIGTERM
#import IPython
#import IPython.display
import glob as glob;
import pandas as pd;
import seaborn as sns;
import tensorflow as tf;
import math;
import numpy as np;
import shutil;
import matplotlib as mpl;
import matplotlib.pyplot as plt;
import platform;
from lockfile.pidlockfile import PIDLockFile

from dateutil import parser
from sklearn.preprocessing import MinMaxScaler;
from sklearn.metrics import mean_squared_error;
from sklearn.metrics import max_error;
from sklearn.utils import shuffle
from numpy import asarray;
#from matplotlib import pyplot;
from dataclasses import dataclass;
from datetime import datetime
from tabulate import tabulate
from pathlib import Path

#from tensorflow.keras.datasets import imdb;
from tensorflow.keras import models;
from tensorflow.keras import layers;
from tensorflow.keras import optimizers;
from tensorflow.keras import losses;
from tensorflow.keras import metrics;
from tensorflow.keras import callbacks;
from tensorflow.keras.models import Sequential;
from tensorflow.keras.layers import InputLayer;
from tensorflow.keras.layers import Input;
from tensorflow.keras.layers import Dense;
from tensorflow.keras.layers import LSTM;
from tensorflow.keras.layers import GRU;
from tensorflow.keras.layers import Conv1D;
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model
#from keras.utils.vis_utils import plot_model
from matplotlib import cm;
from _cffi_backend import string
from pandas.core.frame import DataFrame

from scipy.signal import butter, lfilter, freqz

from opcua import ua
from opcua import *

from subprocess import call;
from plistlib import InvalidFileException



#---------------------------------------------------------------------------
# nastaveni globalnich parametru logu pro demona
#---------------------------------------------------------------------------

'''
global logger;
global log_handler;

logger = None;
log_handler = None;

progname = os.path.basename(__file__);

logging.basicConfig(filename="./log/"+progname+".log",
                    filemode='a',
                    level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S');

log_handler = logging.StreamHandler();

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>LOG_HANDLER", log_handler);
logger = logging.getLogger("parent");
logger.addHandler(log_handler)
'''

#---------------------------------------------------------------------------
# DataFactory
#---------------------------------------------------------------------------
class OPCAgent():
    
    # konstrukter    
    def __init__(self):
        self.prefix = "opc.tcp://";
        self.host1   = "opc998.os.zps"; # BR-PLC
        self.port1  = "4840";
        self.host2   = "opc999.os.zps";# HEIDENHANIN-PLC
        self.port2  = "48010";
        self.is_ping = False;
        
    #---------------------------------------------------------------------------
    # myformat         
    #---------------------------------------------------------------------------
    def myformat(self, x):
        
        return ('%.4f' % x).rstrip('0').rstrip('.');


    #---------------------------------------------------------------------------
    # isPing ????         
    #---------------------------------------------------------------------------
    def isPing(self):
        return self.is_ping;
    #---------------------------------------------------------------------------
    # ping         
    #---------------------------------------------------------------------------
    def ping_(self, host):
        
        parameter = '-n' if platform.system().lower()=='windows' else '-c';                                                 
                                                                                                                            
        command = ['ping', parameter, '1', host];                                                                           
        response = call(command);                                                                                           
                                                                                                                            
        if response == 0:    
            self.is_ping = True;                                                                                               
            return self.is_ping;                                                                                                    
        else:                                                                                                               
            self.is_ping = False;                                                                                               
            return self.is_ping;
        
                                                                                                            
        
    #---------------------------------------------------------------------------
    # opcCollectorTemp - nacti teploty
    #---------------------------------------------------------------------------
    def opcCollectorBR_PLC(self):
        
        plc_isRunning = True;
        uri = self.prefix+self.host1+":"+self.port1;
        # tabulka nodu v br plc         
        plc_br_table        = np.array([["temp_ch01",     "ns=6;s=::AsGlobalPV:teplota_ch01"],
                                        ["temp_lo01",     "ns=6;s=::AsGlobalPV:teplota_lo01"],
                                        ["temp_lo03",     "ns=6;s=::AsGlobalPV:teplota_lo03"],
                                        ["temp_po01",     "ns=6;s=::AsGlobalPV:teplota_po01"],
                                        ["temp_pr01",     "ns=6;s=::AsGlobalPV:teplota_pr01"],
                                        ["temp_pr02",     "ns=6;s=::AsGlobalPV:teplota_pr02"],
                                        ["temp_pr03",     "ns=6;s=::AsGlobalPV:teplota_pr03"],
                                        ["temp_sl01",     "ns=6;s=::AsGlobalPV:teplota_sl01"],
                                        ["temp_sl02",     "ns=6;s=::AsGlobalPV:teplota_sl02"],
                                        ["temp_sl03",     "ns=6;s=::AsGlobalPV:teplota_sl03"],
                                        ["temp_sl04",     "ns=6;s=::AsGlobalPV:teplota_sl04"],
                                        ["temp_st01",     "ns=6;s=::AsGlobalPV:teplota_st01"],
                                        ["temp_st02",     "ns=6;s=::AsGlobalPV:teplota_st02"],
                                        ["temp_st03",     "ns=6;s=::AsGlobalPV:teplota_st03"],
                                        ["temp_st04",     "ns=6;s=::AsGlobalPV:teplota_st04"],
                                        ["temp_st05",     "ns=6;s=::AsGlobalPV:teplota_st05"],
                                        ["temp_st06",     "ns=6;s=::AsGlobalPV:teplota_st06"],
                                        ["temp_st07",     "ns=6;s=::AsGlobalPV:teplota_st07"],
                                        ["temp_st08",     "ns=6;s=::AsGlobalPV:teplota_st08"],
                                        ["temp_vr01",     "ns=6;s=::AsGlobalPV:teplota_vr01"],
                                        ["temp_vr02",     "ns=6;s=::AsGlobalPV:teplota_vr02"],
                                        ["temp_vr03",     "ns=6;s=::AsGlobalPV:teplota_vr03"],
                                        ["temp_vr04",     "ns=6;s=::AsGlobalPV:teplota_vr04"],
                                        ["temp_vr05",     "ns=6;s=::AsGlobalPV:teplota_vr05"],
                                        ["temp_vr06",     "ns=6;s=::AsGlobalPV:teplota_vr06"],
                                        ["temp_vr07",     "ns=6;s=::AsGlobalPV:teplota_vr07"],
                                        ["temp_vz02",     "ns=6;s=::AsGlobalPV:teplota_vz02"],
                                        ["temp_vz03",     "ns=6;s=::AsGlobalPV:teplota_vz03"],
                                        ["light_ambient", "ns=6;s=::AsGlobalPV:vstup_osvit"],
                                        ["temp_ambient",  "ns=6;s=::AsGlobalPV:vstup_teplota"],
                                        ["humid_ambient", "ns=6;s=::AsGlobalPV:vstup_vlhkost"]]);

        if not self.ping_(self.host1):
            plc_isRunning = False;
            return(plc_br_table, plc_isRunning);
   
        client = Client(uri)
        try:        
            client.connect();
            sys.stderr.write("Client" + uri+ "Connected\n");
            #logger.info("Client" + uri+ "Connected")
            plc_br_table = np.c_[plc_br_table, np.zeros(len(plc_br_table))];
            
            for i in range(len(plc_br_table)):
                node = client.get_node(str(plc_br_table[i, 1]));
                typ  = type(node.get_value());
                val = self.myformat(node.get_value()) if typ is float else node.get_value();
                plc_br_table[i, 2] = val;
            
            
        except Exception as ex:
            traceback.print_exc();
            sys.stderr.write(str(traceback.print_exc())+"\n");
            #logger.error(traceback.print_exc());
            plc_isRunning = False;
            
        finally:
            client.disconnect();
            return(plc_br_table, plc_isRunning);
 


    #---------------------------------------------------------------------------
    # opcCollectorTemp - nacti PLC HEIDENHAIN
    #---------------------------------------------------------------------------
    def opcCollectorHH_PLC(self):
        
        plc_isRunning = True;
        uri = self.prefix+self.host2+":"+self.port2;
        #tabulka nodu v plc heidenhain
        plc_hh_table        = np.array([["datetime",     ""],
                                        ["tool",         "ns=2;s=Technology data.ACTUAL_TOOL_T"],
                                        ["state",        "ns=2;s=Technology data.PROGRAM_STATE"],
                                        ["program",      "ns=2;s=Technology data.PIECE_PROGRAM"],
                                        ["load_s1",      "ns=2;s=S1.Load"],
                                        ["mcs_s1",       "ns=2;s=S1.MCS"],
                                        ["speed_s1",     "ns=2;s=S1.Motor_speed"],
                                        ["temp_s1",      "ns=2;s=S1.Temperature"],
                                        ["load_x",       "ns=2;s=X.Load"],
                                        ["mcs_x",        "ns=2;s=X.MCS"],
                                        ["speed_x",      "ns=2;s=X.Motor_speed"],
                                        ["temp_x",       "ns=2;s=X.Temperature"],
                                        ["load_y",       "ns=2;s=Y.Load"],
                                        ["mcs_y",        "ns=2;s=Y.MCS"],
                                        ["speed_y",      "ns=2;s=Y.Motor_speed"],
                                        ["temp_y",       "ns=2;s=Y.Temperature"],
                                        ["load_z",       "ns=2;s=Z.Load"],
                                        ["mcs_z",        "ns=2;s=Z.MCS"],
                                        ["speed_z",      "ns=2;s=Z.Motor_speed"],
                                        ["temp_z",       "ns=2;s=Z.Temperature"],
                                        ["dev_datetime1","ns=2;s=Machine data.M0111"],
                                        ["dev_x1",       "ns=2;s=Machine data.M0112"],
                                        ["dev_y1",       "ns=2;s=Machine data.M0113"],
                                        ["dev_z1",       "ns=2;s=Machine data.M0114"],
                                        ["dev_datetime2","ns=2;s=Machine data.M0211"],
                                        ["dev_x2",       "ns=2;s=Machine data.M0212"],
                                        ["dev_y2",       "ns=2;s=Machine data.M0213"],
                                        ["dev_z2",       "ns=2;s=Machine data.M0214"],
                                        ["dev_datetime3","ns=2;s=Machine data.M0311"],
                                        ["dev_x3",       "ns=2;s=Machine data.M0312"],
                                        ["dev_y3",       "ns=2;s=Machine data.M0313"],
                                        ["dev_z3",       "ns=2;s=Machine data.M0314"],
                                        ["dev_datetime4","ns=2;s=Machine data.M0411"],
                                        ["dev_x4",       "ns=2;s=Machine data.M0412"],
                                        ["dev_y4",       "ns=2;s=Machine data.M0413"],
                                        ["dev_z4",       "ns=2;s=Machine data.M0414"],
                                        ["dev_datetime5",""],
                                        ["dev_x5",       ""],
                                        ["dev_y5",       ""],
                                        ["dev_z5",       ""]]);
                                        
                                        
        if not self.ping_(self.host2):
            plc_isRunning = False;
            return(plc_hh_table, plc_isRunning);
                                                    
        client = Client(uri);
        try:        
            client.connect();
            sys.stderr.write("Client" + uri+ "Connected\n");
            #logger.info("Client" + uri+ "Connected")
            
            plc_hh_table = np.c_[plc_hh_table, np.zeros(len(plc_hh_table))];
            
            for i in range(len(plc_hh_table)):
                if "datetime" in plc_hh_table[i, 0]:
                    plc_hh_table[i, 2] = datetime.now().strftime("%Y-%m-%d_%H:%M:%S");
                else:
                    if plc_hh_table[i, 1]:
                        node = client.get_node(str(plc_hh_table[i, 1]));
                        typ  = type(node.get_value());
                        val = self.myformat(node.get_value()) if typ is float else node.get_value();
                        plc_hh_table[i, 2] = val;
            
        except OSError as ex:
            traceback.print_exc();
            sys.stderr.write(str(traceback.print_exc()));
            #logger.error(traceback.print_exc());
            plc_isRunning = False;
            
        except Exception as ex:
            traceback.print_exc();
            sys.stderr.write(str(traceback.print_exc()));
            #logger.error(traceback.print_exc());
            plc_isRunning = False;
            
        finally:
            client.disconnect(); 
            return(plc_hh_table, plc_isRunning);


    #---------------------------------------------------------------------------
    # opcCollectorTemp - nacti PLC HEIDENHAIN
    #---------------------------------------------------------------------------
    def opcCollectorGetPredictData(self):
        
        sys.stderr.write("Nacitam 120 vzorku dat pro predict - v intervalu 1 [s]\n");
        #logger.info("Nacitam 120 vzorku dat pro predict - v intervalu 1 [s]");
        for i in range(120):
            br_plc, plc_isRunning = self.opcCollectorBR_PLC();
            if not plc_isRunning:
                return(None);
            
            hh_plc, plc_isRunning = self.opcCollectorHH_PLC();
            if not plc_isRunning:
                return(None);
        
            hh_plc = np.concatenate((hh_plc, br_plc)).T;
            cols = np.array(hh_plc[0]);
            data = list((hh_plc[2]));
            if i == 0:
                df_predict = pd.DataFrame(columns = cols);
            df_predict.loc[len(df_predict)] = data;
            time.sleep(1.0);

        return(df_predict);

#---------------------------------------------------------------------------
# DataFactory
#---------------------------------------------------------------------------
class DataFactory():

    df_parmX_predict = [];
    
    @dataclass
    class DataTrain:
        model:     object              #model neuronove site
        train:     object              #treninkova mnozina
        valid:     object              #validacni mnozina
        test:      object              #testovaci mnozina
        df_parm_x: string              #mnozina vstup dat (df_parmx, df_parmy, df_parmz
        df_parm_y: string              #mnozina vstup dat (df_parmX, df_parmY, df_parmZ
        axis:      string              #osa X, Y, Z


    @dataclass
    class DataTrainDim:
          DataTrain: object;
          
    @dataclass
    class DataResult:
        # return self.DataResult(x_test, y_test, y_result, mse, mae)
        x_test:    object           #testovaci mnozina v ose x
        y_test:    object           #testovaci mnozina v ose y
        y_result:  object           #vysledna mnozina
        axis:      string           #osa stroje [X,Y, nebo Z]

    @dataclass
    class DataResultDim:
          DataResultX: object;

    def __init__(self, path_to_result, window):
        
    #Vystupni list parametru - co budeme chtit po siti predikovat
        self.df_parmx = ['temp_S1','temp_pr01',
                         'temp_pr02',
                         'temp_pr03',
                         'temp_vr01',
                         'temp_vr02',
                         'temp_vr03',
                         'temp_vr04',
                         'temp_vr05',
                         'temp_vr06',
                         'temp_vr07'];
        
    #Tenzor predlozeny k uceni site
        self.df_parmX = ['dev_x4',
                         'dev_y4',
                         'dev_z4', 
                         'temp_S1',
                         'temp_pr01',
                         'temp_pr02',
                         'temp_pr03',
                         'temp_vr01',
                         'temp_vr02',
                         'temp_vr03',
                         'temp_vr04',
                         'temp_vr05',
                         'temp_vr06',
                         'temp_vr07'];
        
        self.path_to_result = path_to_result;
        self.getParmsFromFile();
        self.window = window;
        self.df_multiplier = 1;
        self.train = pd.DataFrame();
        self.valid = pd.DataFrame();
        self.predict = pd.DataFrame();
        self.opc = OPCAgent();

    #---------------------------------------------------------------------------
    # isPing         
    #---------------------------------------------------------------------------
    def isPing(self):
        return(self.opc.isPing());
    #---------------------------------------------------------------------------
    # myformat         
    #---------------------------------------------------------------------------
    def myformat(self,x):
        return ('%.6f' % x).rstrip('0').rstrip('.');

    #---------------------------------------------------------------------------
    # butterworth Filter         
    #---------------------------------------------------------------------------

    def buttFilter(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    #---------------------------------------------------------------------------
    # butterworth Filter         
    #---------------------------------------------------------------------------
    def buttLowpassFilter(self, data, cutoff, fs, order=5):
        b, a = self.buttFilter(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y


    #---------------------------------------------------------------------------
    # butterworth Filter         
    #---------------------------------------------------------------------------
    def lowpassFilter(self, df, parmx):
        
        order = 6
        fs = 70.0       
        cutoff = 3.667
        
        cols = df.columns;
        for col in cols:
            if col in parmx:
                df[col] = self.buttLowpassFilter(df[col], cutoff, fs, order);
        
        return (df);
        

    #---------------------------------------------------------------------------
    # multDF
    #---------------------------------------------------------------------------
    def multDF(self, df):

        if self.df_multiplier == 1:
            return df;

        df = df.assign(Index=range(len(df))).set_index('Index')

        df_mult = pd.DataFrame();
        
        rows_list = [];
        for ind in df.index:
            for i in range(self.df_multiplier):
                rows_list.append(df.iloc[ind]);
        
        df_mult = pd.DataFrame(rows_list);
        return df_mult;

    #---------------------------------------------------------------------------
    # divDF
    #---------------------------------------------------------------------------
    def divDF(self, df):

        if self.df_multiplier == 1:
            return df;


        #index datetime 
        df['Index'] = df["datetime"];
        df.set_index('Index', inplace=True);

        #mean ve sloupcich "predict"        
        for col in df.columns:
            if "predict" in col:
                df[col] = df[col].groupby(['Index']).mean();
        
        df.reset_index(drop=True);
        #index integer        
        df = df.assign(Index=range(len(df))).set_index('Index')
        df_div = pd.DataFrame();
        
        rows_list = [];
        i = 0;
        for ind in df.index:
                if i < self.df_multiplier:
                    i += 1;
                    
                    if i == self.df_multiplier:
                        rows_list.append(df.iloc[ind]);
                        i = 0;
        
        df_div = pd.DataFrame(rows_list);
        return df_div;
    
    #---------------------------------------------------------------------------
    # setDataX(self, df,  size_train, size_valid, size_test)
    #---------------------------------------------------------------------------
    def setDataX(self, df, df_test,  size_train, size_valid, size_test, txdt_b=False, shuffling=False):
        #OSA XYZ
        try:

            DataTrain_x = self.DataTrain;
            DataTrain_x.train = pd.DataFrame(df[0 : size_train][self.df_parmX]);
            DataTrain_x.valid = pd.DataFrame(df[size_train+1 : size_train + size_valid][self.df_parmX]);
            
            if shuffling:
                DataTrain_x.train = DataTrain_x.train.reset_index(drop=True)
                DataTrain_x.train = shuffle(DataTrain_x.train)
                DataTrain_x.train = DataTrain_x.train.reset_index(drop=True)
                #logger.info("--shuffle = True");
            
            DataTrain_x.test  = df_test;
            DataTrain_x.df_parm_x = self.df_parmx;  # data na ose x, pro rovinu X
            DataTrain_x.df_parm_y = self.df_parmX;  # data na ose y, pro rovinu Y
            DataTrain_x.axis = "OSA_XYZ";
            
            self.train = DataTrain_x.train;
            self.valid = DataTrain_x.valid;
            self.predict = DataTrain_x.test;
            
            return(DataTrain_x);
    
        except Exception as ex:
            traceback.print_exc();
            #logger.error(traceback.print_exc());
            sys.stderr.write(str(traceback.print_exc()));
    
    #---------------------------------------------------------------------------
    # getData
    #---------------------------------------------------------------------------
    def getData(self, shuffling=False, timestamp_start='2022-06-29 05:00:00', timestamp_stop='2022-07-01 23:59:59', type="predict"):
        
        txdt_b  = False;
        df      = pd.DataFrame();
        df_test = pd.DataFrame();
        
        try:
           
            #self.DataTrainDim.DataTrain = None;
            
            #if "train" in type: 
            files = os.path.join("./br_data", "tm-ai_2022*.csv");
            
            # list souboru pro join
            joined_list = glob.glob(files);
            
            # sort souboru pro join
            joined_list.sort(key=None, reverse=False);
            
            df = pd.concat([pd.read_csv(csv_file,
                                         sep=",|;", 
                                         engine='python',  
                                         header=0, encoding="utf-8",
                                       )
                                    for csv_file in joined_list],
                                    axis=0, 
                                    ignore_index=True
                    );
            # bordel pri domluve nazvoslovi...            
            df.columns = df.columns.str.lower();
            # vyber dat dle timestampu
            df["timestamp"] = pd.to_datetime(df["datetime"].str.slice(0, 18));
                
            # treninkova a validacni mnozina    
            df = df[(df["timestamp"] > timestamp_start) & (df["timestamp"] <= timestamp_stop)];
            
            if len(df) <= 1:
                sys.stderr.write("Data pro trenink maji nulovou velikost - exit(0)\n");
                #logger.info("Data pro trenink maji nulovou velikost - exit(0)");
                sys.exit(0);
                
            
            df["index"] = pd.Index(range(0, len(df), 1));
            df.set_index("index", inplace=True);

            size = len(df.index)
            size_train = math.floor(size * 8 / 12)
            size_valid = math.floor(size * 4 / 12)
            size_test  = math.floor(size * 0 / 12)

            # type == 'predict' 
            # predikcni mnozina - pokus zvetsi testovaci mnozinu self.df_multiplier krat...
            df_test = self.opc.opcCollectorGetPredictData();
            
            if df_test is None:
                sys.stderr.write("Patrne nebezi nektery OPC server\n");
                #logger.info("Patrne nebezi nektery OPC server");
                #pro ladeni
                '''
                sys.stderr.write("Jsou nactena ladici data !!!\n");
                df_test = pd.read_csv("./br_data/predict.csv",
                                         sep=",", 
                                         engine='python',  
                                         header=0, encoding="utf-8",
                                       );
                '''                       
            if  not df_test is None and self.window >= len(df_test):
                sys.stderr.write("Prilis maly vzorek dat pro predikci - exit(1)\n");
                #logger.info("Prilis maly vzorek dat pro predikci - exit(1)");
                sys.exit(1);
                    
            if self.df_parmx == None or self.df_parmX == None:
                pass;
            else:
                self.DataTrainDim.DataTrain = self.setDataX( df=df, 
                                                             df_test=df_test, 
                                                             size_train=size_train, 
                                                             size_valid=size_valid, 
                                                             size_test=size_test,
                                                             txdt_b=txdt_b,
                                                             shuffling=shuffling 
                                                        );
            

            return self.DataTrainDim(self.DataTrainDim.DataTrain);

        except Exception as ex:
            traceback.print_exc();
            sys.stderr.write(str(traceback.print_exc()));
            #logger.error(traceback.print_exc());
            
    #-----------------------------------------------------------------------
    # saveDataResult  - result
    #-----------------------------------------------------------------------
    def saveDataResult(self, timestamp_start, model, typ, saveresult=True):
        
        filename = "./result/predicted_"+model+".csv"
        
        if "train" in typ:
            saveresult=True;
            
        
        if not saveresult:
            sys.stderr.write("Vystupni soubor " + filename + " nevznikne !!!, saveresult = " +str(saveresult)+"\n");
            return;
        else:
            sys.stderr.write("Vystupni soubor " + filename + " vznikne.\n");
        
        try:
            col_names_y = list(self.DataTrain.df_parm_y);
            col_names_x = list(self.DataTrain.df_parm_x);
            
            col_names_predict = list("");
            col_names_drop    = list("");
            col_names_drop2   = list("");
            col_names_train   = list({"datetime"});
            col_names_dev     = list("");
            
            for col in col_names_y:
                col_names_train.append(col);
            
            for i in range(len(col_names_y)):
                if "dev" in col_names_y[i]:
                    col_names_predict.append(col_names_y[i]+"_predict");
                    col_names_dev.append(col_names_y[i]);
                else:    
                    col_names_drop.append(col_names_y[i]);

            for col in self.DataTrain.test.columns:
                if col in col_names_train:
                    a = 0;
                else:    
                    col_names_drop2.append(col);
                    
            
        except Exception as ex:
            traceback.print_exc();
            sys.stderr.write(str(traceback.print_exc()));
            #logger.error(traceback.print_exc());

            
        try:
            self.DataTrain.test.reset_index(drop=True, inplace=True)
            
            df_result = pd.DataFrame();
            df_result  = pd.DataFrame(self.DataResultDim.DataResultX.y_result, columns = col_names_y);
            df_result.drop(col_names_drop, inplace=True, axis=1);
            df_result  = pd.DataFrame(np.array(df_result), columns =col_names_predict);
            
            df_result2 = pd.DataFrame();
            df_result2 = pd.DataFrame(self.DataTrain.test);
            df_result2.drop(col_names_drop2, inplace=True, axis=1);

            df_result  = pd.concat([df_result2, df_result], axis=1);
            df_result["txdat_group"] = timestamp_start;
            
            df_result = self.divDF(df_result);
            
            # Absolute Error
            for col in col_names_dev:
                ae = (df_result[col] - df_result[col+"_predict"]);
                df_result[col+"_ae"] = ae;
            # Mean Squared Error
            for col in col_names_dev:
                mse = mean_squared_error(df_result[col],df_result[col+"_predict"]);
                df_result[col+"_mse"] = mse;
            
            path = Path(filename)

            if path.is_file():
                append = True;
            else:
                append = False;
        
            if append:             
                sys.stderr.write(f"\nSoubor {filename} existuje - append: "+ str(len(df_result)) + " vet\n");
                df_result.to_csv(filename, mode = "a", index=True, header=False, float_format='%.5f');
            else:
                sys.stderr.write(f"\nSoubor {filename} neexistuje - create: "+ str(len(df_result))+"\n");
                df_result.to_csv(filename, mode = "w", index=True, header=True, float_format='%.5f');

        except Exception as ex:
            traceback.print_exc();
            #logger.error(traceback.print_exc());
            sys.stderr.write(str(traceback.print_exc()));
    
        return;    
        
    #-----------------------------------------------------------------------
    # getParmsFromFile - nacte parametry z ./parms/parms.txt
    #-----------------------------------------------------------------------
    def getParmsFromFile(self):
        # otevreno v read mode
        
        parmfile = "ai-parms.txt"
        try:
            file = open(parmfile, "r")
            lines = file.readlines();
            
            count = 0
            for line in lines:
                line = line.replace("\n","").replace("'","").replace(" ","");
                line = line.strip();
                #if not line:
                x = line.startswith("df_parmx=")
                if x:
                    line = line.replace("df_parmx=", "").lower();
                    self.df_parmx = line.split(",");
                    if "null" in line:
                        self.df_parmx = None;
                        
                X = line.startswith("df_parmX=");
                if X:
                    line = line.replace("df_parmX=", "").lower();
                    self.df_parmX = line.split(",");
                    if "null" in line:
                        self.df_parmX = None;
            
                
            file.close();
            sys.stderr.write("parametry nacteny z "+ parmfile +"\n");       
            #logger.info("parametry nacteny z "+ parmfile);                 
            
                
        except:
            sys.stderr.write("Soubor parametru "+ parmfile + " nenalezen!\n");                
            sys.stderr.write("Parametry pro trenink site budou nastaveny implicitne v programu\n");                 
            #logger.info("Soubor parametru " + parmfile + " nenalezen!");
        
        return();  
    
    #---------------------------------------------------------------------------
    # DataFactory
    #---------------------------------------------------------------------------
    def prepareParmsPredict(self):
        
        i = 0;
        for i in self.df_parmX:
            df_parmX_predict[i] = self.df_parmX[i]+"_predict";
        i = 0;
             


    #---------------------------------------------------------------------------
    # DataFactory getter metody
    #---------------------------------------------------------------------------
    def getDf_parmx(self):
        return self.df_parmx;
    
    def getDf_parmX(self):
        return self.df_parmX;

    def getDfTrainData(self):
        return (self.train);
    
    def getDfValidData(self):
        return (self.valid);
    
    def getDfPredictData(self):
        return (self.predict);

#---------------------------------------------------------------------------
# Neuronova Vrstava DENSE
#---------------------------------------------------------------------------
class NeuronLayerDENSE():
    #definice datoveho ramce
    
    @dataclass
    class DataSet:
        X_dataset: object              #data k uceni
        y_dataset: object              #vstupni data
        cols:      int                 #pocet sloupcu v datove sade

    def __init__(self, path_to_result, typ, model, epochs, batch, txdat1, txdat2, window, units=256, shuffling=True, actf="tanh"):
        
        self.path_to_result = path_to_result; 
        self.typ = typ;
        self.model = model;
        self.epochs = epochs;
        self.batch = batch;
        self.txdat1=txdat1;
        self.txdat2=txdat2;
        
        self.df = pd.DataFrame();
        self.df_out = pd.DataFrame();
        self.graph = None;
        self.window = window;
        self.units = units;
        self.shuffling = shuffling;
        self.actf = actf;
        self.data = DataFactory(path_to_result=self.path_to_result, window=self.window);

    #---------------------------------------------------------------------------
    # isPing ????
    #---------------------------------------------------------------------------
    def isPing(self):
        return self.data.isPing();

    #---------------------------------------------------------------------------
    # smoothGraph - trochu vyhlad graf
    #---------------------------------------------------------------------------
    def smoothGraph(self, points, factor=0.9):
        smoothed_points = [];
        
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1];
                smoothed_points.append(previous * factor + point * (1 - factor));
            else:
                smoothed_points.append(point);
                
        return smoothed_points;        
        
    #---------------------------------------------------------------------------
    # Neuronova Vrstava DENSE 
    #---------------------------------------------------------------------------
    def neuralNetworkDENSEtrain(self, DataTrain):

        try:
            
            y_train_data = np.array(DataTrain.train[DataTrain.df_parm_y]);
            x_train_data = np.array(DataTrain.train[DataTrain.df_parm_x]);
            y_valid_data = np.array(DataTrain.valid[DataTrain.df_parm_y]);
            x_valid_data = np.array(DataTrain.valid[DataTrain.df_parm_x]);
        
            inp_size = len(x_train_data[0])
            out_size = len(y_train_data[0])
            
        # normalizace dat k uceni a vstupnich treninkovych dat 
            x_train = MinMaxScaler(feature_range=(0, 1))
            x_train = x_train.fit_transform(x_train_data)
            y_train = MinMaxScaler(feature_range=(0, 1))
            y_train = y_train.fit_transform(y_train_data)
        
        # normalizace dat k uceni a vstupnich validacnich dat 
            x_valid = MinMaxScaler(feature_range=(0, 1))
            x_valid = x_valid.fit_transform(x_valid_data)
            y_valid = MinMaxScaler(feature_range=(0, 1))
            y_valid = y_valid.fit_transform(y_valid_data)

            
        # neuronova sit
            model = Sequential();
            model.add(tf.keras.Input(shape=(inp_size,)));
            model.add(layers.Dense(units=inp_size,   activation=self.actf, kernel_initializer='he_normal'));
            model.add(layers.Dense(units=self.units, activation=self.actf, kernel_initializer='he_normal'));
            model.add(layers.Dense(units=self.units, activation=self.actf, kernel_initializer='he_normal'));
            model.add(layers.Dense(units=self.units, activation=self.actf, kernel_initializer='he_normal'));
#            model.add(layers.Dense(units=self.units, activation=self.actf, kernel_initializer='he_normal'));
#            model.add(layers.Dense(units=self.units, activation=self.actf, kernel_initializer='he_normal'));
            model.add(layers.Dense(out_size));
            
        # definice ztratove funkce a optimalizacniho algoritmu
            model.compile(loss='mse', optimizer='adam', metrics=['mse', 'acc'])
            
        # natrenuj model na vstupni dataset
            history = model.fit(x_train, 
                            y_train, 
                            epochs=self.epochs, 
                            batch_size=self.batch, 
                            verbose=2, 
                            validation_data=(x_valid, y_valid)
                            )
        
            model.save('./models/model_'+self.model+'_'+ DataTrain.axis, overwrite=True, include_optimizer=True)
        
        # make predictions for the input data
            return (model);
    
            
        except Exception as ex:
            traceback.print_exc();
            sys.stderr.write(str(traceback.print_exc()));
            #logger.error(traceback.print_exc());
        
    #---------------------------------------------------------------------------
    # Neuronova Vrstava DENSE predict 
    #---------------------------------------------------------------------------
    def neuralNetworkDENSEpredict(self, model, DataTrain):
        
        try:
            axis     = DataTrain.axis;  
            x_test   = np.array(DataTrain.test[DataTrain.df_parm_x]);
            y_test   = np.array(DataTrain.test[DataTrain.df_parm_y]);
            
        # normalizace vstupnich a vystupnich testovacich dat 
            x_test_scale  = MinMaxScaler(feature_range=(0, 1));
            x_test        = x_test_scale.fit_transform(x_test);
            y_test_scale  = MinMaxScaler(feature_range=(0, 1));
            y_test        = y_test_scale.fit_transform(y_test);
        
        # predict
            y_result = model.predict(x_test);
        
            x_test   = x_test_scale.inverse_transform(x_test);
            y_test   = y_test_scale.inverse_transform(y_test)

        #y_result_scale = MinMaxScaler();
            y_result = y_test_scale.inverse_transform(y_result);
            
        # plot grafu compare...
            model.summary()

            return DataFactory.DataResult(x_test, y_test, y_result, axis)

        except Exception as ex:
            sys.stderr.write("POZOR !!! patrne se neshoduji predkladana data s natrenovanym modelem\n");
            sys.stderr.write("          zkuste nejdrive --typ == train !!!\n");
            traceback.print_exc();
            sys.stderr.write(str(traceback.print_exc()));
            #logger.error(traceback.print_exc());

    #---------------------------------------------------------------------------
    # neuralNetworkDENSEexec_x 
    #---------------------------------------------------------------------------
    def neuralNetworkDENSEexec_x(self, data, graph):

        
        try:
            startTime = datetime.now();
            model_x = ''
            if self.typ == 'train':
                sys.stderr.write("Start vcetne treninku, model bude zapsan\n");
                #logger.info("Start vcetne treninku, model bude zapsan");
                model_x = self.neuralNetworkDENSEtrain(data.DataTrainDim.DataTrain);
            else:    
                sys.stderr.write("Start bez treninku - model bude nacten\n");
                #logger.info("Start bez treninku - model bude nacten");
                model_x = load_model('./models/model_'+self.model+'_'+ data.DataTrainDim.DataTrain.axis);
            
            if data.DataTrainDim.DataTrain.test is None:
                sys.stderr.write("Data pro predikci nejsou k dispozici....\n");
                #logger.info("Data pro predikci nejsou k dispozici....");
                return();
            
            data.DataResultDim.DataResultX = self.neuralNetworkDENSEpredict(model_x, data.DataTrainDim.DataTrain);
            data.saveDataResult(self.txdat1, self.model, self.typ);
            stopTime = datetime.now();
            sys.stderr.write("cas vypoctu[s] " + str(stopTime - startTime) + "\n");
            #logger.info("cas vypoctu[s] %s",  str(stopTime - startTime));

            return();

        except FileNotFoundError as e:
            sys.stderr.write(f"Nenalezen model site, zkuste nejdrive spustit s parametem train !!!\n");    
            #logger.error(f"Nenalezen model, zkuste nejdrive spustit s parametem train !!!\n" f"{e}");
        except Exception as ex:
            sys.stderr.write(str(traceback.print_exc()));
            #logger.error(traceback.print_exc());
            traceback.print_exc();
            
    #------------------------------------------------------------------------
    # neuralNetworkDENSEexec
    #------------------------------------------------------------------------
    def neuralNetworkDENSEexec(self):

        try:
            sys.stderr.write("\nPocet GPU jader: "+ str(len(tf.config.experimental.list_physical_devices('GPU')))+"\n")
        
            if self.typ == 'predict':
                self.shuffling = False;
                
            self.data.Data = self.data.getData(shuffling=self.shuffling, 
                                               timestamp_start=self.txdat1, 
                                               timestamp_stop=self.txdat2,
                                               type=self.typ);
            
            if self.data.getDfTrainData().empty and "predict" in self.typ:
                sys.stderr.write("Data pro predict, nejsou k dispozici\n");
                return(0);
                
                                                
    # Execute.....
            self.neuralNetworkDENSEexec_x(data=self.data, graph=self.graph);
            
    # archivuj vyrobeny model site            
            if self.typ == 'train':
                pass;
                #u demona nebudeme archivovat model....
                #saveModelToArchiv(model="DENSE", dest_path=self.path_to_result, data=self.data);
            return(0);

        except Exception as ex:
            traceback.print_exc();
            sys.stderr.write(str(traceback.print_exc()));
            #logger.error(traceback.print_exc());


#------------------------------------------------------------------------
# Daemon    
#------------------------------------------------------------------------
class NeuroDaemon():
    
    def __init__(self, pidfile, path_to_result, model, epochs, batch, units, shuffling, txdat1, txdat2, actf, window):
        
        self.pidfile        = pidfile; 
        self.path_to_result = path_to_result;
        self.model          = model;
        self.epochs         = epochs;
        self.batch          = batch;
        self.units          = units;
        self.shuffling      = shuffling;
        self.txdat1         = txdat1; 
        self.txdat2         = txdat2;
        self.actf           = actf;
        self.window         = window;
#------------------------------------------------------------------------
# file handlery pro file_preserve
#------------------------------------------------------------------------
    def getLogFileHandles(self,logger):
        """ Get a list of filehandle numbers from logger
            to be handed to DaemonContext.files_preserve
        """
        handles = []
        for handler in logger.handlers:
            handles.append(handler.stream.fileno())
            if logger.parent:
                handles += self.getLogFileHandles(logger.parent)
        return handles
    
#------------------------------------------------------------------------
# file handlery pro file_preserve
#------------------------------------------------------------------------
    def setLogHandler(self):
        return;
    '''
        progname = os.path.basename(__file__);

        logging.basicConfig(filename="./log/"+progname+".log",
                    filemode='a',
                    level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S');

        log_handler = logging.StreamHandler();
        logger = logging.getLogger("parent");
        logger.addHandler(log_handler)
        return(log_handler);
    '''    

        
#------------------------------------------------------------------------
# start daemon
# tovarna pro beh demona
#------------------------------------------------------------------------
    def runDaemon(self):
        
        #treninkovy beh
        neural = NeuronLayerDENSE(path_to_result = path_to_result, 
                                  typ            ="train", 
                                  model          = self.model, 
                                  epochs         = self.epochs, 
                                  batch          = self.batch,
                                  txdat1         = self.txdat1,
                                  txdat2         = self.txdat2,
                                  window         = self.window,
                                  units          = self.units,
                                  shuffling      = self.shuffling,
                                  actf           = self.actf 
                                );
        current_date =  datetime.now().strftime("%Y-%m-%d %H:%M:%S");
        sys.stderr.write("start train:"+ current_date +"\n");
        #logger.info("start train:"+ current_date);
        neural.neuralNetworkDENSEexec();

        #predikcni beh
        
        while True:
            
            sleep_interval = 10;     #10 sekund
            
                
            
            neural = NeuronLayerDENSE(path_to_result=path_to_result, 
                                      typ            ="predict", 
                                      model          = self.model, 
                                      epochs         = self.epochs, 
                                      batch          = self.batch,
                                      txdat1         = self.txdat1,
                                      txdat2         = self.txdat2,
                                      window         = self.window,
                                      units          = self.units,
                                      shuffling      = self.shuffling,
                                      actf           = self.actf 
                                    );
                                    
            if not neural.isPing():
                sleep_interval = 60; #60 sekund
                sys.stderr.write("opc ping = False, prodlouzen thread sleep_interval na 60 [s]\n");
                                    
            current_date =  datetime.now().strftime("%Y-%m-%d_%H:%M:%S");
            sys.stderr.write("start predict:"+ current_date +"\n");
            #logger.info("start predict:"+ current_date);
            neural.neuralNetworkDENSEexec();
            time.sleep(sleep_interval);
        
    #------------------------------------------------------------------------
    # info daemon
    #------------------------------------------------------------------------
    def info(self):
        sys.stderr.write("daemon pro sledovani a kompenzaci teplotnich zmen stroje\n");
        return;
    #------------------------------------------------------------------------
    # daemonize - zduchovateni....
    #    do the UNIX double-fork magic, see Stevens' "Advanced
    #    Programming in the UNIX Environment" for details (ISBN 0201563177)
    #    http://www.erlenstar.demon.co.uk/unix/faq_2.html#SEC16
    #
    #------------------------------------------------------------------------
    def daemonize(self):
        
        sys.stderr.write("daemonize.....\n");

        l_handler = self.setLogHandler();
        
        context = daemon.DaemonContext(working_directory='./',
                                       pidfile=lockfile.FileLock(self.pidfile),
                                       stdout=sys.stdout,
                                       stderr=sys.stderr,
                                       umask=0o002,
                                       files_preserve = [l_handler]
                  );

        context.signal_map = {
            #signal.SIGKILL:  self.stop(),
            #signal.SIGINT:  self.stop(),
            signal.SIGHUP:  'terminate',
            #signal.SIGUSR1: reload_program_config,
        }
        
        return(context);

    #------------------------------------------------------------------------
    # start daemon
    #------------------------------------------------------------------------
    def start(self):
        # Kontrola existence pid - daemon run...
        try:                                                                                                                
            pf = open(self.pidfile,'r');                                                                                    
            pid = int(pf.read().strip());                                                                                   
            pf.close();                                                                                                     
        except:                                                                                                             
            pid = None;                                                                                                     
                                                                                                                            
        if pid:                                                                                                             
            message = "pid procesu %d existuje!!!. Daemon patrne bezi - exit(1)\n";                                         
            sys.stderr.write(message % pid);                                                                       
            os._exit(1);                                                                                                    

        context = self.daemonize();
        
        try:
            with context:
                self.runDaemon();
        except (Exception, getopt.GetoptError)  as ex:
            traceback.print_exc();
            sys.stderr.write(+str(traceback.print_exc()));
            #logger.error(traceback.print_exc());
            help(activations);

    #------------------------------------------------------------------------
    # start daemon
    #------------------------------------------------------------------------
    def stop(self):
        try:                                                                                                                
            pf = open(self.pidfile,'r');                                                                                    
            pid = int(pf.read().strip());                                                                                   
            pf.close();                                                                                                     
        except:                                                                                                             
            pid = None;                                                                                                     
                                                                                                                            
        if pid is None:
            return;                                                                                                             
            #message = "pid procesu neexistuje!!!. Daemon patrne nebezi - exit(1)\n";                                         
            #sys.stderr.write(message);                                                                       
            #os._exit(1);
        else:                                                                                                        
            message = "pid procesu %d existuje!!!. Daemon %d stop....\n";                                         
            sys.stderr.write(message % pid);                                                                       
            os._exit(0);

            
    

#------------------------------------------------------------------------
# MAIN CLASS
#------------------------------------------------------------------------

#------------------------------------------------------------------------
# saveModelToArchiv - zaloha modelu, spusteno jen pri parametru train
#------------------------------------------------------------------------
def saveModelToArchiv(model, dest_path, data):

    axes = np.array([data.DataTrainDim.DataTrain.axis]);
    src_dir  = "./models/model_"+model+"_";
    dest_dir = "/models/model_"+model+"_";
    try:
        if data.DataTrainDim.DataTrain  == None:
            pass;
        else:    
            src_dir_  = src_dir + axes[0]
            dest_dir_ = dest_path + dest_dir + axes[0]
            files = os.listdir(src_dir_)
            shutil.copytree(src_dir_, dest_dir_)
            
        return(0);    
   
    except Exception as ex:
        traceback.print_exc();
        sys.stderr.write(str(traceback.print_exc()));
        #logger.error(traceback.print_exc());
 


#------------------------------------------------------------------------
# setEnv
#------------------------------------------------------------------------
def setEnv(path, model, type):

        progname = os.path.basename(__file__);
        current_date =  datetime.now().strftime("%Y-%m-%d_%H:%M:%S");
        path1 = path+model+"_3D";
        path2 = path1+"/"+current_date+"_"+type
                
        
        try: 
            os.mkdir("./log");
        except OSError as error: 
            pass;
        
        try: 
            os.mkdir("./run");
        except OSError as error: 
            pass; 
 

        try: 
            os.mkdir("./result")
        except OSError as error: 
            pass; 

        try: 
            os.mkdir(path1);
        except OSError as error: 
            pass; 
        
        try: 
            os.mkdir(path2);
        except OSError as error: 
            pass; 
            
        try: 
            os.mkdir(path2+"/src");
        except OSError as error: 
            pass; 
            
        try: 
            os.mkdir("./models");
        except OSError as error: 
            pass; 

        try:
            shutil.copy(progname, path2+"/src");
        except shutil.SpecialFileError as error:
            print("Chyba pri kopii zdrojoveho kodu.", error)
        except:
            print("Chyba pri kopii zdrojoveho kodu.")

        try:
            shutil.copy("ai-parms.txt", path2+"/src");
        except shutil.SpecialFileError as error:
            print("Chyba pri kopii ai-parms.txt.", error)
        except:
            print("Chyba pri kopii ai-parms.txt.")
            
        #logging.basicConfig(filename='./log/'+progname+'.log',
        #    filemode='a',level=logging.INFO,
        #    format='%(asctime)s - %(message)s',
        #    datefmt='%Y-%m-%d %H:%M:%S');
            
        #logger = logging.getLogger()
        #logger.setLevel(logging.DEBUG)
        #fh = logging.FileHandler("./foo.log")
        #logger.addHandler(fh)
        
        #file_logger = logging.FileHandler("/tmp/aaa.log", "w")
        #logger = logging.getLogger()
        #logger.addHandler(file_logger)
        #logger.setLevel(logging.INFO)
        #with daemon.DaemonContext(files_preserve=[file_logger.stream.fileno()]):
        #    while True:
        #        logger.info(datetime.now())
        #        sleep(1)

        return path2    

#------------------------------------------------------------------------
# Exception handler
#------------------------------------------------------------------------
def exception_handler(exctype, value, tb):
    pass;
    #logger.error(exctype)
    #logger.error(value)
    #logger.error(traceback.extract_tb(tb))

#------------------------------------------------------------------------
# Help
#------------------------------------------------------------------------
def help (activations):
    print ("HELP:");
    print ("------------------------------------------------------------------------------------------------------ ");
    print ("pouziti: <nazev_programu> <arg1> <arg2> <arg3> <arg4>");
    print ("ai-daemon.py -t1 <--txdat1> -t2 <--txdat2> ")
    print (" ");
    print ("        --help            list help ")
    print ("        --txdat1          timestamp zacatku datove mnoziny pro train, napr '2022-04-09 08:00:00' ")
    print (" ");
    print ("        --txdat2          timestamp konce   datove mnoziny pro train, napr '2022-04-09 12:00:00' ")
    print (" ");
    print ("                                 parametry txdat1, txdat2 jsou nepovinne. Pokud nejsou uvedeny, bere");
    print ("                                 se v uvahu cela mnozina dat k trenovani.");
    print (" ");
    print (" ");
    print (" ");
    print ("POZOR! typ behu 'train' muze trvat nekolik hodin, zejmena u typu site LSTM, GRU nebo BIDI!!!");
    print ("       pricemz 'train' je povinny pri prvnim behu site. V rezimu 'train' se zapise ");
    print ("       natrenovany model site..");
    print ("       V normalnim provozu natrenovane site doporucuji pouzit parametr 'predict' ktery.");
    print ("       spusti normalni beh site z jiz natrenovaneho modelu.");
    print ("       Takze: budte trpelivi...");
    print (" ");
    print (" ");
    print (" ");
    print ("Pokud pozadujete zmenu parametu j emozno primo v programu poeditovat tyto promenne ");
    print (" ");
    print ("a nebo vyrobit soubor ai-parms.txt s touto syntaxi ");
    print ("  #Vystupni list parametru - co budeme chtit po siti predikovat");
    print ("  df_parmx = machinedata_m0412,teplota_pr01,x_temperature'");
    print (" ");
    print ("  #Tenzor predlozeny k uceni site");
    print ("  df_parmX = machinedata_m0412,teplota_pr01, x_temperature");
    print (" ");
    print ("a ten nasledne ulozit v rootu aplikace. (tam kde je pythonovsky zdrojak. ");
    print ("POZOR!!! nazvy promennych se MUSI shodovat s hlavickovymi nazvy vstupniho datoveho CSV souboru (nebo souboruuu)");
    print ("a muzou tam byt i uvozovky: priklad: 'machinedata_m0112','machinedata_m0212', to pro snazsi copy a paste ");
    print ("z datoveho CSV souboru. ");
    print (" ");
    print ("(C) GNU General Public License, autor Petr Lukasik , 2022 ");
    print (" ");
    print ("Prerekvizity: linux Debian-11 nebo Ubuntu-20.04, (Windows se pokud mozno vyhnete)");
    print ("              miniconda3,");
    print ("              python 3.9, tensorflow 2.8, mathplotlib,  ");
    print ("              tensorflow 2.8,");
    print ("              mathplotlib,  ");
    print ("              scikit-learn-intelex,  ");
    print ("              pandas,  ");
    print ("              numpy,  ");
    print ("              keras   ");
    print (" ");
    print (" ");
    print ("Povolene aktivacni funkce: ");
    print(tabulate(activations, headers=['Akt. funkce', 'Popis']));

    return();

#------------------------------------------------------------------------
# kontrola zda byla zadana platna aktivacni funkce 
# ze seznamu activations...
#------------------------------------------------------------------------
def checkActf(actf, activations):
    

    for i in activations:
        if i[0] in actf:
            return(True);

    return(False);




#------------------------------------------------------------------------
# main
# preddefinovane hyperparametry neuronove site
# ["typ=", "model=", "epochs=", "batch=", "units=", "shuffle=","actf=", "txdat1=","txdat2=", "help="])
#   parm0  = sys.argv[0];   - nazev programu
#   typ    = "train";       - typ behu site <train, predict>
#   model  = "DENSE";       - model site
#   epochs = 128;           - pocet treninkovych metod
#   batch  = 128;           - velikost davky
#   units  = 512;           - pocet neuronu
#   txdat1 = "";            - timestamp start - vyber dat treninkove a validacni mnoziny
#   txdat2 = "";            - timestamp stop  - vyber dat treninkove a validacni mnoziny
#   shuffling = False;      - promichat nahodne data <True, False>
#   actf = "tanh";          - aktivacni funkce
#   pid = 0;
#------------------------------------------------------------------------
def main(argv):
    
    global path_to_result;
    path_to_result = "./result";
    pidfile = "./run/ai-daemon.pid"
    
    global g_window;
    g_window = 48;
    
    parm0  = sys.argv[0];        
    model  = "DENSE";
    epochs = 128;
    batch  = 128;
    units  = 128;
    txdat1 = "2022-02-15 00:00:00";
    txdat2 = datetime.now().strftime("%Y-%m-%d %H:%M:%S");
    shuffling = False;
    actf = "tanh";
    pid = 0;
    status = "";
    startTime = datetime.now();
    type = "train";
    
        
    activations = [["deserialize", "Returns activation function given a string identifier"],
                   ["elu", "Exponential Linear Unit"],
                   ["exponential", "Exponential activation function"],
                   ["gelu", "Gaussian error linear unit (GELU) activation function"],
                   ["get", "Returns function"],
                   ["hard_sigmoid", "Hard sigmoid activation function"],
                   ["linear", "Linear activation function (pass-through)"],
                   ["relu", "Rectified linear unit activation function"],
                   ["selu","Scaled Exponential Linear Unit"],
                   ["serialize","Returns the string identifier of an activation function"],
                   ["sigmoid","Sigmoid activation function: sigmoid(x) = 1 / (1 + exp(-x))"],
                   ["softmax","Softmax converts a vector of values to a probability distribution"],
                   ["softplus","Softplus activation function: softplus(x) = log(exp(x) + 1)"],
                   ["softsign","Softsign activation function: softsign(x) = x / (abs(x) + 1)"],
                   ["swish","Swish activation function: swish(x) = x * sigmoid(x)"],
                   ["tanh","Hyperbolic tangent activation function"]];


        #init objektu daemona
    path_to_result = setEnv(path=path_to_result, model=model, type=type);
        
    daemon_ = NeuroDaemon(pidfile        = pidfile,
                         path_to_result = path_to_result,
                         model          = model,
                         epochs         = epochs,
                         batch          = batch,
                         units          = units,
                         shuffling      = shuffling,
                         txdat1         = txdat1, 
                         txdat2         = txdat2,
                         actf           = actf,
                         window         = g_window
                );
       
    daemon_.info();

    try:
        sys.stderr.write("start...\n");
        #logger.info("start...");

        #kontrola platne aktivacni funkce        
        if not checkActf(actf, activations):
            print("Chybna aktivacni funkce - viz help...");
            help(activations)
            sys.exit(1);
            
        #logger.info("Verze TensorFlow :" + tf.__version__);
        print("Verze TensorFlow :", tf.__version__);
        
        txdat_format = "%Y-%m-%d %h:%m:%s"
        try:
            opts, args = getopt.getopt(sys.argv[1:],"hs:t1:t2:h:x",["status=","txdat1=","txdat2=", "help="])
        except getopt.GetoptError:
            print("Chyba pri parsovani parametru:");
            help(activations);
            
        for opt, arg in opts:
            
            if opt in ["-s","--status"]:
                status = arg;
                if "start" in status or "stop" in status or "restart" in status or "status" in status:
                    pass;
                else:
                    print("Chyba stavu demona povoleny jen <start, stop, restart a status");
                    help(activations);
                    sys.exit(1);    

            elif opt in ["-t1","--txdat1"]:
                txdat1 = arg;
                if txdat1:
                    try:
                        res = bool(parser.parse(txdat1));
                    except ValueError:
                        print("Chyba formatu txdat1, musi byt YYYY-MM-DD HH:MM:SS");
                        help(activations);
                        sys.exit(1);    
        
            elif opt in ["-t2","--txdat2"]:
                txdat2 = arg;
                if txdat2:
                    try:
                        res = bool(parser.parse(txdat2));
                    except ValueError:
                        print("Chyba formatu txdat2, musi byt YYYY-MM-DD HH:MM:SS");
                        help(activations);
                        sys.exit(1);    
        
         
            elif opt in ["-h","--help"]:
                help(activations);
                sys.exit(0);
        
        if len(sys.argv) < 2:
            help(activations);
            sys.exit(1);
            
        #-----------------------------------------------------------------------------
        #obsluha demona
        #-----------------------------------------------------------------------------
        if 'start' in status:
            try:
                sys.stderr.write("ai-daemon start....\n");
                daemon_.start();
                #daemon_.runDaemon();
            except:
                traceback.print_exc();
                sys.stderr.write(str(traceback.print_exc()));
                sys.stderr.write("ai-daemon start exception...\n");
                pass
            
        elif 'stop' in status:
            sys.stderr.write("ai-daemon stop...\n");
            daemon_.stop();
            
        elif 'restart' in status:
            sys.stderr.write("ai-daemon restart...\n");
            daemon_.restart()
            
        elif 'status' in status:
            try:
                pf = file(PIDFILE,'r');
                pid = int(pf.read().strip())
                pf.close();
            except IOError:
                pid = None;
            except SystemExit:
                pid = None;
            if pid:
                sys.stderr.write("Daemon ai-daemon je ve stavu run...\n");
            else:
                sys.stderr.write("Daemon ai-daemon je ve stavu stop....\n");
        else:
            sys.stderr.write("Neznamy parametr:<"+status+">");
            sys.exit(0)
        
    except (Exception, getopt.GetoptError)  as ex:
        traceback.print_exc();
        #logger.error(traceback.print_exc());
        help(activations);
        
    finally:    
        stopTime = datetime.now();
        #print("cas vypoctu [s]", stopTime - startTime );
        #logger.info("cas vypoctu [s] %s",  str(stopTime - startTime));
        sys.stderr.write("\nstop obsluzneho programu pro demona - ai-daemon...\n");
        #logger.info("stop obsluzneho programu pro demona - ai-daemon...");
        sys.exit(0);




#------------------------------------------------------------------------
# main entry point
#------------------------------------------------------------------------
        
if __name__ == "__main__":

    main(sys.argv[1:])
    

